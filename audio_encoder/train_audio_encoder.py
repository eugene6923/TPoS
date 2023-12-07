import torch
import clip
import random
import argparse
import torch.optim as optim
from datasets_final import VggsoundCurationTestDataset,VggsoundCurationDataset
from model_final import Mapping_Model, Audio_Encoder, FrozenCLIPEmbedder, copyStateDict
import torch.nn.functional as F
import torch.nn as nn
import math
import time
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoTokenizer, CLIPTextModel


parser = argparse.ArgumentParser(description="Audio Text Clip Implementation")

parser.add_argument("--epochs", default=50, type=int,
                help="epochs of training")
parser.add_argument("--batch_size", default=150, type=int,
                help="batch size of training")
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.8685, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--step_size', default=1, type=float,
                    help='Step size for SGD')
parser.add_argument('--num_workers', default=16, type=int,
                    help='Number of workers used in dataloading')        

args = parser.parse_args()

os.makedirs("../pretrained_models/",exist_ok=True)

if __name__ == "__main__":
    random.seed(42)
    vggsound_dataset = VggsoundCurationDataset()
    print(f"trainset length: {vggsound_dataset.__len__()}")
    vggsound_test_dataset = VggsoundCurationTestDataset()
    print(f"testset length: {vggsound_test_dataset.__len__()}")

    train_dataset=vggsound_dataset
    validation_dataset=vggsound_test_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ngpu=len(device)
    clip_model, _ = clip.load("ViT-L/14", device=device)

    model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True)

    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True)

    audioencoder = Audio_Encoder(batch_size = args.batch_size, ngpus = torch.cuda.device_count())
    audioencoder=nn.DataParallel(audioencoder).to(device)
    map_model = Mapping_Model()
    map_model = nn.DataParallel(map_model).to(device)
    mse_loss = torch.nn.MSELoss()
    clip_768 = FrozenCLIPEmbedder()
    optimizer = optim.SGD(audioencoder.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    map_optimizer = optim.Adam(map_model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1, step_size_up=5, mode="triangular")
    ce = torch.nn.CrossEntropyLoss()
    min_validation_loss_value = 50000

    for epoch in range(args.epochs):
        start = time.time()
        train_loss_value, validation_loss_value = 0, 0
        audioencoder.train()
        map_model.train()
        result_loss = 0

        for idx, (batch_audio, batch_audio_aug, batch_text) in enumerate(train_dataloader):
            audio_embedding1, audio_embedding2, beta_t = audioencoder(batch_audio.cuda())
            audio_embedding_aug1, audio_embedding_aug2, beta_aug_t  = audioencoder(batch_audio_aug.cuda())

            text_tokens1 = torch.cat([clip.tokenize(text) for text in batch_text])

            with torch.no_grad():
                clip_768_data = torch.cat([clip_768(text) for text in batch_text])
                text_embedding1 = clip_model.encode_text(text_tokens1.to(device)).float()
                text_embedding1 = text_embedding1 / text_embedding1.norm(dim=-1, keepdim=True)

            optimizer.zero_grad()
            map_optimizer.zero_grad()

            audio_embedding2_sum = torch.sum(audio_embedding2,dim=1)
            audio_embedding_aug2_sum = torch.sum(audio_embedding_aug2,dim=1)
            map_result = map_model(audio_embedding2_sum.clone().unsqueeze(1))

            loss = 0
            loss_list = []

            projection_audio_text1 = (audio_embedding1 @ text_embedding1.T) * math.exp(0.07)
            projection_audio_text2 = (audio_embedding2_sum @ text_embedding1.T) * math.exp(0.07)
            projection_self_audio1 = (audio_embedding_aug1 @ audio_embedding_aug1.T) * math.exp(0.07)
            projection_self_audio2 = (audio_embedding_aug2_sum @ audio_embedding_aug2_sum.T) * math.exp(0.07)

            label = torch.arange(args.batch_size, dtype=torch.long).cuda()

            audio_contrastive_loss1 = ce(projection_audio_text1, label) + ce(projection_audio_text1.T, label)
            audio_contrastive_loss2 = ce(projection_audio_text2, label) + ce(projection_audio_text2.T, label)
            self_contrastive_loss1 = ce(projection_self_audio1, label) + ce(projection_self_audio1.T, label)
            self_contrastive_loss2 = ce(projection_self_audio2, label) + ce(projection_self_audio2.T, label)
            loss += (audio_contrastive_loss1 + audio_contrastive_loss2) / 4
            loss += (self_contrastive_loss1 + self_contrastive_loss1) / 6

            result_loss = mse_loss(map_result, clip_768_data[:,1:])
            loss += result_loss

            loss_list.append(audio_contrastive_loss1.item()/4)
            loss_list.append(audio_contrastive_loss2.item()/4)
            loss_list.append(self_contrastive_loss1.item()/6)
            loss_list.append(self_contrastive_loss2.item()/6)
            loss_list.append(result_loss.item())

            loss.backward()
            
            optimizer.step()
            map_optimizer.step()
            
            train_loss_value += loss.item()

            

            if idx % 100 == 0:
                print("VGG, Batch : {:3d} , total loss : {:.3f}, ".format(idx, loss.item()))
                # print(f"beta_t : {beta_t[0:5]}")

                for i,loss_value in enumerate(loss_list):
                    print(f"loss_{i} : {loss_value:.6f}")
        scheduler.step()

        audioencoder.eval()
        map_model.eval()
        
        
        print("Validation !")

        for idx, (batch_audio, batch_audio_aug, batch_text) in enumerate(validation_dataloader):
            
            with torch.no_grad():
                
                audio_embedding1, audio_embedding2, beta_t = audioencoder(batch_audio.cuda())
                audio_embedding_aug1, audio_embedding_aug2, beta_aug_t  = audioencoder(batch_audio_aug.cuda()) # 여기서 에러

                text_tokens1 = torch.cat([clip.tokenize(text) for text in batch_text])

                clip_768_data = torch.cat([clip_768(text) for text in batch_text])
                text_embedding1 = clip_model.encode_text(text_tokens1.to(device)).float()
                text_embedding1 = text_embedding1 / text_embedding1.norm(dim=-1, keepdim=True)

                optimizer.zero_grad()
                map_optimizer.zero_grad()

                audio_embedding2_sum = torch.sum(audio_embedding2,dim=1)
                audio_embedding_aug2_sum = torch.sum(audio_embedding_aug2,dim=1)
                map_result = map_model(audio_embedding2_sum.clone().unsqueeze(1))
                torch.autograd.set_detect_anomaly(True)
                loss = 0
                loss_list = []

                projection_audio_text1 = (audio_embedding1 @ text_embedding1.T) * math.exp(0.07)
                projection_audio_text2 = (audio_embedding2_sum @ text_embedding1.T) * math.exp(0.07)
                projection_self_audio1 = (audio_embedding_aug1 @ audio_embedding_aug1.T) * math.exp(0.07)
                projection_self_audio2 = (audio_embedding_aug2_sum @ audio_embedding_aug2_sum.T) * math.exp(0.07)

                label = torch.arange(args.batch_size, dtype=torch.long).cuda()

                audio_contrastive_loss1 = ce(projection_audio_text1, label) + ce(projection_audio_text1.T, label)
                audio_contrastive_loss2 = ce(projection_audio_text2, label) + ce(projection_audio_text2.T, label)
                self_contrastive_loss1 = ce(projection_self_audio1, label) + ce(projection_self_audio1.T, label)
                self_contrastive_loss2 = ce(projection_self_audio2, label) + ce(projection_self_audio2.T, label)
                loss += (audio_contrastive_loss1 + audio_contrastive_loss2) / 4
                loss += (self_contrastive_loss1 + self_contrastive_loss1) / 6

                result_loss = mse_loss(map_result, clip_768_data[:,1:])
                loss += result_loss

                loss_list.append(audio_contrastive_loss1.item()/4)
                loss_list.append(audio_contrastive_loss2.item()/4)
                loss_list.append(self_contrastive_loss1.item()/6)
                loss_list.append(self_contrastive_loss2.item()/6)
                loss_list.append(result_loss.item())
                                        
            validation_loss_value += loss.item()
            if idx % 100 == 0:
                print("VGG, Batch : {:3d} , total loss : {:.3f}".format(idx, loss.item()))
        
        print("Epoch : {:2d} , train loss : {:.5f}, validation loss : {:.5f}, Time : {}".format(epoch, train_loss_value / len(train_dataloader), validation_loss_value / len(validation_dataloader), time.time() - start))
        with open("../pretrained_models/LSTM_loss.txt", "a") as f:
                    f.write("\n\nEpoch : {:2d} , train loss : {:.5f}, validation loss : {:.5f}, Time : {}".format(epoch, train_loss_value / len(train_dataloader), validation_loss_value / len(validation_dataloader), time.time() - start))
        
        if min_validation_loss_value > validation_loss_value:
            save_path = "../pretrained_models/audio_encoder_" + str(epoch) + ".pth"
            torch.save(audioencoder.state_dict(), save_path)
            save_path2 = "../pretrained_models/map_model_" + str(epoch) + ".pth"
            torch.save(map_model.state_dict(), save_path2)
            min_validation_loss_value = validation_loss_value
        
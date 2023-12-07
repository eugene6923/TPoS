import os
import random
from collections import OrderedDict
import numpy as np
import librosa

import torch
import torch.nn as nn
from torch.autograd import Variable
import timm

def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)


def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)

class Audio_generator(object):
    def __init__(self,condition,num_frames,file_name):
        self.condition = condition
        self.num_frames =num_frames
        self.file_name = file_name
    def __enter__(self):
        preprocessed_audio, volume_mean = audio_preprocess(self.condition,self.num_frames, self.file_name)
        return preprocessed_audio, volume_mean
       
    def __exit__(self, *args, **kwargs) :

        os.remove(f"./{self.file_name}.pt")
        os.remove(f"./{self.file_name}_volume.pt")


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


class AudioEncoder(torch.nn.Module):
    def __init__(self, backbone_name="resnet18"):
        super(AudioEncoder, self).__init__()
        self.backbone_name = backbone_name
        self.conv = torch.nn.Conv2d(1, 3, (3, 3))
        self.feature_extractor = timm.create_model(self.backbone_name, num_classes=768, pretrained=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.feature_extractor(x)
        return x

def audio_preprocess(audio_path,num_frames, file_name) : 
    import os
    memo_path = f"./{file_name}.pt"
    memo_path2 = f"./{file_name}_volume.pt"
    if os.path.exists(memo_path) or os.path.exists(memo_path2):
        return torch.load(memo_path), torch.load(memo_path2)

    else:
        preprocessed_data, volume_mean = _preprocess(audio_path, num_frames)
        torch.save(preprocessed_data, memo_path)
        torch.save(volume_mean, memo_path2)
        return preprocessed_data, volume_mean


import cv2

def _preprocess(audio_path, num_frames):
    y, sr = librosa.load(audio_path, sr=44100)

    hop_length = 512
    n_fft = 1024

    stft = librosa.stft(y,n_fft=n_fft, hop_length=hop_length)

    magnitude = np.abs(stft)

    log_spectrogram = librosa.amplitude_to_db(magnitude) / 80.0 + 1

    audio_inputs = np.array([log_spectrogram])

    audio_inputs = audio_inputs[:,:,:8640]
    input_size =513
    time_length = 864
    width_resolution = 768
    c, h, w = audio_inputs.shape
    n_mels = 128

    if w >= time_length:
        j = random.randint(0, w-time_length)
        audio_inputs = audio_inputs[:,:,j:j+time_length]
    elif w < time_length:
        zero = np.zeros((1, input_size, time_length))
        j = random.randint(0, time_length - w - 1)
        zero[:,:,j:j+w] = audio_inputs[:,:,:w]
        audio_inputs = zero


    audio_resize = audio_inputs[0,:input_size,:width_resolution]

    w = audio_resize.shape[1]
    audio_inputs = cv2.resize(audio_resize.squeeze(), (width_resolution, n_mels))
    audio_inputs = audio_inputs.reshape(-1,n_mels,width_resolution)
    audio_per_frame = []
    audio_volume_mean = []
    inter_frame = w // num_frames
    frame_per_audio = time_length // num_frames

    for idx_audio in range(num_frames):
        audio_seg = audio_inputs[:,:,idx_audio*inter_frame:idx_audio*inter_frame+inter_frame]
        _,_,seg_w = audio_seg.shape
        if seg_w >= frame_per_audio:
            j = random.randint(0, seg_w-frame_per_audio)
            audio_seg = audio_seg[:,:,j:j+frame_per_audio]
        elif seg_w < frame_per_audio:
            zero = np.zeros((1, n_mels, frame_per_audio))
            j = random.randint(0, frame_per_audio - seg_w - 1)
            zero[:,:,j:j+seg_w] = audio_seg[:,:,:seg_w]
            audio_seg = zero

        audio_seg = audio_seg[0,:n_mels,:inter_frame]
        audio_volume_mean.append(np.mean(audio_seg))

        audio_per_frame.append(audio_seg)

    audio_per_frame=np.array(audio_per_frame)

    audio_per_frame = torch.from_numpy(audio_per_frame).float()
    

    return audio_per_frame, audio_volume_mean




class map_model(nn.Module):
    def __init__(self, max_length=77):
        super().__init__()
        self.max_length = max_length-1
        self.linear1 = torch.nn.Linear(768,self.max_length//7*768)
        self.linear2 = torch.nn.Linear(self.max_length//7*768,self.max_length*768)
        self.act = torch.nn.GELU()
        self.drop = torch.nn.Dropout(0.2)
        
    def forward(self, x):
        return self.act(self.drop(self.linear2(self.act(self.drop(self.linear1(x)))))).reshape(x.shape[0],self.max_length,768)




class LSTMModel(nn.Module):
    def __init__(self, sequence_length=5, lstm_hidden_dim=768, input_size=768, hidden_size=768, num_layers=1,backbone_name="resnet18",batch_size=1, ngpus = 4):

        super(LSTMModel,self).__init__()

        self.sequence_length = sequence_length
        self.lstm_hidden_dim=lstm_hidden_dim
        
        self.T_A = nn.Linear(sequence_length*lstm_hidden_dim, 512)
        self.T_A2 = nn.Linear(self.sequence_length*lstm_hidden_dim, self.sequence_length*512)

        # activate functions
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.backbone_name = backbone_name
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.conv = torch.nn.Conv2d(1, 3, (3, 3))
        self.conv2 = torch.nn.Conv2d(1,77,(1,1)) 
        self.feature_extractor = timm.create_model(self.backbone_name, num_classes=self.input_size, pretrained=True)
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,num_layers=num_layers, batch_first=True)
        self.ngpus=ngpus
        self.batch_size=batch_size
        self.size=1
        self.cnn = nn.Conv1d(768,1, kernel_size=1)

    def forward (self,x):
        a=torch.zeros(self.size,self.sequence_length,768).cuda()
        for i in range(self.sequence_length):
            a[:,i,:] = self.feature_extractor(self.conv(x[i,:,:].reshape(self.size,1,128,self.hidden_size//self.sequence_length)))
        x=a
        h_0 = Variable(torch.zeros( self.num_layers,x.size(0), self.hidden_size)).cuda()
        c_0 = Variable(torch.zeros( self.num_layers,x.size(0),  self.hidden_size)).cuda()
        self.lstm.flatten_parameters()
        output, (hn, cn) = self.lstm(x, (h_0, c_0))

        output = output/output.norm(dim=-1,keepdim=True)
        
        output_permute = output.permute(0,2,1)

        beta_t = self.cnn(output_permute).squeeze()


        beta_t=self.softmax(beta_t)

        out=output[:,0,:].mul(beta_t[0].reshape(self.size,-1))
        out=out.unsqueeze(1)

        for i in range(1,self.sequence_length):
            next_z=output[:,i,:].mul(beta_t[i].reshape(self.size,-1))
            out=torch.cat([out,next_z.unsqueeze(1)],dim=1)

        return output, out, beta_t

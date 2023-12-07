import torch 
import timm
from collections import OrderedDict
import torch.nn as nn
from torch.autograd import Variable

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


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError

from transformers import CLIPTokenizer, CLIPTextModel

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77):
        super().__init__()
        self.device = device
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.max_length = max_length
        self.layernorm = nn.LayerNorm(max_length)
        self.to(device).eval()
        self.train = disabled_train
        self.freeze()
    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False
    def forward(self, text, option=None):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens) # 여기서 layernorm 돌리고 eof 젤 큰걸로하는듯? layernorm안해서 그럼 어카지?
        z = outputs.last_hidden_state
        if option == None: return z
        else: return tokens.argmax(dim=-1).item()
    def encode(self, text, option=None):
        return self(text, option)

class Mapping_Model(nn.Module):
    def __init__(self, max_length=77):
        super().__init__()
        self.max_length = max_length-1
        self.linear1 = torch.nn.Linear(768,self.max_length//7*768)
        self.linear2 = torch.nn.Linear(self.max_length//7*768,self.max_length*768)
        self.act = torch.nn.GELU()
        self.drop = torch.nn.Dropout(0.2)
        
    def forward(self, x):
        return self.act(self.drop(self.linear2(self.act(self.drop(self.linear1(x)))))).reshape(x.shape[0],self.max_length,768)



class Audio_Encoder(nn.Module):
    
    def __init__(self, sequence_length=5, lstm_hidden_dim=768, input_size=768, hidden_size=768, num_layers=1,backbone_name="resnet18",batch_size=320, ngpus = 4):

        super(Audio_Encoder,self).__init__()

        self.sequence_length = sequence_length
        self.lstm_hidden_dim=lstm_hidden_dim
        
        
        self.T_A = nn.Linear(sequence_length*lstm_hidden_dim, 512)
        self.T_A2 = nn.Linear(self.sequence_length*lstm_hidden_dim, self.sequence_length*512)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
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
        self.size=int(self.batch_size / self.ngpus)
    
        self.cnn = nn.Conv1d(768,1, kernel_size=1)

    def forward (self,x):

        a=torch.zeros(self.size,self.sequence_length,768).cuda()
        for i in range(self.sequence_length):
            a[:,i,:] = self.feature_extractor(self.conv(x[:,i,:,:].reshape(self.size,1,128,self.hidden_size//self.sequence_length)))
        x=a
        h_0 = Variable(torch.zeros( self.num_layers,x.size(0), self.hidden_size)).cuda()
        c_0 = Variable(torch.zeros( self.num_layers,x.size(0),  self.hidden_size)).cuda()
        self.lstm.flatten_parameters()
        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        output = output/output.norm(dim=-1,keepdim=True)
        
        output_permute = output.permute(0,2,1)

        beta_t = self.cnn(output_permute).squeeze()

        beta_t=self.softmax(beta_t)

        out=output[:,0,:].mul(beta_t[:,0].reshape(self.size,-1))

        out=out.unsqueeze(1)


        for i in range(1,self.sequence_length):
            next_z=output[:,i,:].mul(beta_t[:,i].reshape(self.size,-1) )
            out=torch.cat([out,next_z.unsqueeze(1)],dim=1)

        return output[:,-1,:], out, beta_t


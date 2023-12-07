from torch.utils.data.dataset import Dataset
from glob import glob
import numpy as np
import torch 
import random
from textaugment import EDA
import nltk

nltk.download("stopwords")
nltk.download("wordnet")

class VggsoundCurationDataset(Dataset):
    def __init__(self):
        self.audio_lists = glob("./landscape_curation/*.npy") + glob("./vggsound_curation/*.npy") # Put postprocessed audio files here
        self.time_length = 864
        self.n_mels = 128
        self.num_frames = 5
        self.text_aug = EDA()
        self.width_resolution = 768 
        self.frame_per_audio = self.time_length // self.num_frames

    def __getitem__(self, idx):
        wav_name = self.audio_lists[idx]
            
        audio_inputs = np.load(wav_name, allow_pickle=True)

        text_prompt = wav_name.split("/")[-1].split("_")[0]
        c, h, w = audio_inputs.shape

        if w >= self.time_length:
            j = random.randint(0, w-self.time_length)
            audio_inputs = audio_inputs[:,:,j:j+self.time_length]
        elif w < self.time_length:
            zero = np.zeros((1, self.n_mels, self.time_length))
            j = random.randint(0, self.time_length - w - 1)
            zero[:,:,j:j+w] = audio_inputs[:,:,:w]
            audio_inputs = zero

        audio_resize = audio_inputs[0,:self.n_mels,:self.width_resolution]

        w = audio_resize.shape[1]
        audio_inputs = audio_resize.reshape(-1,self.n_mels,self.width_resolution)
        
        audio_per_frame = []
        audio_per_frame_aug = []
        inter_frame = w // self.num_frames

        for idx_audio in range(self.num_frames):
            audio_seg = audio_inputs[:,:,idx_audio*inter_frame:idx_audio*inter_frame+inter_frame]
            _,_,seg_w = audio_seg.shape
            if seg_w >= self.frame_per_audio:
                j = random.randint(0, seg_w-self.frame_per_audio)
                audio_seg = audio_seg[:,:,j:j+self.frame_per_audio]
            elif seg_w < self.frame_per_audio:
                zero = np.zeros((1, self.n_mels, self.frame_per_audio))
                j = random.randint(0, self.frame_per_audio - seg_w - 1)
                zero[:,:,j:j+seg_w] = audio_seg[:,:,:seg_w]
                audio_seg = zero

            audio_seg = audio_seg[0,:self.n_mels,:inter_frame]

            audio_aug = self.spec_augment(audio_seg)

            audio_per_frame.append(audio_seg)

            audio_per_frame_aug.append(audio_aug)

        audio_per_frame=np.array(audio_per_frame)
        audio_per_frame_aug=np.array(audio_per_frame_aug)

        audio_per_frame = torch.from_numpy(audio_per_frame).float()
        audio_per_frame_aug = torch.from_numpy(audio_per_frame_aug).float()

            
        return audio_per_frame, audio_per_frame_aug, text_prompt

    def spec_augment(self, spec, num_mask=2, freq_masking_max_percentage=0.15, time_masking_max_percentage=0.3):
        spec = spec.copy()
        for i in range(num_mask):
            all_frames_num, all_freqs_num = spec.shape # 128,768
            freq_percentage = random.uniform(0.0, freq_masking_max_percentage)
            
            num_freqs_to_mask = int(freq_percentage * all_freqs_num)
            f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
            f0 = int(f0)
            spec[:, f0:f0 + num_freqs_to_mask] = 0

            time_percentage = random.uniform(0.0, time_masking_max_percentage)
            
            num_frames_to_mask = int(time_percentage * all_frames_num)
            t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
            t0 = int(t0)
            spec[t0:t0 + num_frames_to_mask, :] = 0
        return spec

    def __len__(self):
        return len(self.audio_lists)


class VggsoundCurationTestDataset(Dataset):
    def __init__(self):
        self.audio_lists = glob("/vggsound_test_curation/*.npy") # Put postprocessed audio test files here
        self.time_length = 864
        self.n_mels = 128
        self.num_frames = 5
        self.text_aug = EDA()
        self.width_resolution = 768
        self.frame_per_audio = self.time_length // self.num_frames

    def __getitem__(self, idx):
        wav_name = self.audio_lists[idx]
            
        audio_inputs = np.load(wav_name, allow_pickle=True)

        text_prompt = wav_name.split("/")[-1].split("_")[0]
        c, h, w = audio_inputs.shape


        if w >= self.time_length:
            j = random.randint(0, w-self.time_length)
            audio_inputs = audio_inputs[:,:,j:j+self.time_length]
        elif w < self.time_length:
            zero = np.zeros((1, self.n_mels, self.time_length))
            j = random.randint(0, self.time_length - w - 1)
            zero[:,:,j:j+w] = audio_inputs[:,:,:w]
            audio_inputs = zero

        audio_resize = audio_inputs[0,:self.n_mels,:self.width_resolution]

        w = audio_resize.shape[1]
        audio_inputs = audio_resize.reshape(-1,self.n_mels,self.width_resolution)
        
        audio_per_frame = []
        audio_per_frame_aug = []
        inter_frame = w // self.num_frames

        for idx_audio in range(self.num_frames):
            audio_seg = audio_inputs[:,:,idx_audio*inter_frame:idx_audio*inter_frame+inter_frame]
            _,_,seg_w = audio_seg.shape
            if seg_w >= self.frame_per_audio:
                j = random.randint(0, seg_w-self.frame_per_audio)
                audio_seg = audio_seg[:,:,j:j+self.frame_per_audio]
            elif seg_w < self.frame_per_audio:
                zero = np.zeros((1, self.n_mels, self.frame_per_audio))
                j = random.randint(0, self.frame_per_audio - seg_w - 1)
                zero[:,:,j:j+seg_w] = audio_seg[:,:,:seg_w]
                audio_seg = zero

            audio_seg = audio_seg[0,:self.n_mels,:inter_frame]

            audio_aug = self.spec_augment(audio_seg)

            audio_per_frame.append(audio_seg)


            audio_per_frame_aug.append(audio_aug)

        audio_per_frame=np.array(audio_per_frame)
        audio_per_frame_aug=np.array(audio_per_frame_aug)

        audio_per_frame = torch.from_numpy(audio_per_frame).float()
        audio_per_frame_aug = torch.from_numpy(audio_per_frame_aug).float()

            
        return audio_per_frame, audio_per_frame_aug, text_prompt

    def spec_augment(self, spec, num_mask=2, freq_masking_max_percentage=0.15, time_masking_max_percentage=0.3):
        spec = spec.copy()
        for i in range(num_mask):
            all_frames_num, all_freqs_num = spec.shape # 128,768
            freq_percentage = random.uniform(0.0, freq_masking_max_percentage)
            
            num_freqs_to_mask = int(freq_percentage * all_freqs_num)
            f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
            f0 = int(f0)
            spec[:, f0:f0 + num_freqs_to_mask] = 0

            time_percentage = random.uniform(0.0, time_masking_max_percentage)
            
            num_frames_to_mask = int(time_percentage * all_frames_num)
            t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
            t0 = int(t0)
            spec[t0:t0 + num_frames_to_mask, :] = 0
        return spec

    def __len__(self):
        return len(self.audio_lists)

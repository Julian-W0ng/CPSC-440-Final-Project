import torchaudio
import torch
from torch.utils.data import Dataset, DataLoader
import os
from utils import plot_waveform, plot_specgram

class MusicData(Dataset):
    def __init__(self, sample_length=5, sample_rate=44100, mode='train', transform=None):
        self.SAMPLE_RATE = sample_rate 
        self.SAMPLE_LENGTH = sample_length
        ROOT_DIR = './data/musicnet'
        self.root_dir = os.path.join(ROOT_DIR, f'{mode}_data')
        self.transform = transform
        self.data = []
        
        for file in os.listdir(self.root_dir):
            if file.endswith('.wav'):
                path = os.path.join(self.root_dir, file)
                self.data.append(path)
    

    def __len__(self):
        return int(len(self.data)/10)
    
    def __getitem__(self, idx):
        audio_path = self.data[idx]
        waveform, sample_rate = torchaudio.load(audio_path, normalize=False)
        if sample_rate != self.SAMPLE_RATE:
            # print(f'Resampling from {sample_rate} to {self.SAMPLE_RATE}')
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.SAMPLE_RATE)
            waveform = resampler(waveform)
        if self.transform:
            waveform = self.transform(waveform)    
        if waveform.shape[-1] > self.SAMPLE_RATE*self.SAMPLE_LENGTH:
            random_offset = torch.randint(0, waveform.shape[-1] - self.SAMPLE_RATE*self.SAMPLE_LENGTH, (1,)).item()
            waveform = waveform[:, random_offset:random_offset+self.SAMPLE_RATE*self.SAMPLE_LENGTH]

        # switch channels and time
        waveform = waveform.permute(1, 0)

        return waveform
    
if __name__ == '__main__':
    sample_length = 5
    sample_rate = 8000
    dataset = MusicData()
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
    for i, waveform in enumerate(dataloader):
        if i == 1:
            print('Waveform Shape:', waveform.shape)
            waveform = waveform.permute(0, 2, 1)
            torchaudio.save(f'./samples/sample_{i}.wav', waveform[0], sample_rate)
            
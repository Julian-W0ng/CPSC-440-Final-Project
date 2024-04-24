import torchaudio
from torch.utils.data import Dataset, DataLoader
import os

class MusicData(Dataset):
    def __init__(self, mode='train', transform=None):
        self.SAMPLE_RATE = 44100
        ROOT_DIR = './data/musicnet'
        self.root_dir = os.path.join(ROOT_DIR, f'{mode}_data')
        self.transform = transform
        self.data = []
        
        for file in os.listdir(self.root_dir):
            if file.endswith('.wav'):
                self.data.append(os.path.join(self.root_dir, file))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        audio_path = self.data[idx]
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != self.SAMPLE_RATE:
            print(f'Resampling from {sample_rate} to {self.SAMPLE_RATE}')
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.SAMPLE_RATE)
            waveform = resampler(waveform)
        if self.transform:
            waveform = self.transform(waveform)
        return waveform
    
if __name__ == '__main__':
    dataset = MusicData()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for i, (waveform, sample_rate) in enumerate(dataloader):
        print(waveform.shape, sample_rate)
        if i == 5:
            break
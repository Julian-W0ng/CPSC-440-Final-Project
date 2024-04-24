import torchaudio
from torch.utils.data import Dataset, DataLoader
import os

class MusicData(Dataset):
    def __init__(self, sample_length=5, sample_rate=44100, mode='train', transform=None):
        self.SAMPLE_RATE = sample_rate 
        self.SAMPLE_LENGTH = sample_length
        ROOT_DIR = './data/musicnet'
        self.root_dir = os.path.join(ROOT_DIR, f'{mode}_data')
        self.transform = transform
        self.data = []
        
        total_samples = 0
        usable_samples = 0
        for file in os.listdir(self.root_dir):
            if file.endswith('.wav'):
                path = os.path.join(self.root_dir, file)
                waveform, sample_rate = torchaudio.load(path)
                # Check length of audio
                if waveform.shape[-1] >= sample_rate*sample_length:
                    self.data.append(path)
                    usable_samples += 1
                total_samples += 1

        print(f'Total Samples: {total_samples}, Usable Samples: {usable_samples}')

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
        if waveform.shape[-1] > self.SAMPLE_RATE*self.SAMPLE_LENGTH:
            waveform = waveform[:, :self.SAMPLE_RATE*self.SAMPLE_LENGTH]

        # switch channels and time
        waveform = waveform.permute(1, 0)

        return waveform
    
if __name__ == '__main__':
    dataset = MusicData()
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
    for i, waveform in enumerate(dataloader):
        if i == 0:
            print('Waveform Shape:', waveform.shape)
        if waveform.shape[1] != dataset.SAMPLE_RATE*dataset.SAMPLE_LENGTH:
            print('Incorrect Shape:', waveform.shape)
            break
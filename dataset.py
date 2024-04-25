import torchaudio
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
        return len(self.data)
    
    def __getitem__(self, idx):
        audio_path = self.data[idx]
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != self.SAMPLE_RATE:
            # print(f'Resampling from {sample_rate} to {self.SAMPLE_RATE}')
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
        if i < 5:
            print('Waveform Shape:', waveform.shape)
            first_sample_waveform = waveform[0, :dataset.SAMPLE_RATE, 0].unsqueeze(0)
            plot_waveform(first_sample_waveform, dataset.SAMPLE_RATE)
            plot_specgram(first_sample_waveform, dataset.SAMPLE_RATE)
            input('Press Enter to continue...')
        if waveform.shape[1] != dataset.SAMPLE_RATE*dataset.SAMPLE_LENGTH:
            print('Incorrect Shape:', waveform.shape)
            break
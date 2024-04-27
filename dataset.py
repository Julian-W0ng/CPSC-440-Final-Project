import torchaudio
import torch
from torch.utils.data import Dataset, DataLoader
import os
from utils import plot_waveform, plot_specgram
import matplotlib.pyplot as plt
import librosa

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
        waveform, sample_rate = torchaudio.load(audio_path, normalize=True)
        if sample_rate != self.SAMPLE_RATE:
            # print(f'Resampling from {sample_rate} to {self.SAMPLE_RATE}')
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.SAMPLE_RATE)
            waveform = resampler(waveform)
        if self.transform:
            waveform = self.transform(waveform)    
        if waveform.shape[-1] > self.SAMPLE_RATE*self.SAMPLE_LENGTH:
            random_offset = torch.randint(0, waveform.shape[-1] - self.SAMPLE_RATE*self.SAMPLE_LENGTH, (1,)).item()
            waveform = waveform[:, random_offset:random_offset+self.SAMPLE_RATE*self.SAMPLE_LENGTH]
    
        return waveform

class MusicSpectrogram(MusicData):
    def __init__(self, sample_length=5, sample_rate=44100, mode='train', transform=None):
        super(MusicSpectrogram, self).__init__(sample_length, sample_rate, mode, transform)
        # self.sample_rate = sample_rate
        self.n_fft = 2048
        self.hop_length = 512
        # self.n_mels = 128
        # self.f_min = 30
        # self.f_max = 8000
        # self.power = 1.0
        # self.specgram_transform = torchaudio.transforms.MelSpectrogram(
        #     sample_rate=sample_rate,
        #     n_fft=self.n_fft,
        #     hop_length=self.hop_length,
        #     n_mels=self.n_mels,
        #     f_min=self.f_min,
        #     f_max=self.f_max,
        #     power=self.power
        # )
        # self.inverse_specgram_transform = torchaudio.transforms.InverseMelScale(
        #     n_stft=self.n_fft // 2 + 1,
        #     n_mels=self.n_mels,
        #     sample_rate=sample_rate,
        #     # f_min=self.f_min,
        #     # f_max=self.f_max
        # )
        self.specgram_transform = torchaudio.transforms.Spectrogram()
        self.inverse_specgram_transform = torchaudio.transforms.InverseSpectrogram()
    
    def __getitem__(self, idx):
        waveform = super(MusicSpectrogram, self).__getitem__(idx)
        window = torch.hann_window(self.n_fft, device=waveform.device)
        stft = torch.stft(waveform, n_fft=self.n_fft, hop_length=self.hop_length, window=window, return_complex=True)
        return stft
    def spectrogram_to_waveform(self, mel_spectrogram):
        # Convert the mel spectrogram to waveform
        n_fft = (mel_spectrogram.size(-2) - 1) * 2
        # Create a window function
        window = torch.hann_window(n_fft, device=mel_spectrogram.device)
        # Convert the spectrogram to waveform
        waveform = torch.istft(mel_spectrogram, n_fft=n_fft, hop_length=self.hop_length, window=window, return_complex=False)
        return waveform
        
    
if __name__ == '__main__':
    sample_length = 5
    sample_rate = 44100 
    dataset = MusicSpectrogram(sample_length=sample_length, sample_rate=sample_rate, mode='train')
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
    for i, specgram in enumerate(dataloader):
        if i == 1:
            print('Spectrogram Shape:', specgram.shape)
            # Plotting the spectrogram

            # Plotting the spectrogram
            librosa.display.specshow(specgram[0, 0].numpy(), sr=sample_rate, hop_length=512, x_axis='time', y_axis='mel')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Original Mel spectrogram')
            plt.show()

            # Save the spectrogram as wav file
            waveform = dataset.spectrogram_to_waveform(specgram[0])
            print('Waveform Shape:', waveform.shape)
            torchaudio.save(f'./samples/sample_{i}.wav', waveform, sample_rate)

            break

    # dataset = MusicData()
    # dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
    # for i, waveform in enumerate(dataloader):
    #     if i == 1:
    #         print('Waveform Shape:', waveform.shape)
    #         waveform = waveform.permute(0, 2, 1)
    #         torchaudio.save(f'./samples/sample_{i}.wav', waveform[0], sample_rate)
            
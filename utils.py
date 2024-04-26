import torch
import torchaudio
import os
import matplotlib.pyplot as plt


# Parse Arguments
# Reference: https://stackoverflow.com/questions/44561722/why-in-argparse-a-true-is-always-true
def bool_string(s):
    if s.lower() == 'true':
        return True
    elif s.lower() == 'false':
        return False
    else:
        raise ValueError

#Reference: https://www.youtube.com/watch?v=3mju52xBFK8
def _plot(waveform, sample_rate, title):
  waveform = waveform.numpy()

  num_channels, num_frames = waveform.shape
  time_axis = torch.arange(0, num_frames) / sample_rate

  figure, axes = plt.subplots(num_channels, 1)
  if num_channels == 1:
    axes = [axes]
  for c in range(num_channels):
    if title == "Waveform":
      axes[c].plot(time_axis, waveform[c], linewidth=1)
      axes[c].grid(True)
    else:
      axes[c].specgram(waveform[c], Fs=sample_rate)
    if num_channels > 1:
      axes[c].set_ylabel(f'Channel {c+1}')
  figure.suptitle(title)
  plt.show(block=False)

def plot_waveform(waveform, sample_rate):
  _plot(waveform, sample_rate, title="Waveform")

def plot_specgram(waveform, sample_rate):
  _plot(waveform, sample_rate, title="Spectrogram")


# Reference: cpen455 utiils.py file
class mean_tracker:
    def __init__(self):
        self.sum = 0
        self.count = 0
    def update(self, new_value):
        self.sum += new_value
        self.count += 1
    def get_mean(self):
        return self.sum/self.count
    def reset(self):
        self.sum = 0
        self.count = 0
import torch
import argparse
import wandb
import time
import os
from tqdm import tqdm
from pprint import pprint
from dataset import MusicData, MusicSpectrogram
from model import VariationalTransformerAutoencoder, elbo_loss, sample
from vaesimple import SimpleVAE
from vaespec_model import VAE_Spectogram
import torchaudio
from utils import bool_string, plot_waveform, plot_specgram, mean_tracker
import torch.nn as nn
import torch.nn.functional as F

def train_vae(model, data_loader, optimizer, loss_op, device, args, epoch, n_samples=10):
    model.train()
    loss_tracker = mean_tracker()

    for batch_idx, data in enumerate(tqdm(data_loader)):
        data = data.to(device)
        loss = -model.elbo(data)
        loss_tracker.update(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if args.wandb:
        wandb.log({'train_loss': loss_tracker.get_mean()}, step=epoch)
        wandb.log({'train_epoch': epoch})


parser = argparse.ArgumentParser(description='PyTorch Variational Transformer For Music Generation')

parser.add_argument('--data', type=str, default='data',
                    help='location of the midi data')
parser.add_argument('--checkpoint', type=str, default='checkpoints',
                    help='location of the model checkpoints')
parser.add_argument('--sample', type=str, default='samples',
                    help='location of the samples')
parser.add_argument('--sample_size', type=int, default=5,
                    help='number of samples to generate')
parser.add_argument('--model', type=str, default=None,
                    help='location of the model to load and continue training')
parser.add_argument('--epochs', type=int, default=1000,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=5,
                    help='batch size')
parser.add_argument('--seq_len', type=int, default=1,
                    help='length of the sequence in seconds')
parser.add_argument('--sample_rate', type=int, default=3000,
                    help='sample rate')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='dropout rate')
parser.add_argument('--nheads', type=int, default=1,
                    help='number of attention heads')
parser.add_argument('--channels', type=int, default=1,
                    help='number of channels')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed')
parser.add_argument('--wandb', type=bool_string, default=True,
                    help='use wandb for logging')
parser.add_argument('--lr', type=int, default=1e-3, help='learning rate')
parser.add_argument('--save_interval', type=int, default=1, help='save interval')

args = parser.parse_args()
pprint(args.__dict__)

# Set Seed
torch.manual_seed(args.seed)

# wandb
if args.wandb:
    project_name = 'Music-Generation-Variational-Transformer'
    wandb.init(project=project_name)
    wandb.config.current_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    wandb.config.update(args)

# Create Directories if they don't exist
os.makedirs(args.data, exist_ok=True)
os.makedirs(args.checkpoint, exist_ok=True)
os.makedirs(args.sample, exist_ok=True)

# Set Device
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print('USING DEVICE:', device)

# train_dataset = MusicData(sample_length=args.seq_len, sample_rate=args.sample_rate, mode='train')
# test_dataset = MusicData(sample_length=args.seq_len, sample_rate=args.sample_rate, mode='test')

train_dataset = MusicSpectrogram(sample_length=args.seq_len, sample_rate=args.sample_rate, mode='train')
test_dataset = MusicSpectrogram(sample_length=args.seq_len, sample_rate=args.sample_rate, mode='test')


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)




model = VAE_Spectogram(latent_dim=2, height=128, width=128, channels=1)
if args.model:
    model = torch.load(args.model)

model.to(device)

loss_op = -model.elbo
sample_op = sample
optimizer = torch.optim.AdamW(model.parameters(), weight_decay=1e-5, lr=args.lr)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.995)

for epoch in tqdm(range(args.epochs)):
    train_vae(
        model=model,
        data_loader=train_loader,
        optimizer=optimizer,
        loss_op=loss_op,
        device=device,
        args=args,
        epoch=epoch
    )
    lr_scheduler.step()
    if epoch % args.save_interval == 0:
        print('-------Saving, Uploading, and Generating Samples and Checkpoint-------')
        torch.save(model, f'{args.checkpoint}/model_{epoch}.pt')

        # Generate Sample Audio
        n_samples = args.sample_size
        samples_z = torch.randn(n_samples, 1, K).to(device)
        with torch.no_grad():
            samples_x = model.decode(samples_z)

        # Save Waveform
        for i in range(samples_x.shape[0]):
            waveform = samples_x[i].detach().cpu()
            torchaudio.save(f'{args.sample}/sample_{epoch}_{i}.wav', waveform[0], args.sample_rate)

        print('-------Done-------')


import torch
import argparse
import wandb
import time
import os
from tqdm import tqdm
from pprint import pprint
from dataset import MusicData
from model import VariationalTransformerAutoencoder, elbo_loss, sample
from vaesimple import SimpleVAE
from vae import VAE
import torchaudio
from utils import bool_string, plot_waveform, plot_specgram, mean_tracker
import torch.nn as nn
import torch.nn.functional as F

def train_vae(model, data_loader, optimizer, loss_op, device, args, epoch, n_samples=10):
    model.train()
    loss_tracker = mean_tracker()

    for batch_idx, data in enumerate(tqdm(data_loader)):
        data = data.to(device)
        x_hat, mu, log_sigma = model(data)
        loss = model.loss(data, x_hat, mu, log_sigma)
        loss_tracker.update(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if args.wandb:
        wandb.log({'train_loss': loss_tracker.get_mean()}, step=epoch)
        wandb.log({'train_epoch': epoch})


def test(model: VariationalTransformerAutoencoder, data_loader, loss_op, device, args, epoch):
    model.eval()

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(data_loader)):
            data = data.to(device)
            output, mu, log_sigma = model(data)
            loss = loss_op(output, data, mu, log_sigma)

    if args.wandb:
        wandb.log({'test_loss': loss.item()}, step=epoch)
        wandb.log({'test_epoch': epoch})



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
parser.add_argument('--epochs', type=int, default=2000,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size')
parser.add_argument('--seq_len', type=int, default=5,
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
parser.add_argument('--lr', type=int, default=1e-5, help='learning rate')
parser.add_argument('--save_interval', type=int, default=25, help='save interval')

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

train_dataset = MusicData(sample_length=args.seq_len, sample_rate=args.sample_rate, mode='train')
test_dataset = MusicData(sample_length=args.seq_len, sample_rate=args.sample_rate, mode='test')

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)


# load model if specified
# model = VariationalTransformerAutoencoder(device, nheads=args.nheads, sequence_length=args.seq_len*args.sample_rate, channels=args.channels, dropout=args.dropout)
# model = SimpleVAE(K=K, num_filters= 32, sequence_length=args.seq_len*args.sample_rate, channels=args.channels, sample_rate = args.sample_rate)
hidden_size = args.sample_rate * 3 
latent_size = args.sample_rate // 2
model = VAE(input_size=args.sample_rate*args.seq_len, hidden_size=hidden_size, latent_size=latent_size, input_channels=args.channels, num_hidden_layers=15)
if args.model:
    model = torch.load(args.model)

model.to(device)

loss_op = F.mse_loss
optimizer = torch.optim.AdamW(model.parameters(), weight_decay=1e-5, lr=args.lr)
for param_group in optimizer.param_groups:
        param_group['initial_lr'] = args.lr
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.995)
# lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(T_0=3, T_mult=2, eta_min=1e-5, last_epoch=args.epochs, optimizer=optimizer)

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
    # test(
    #     model=model,
    #     data_loader=test_loader,
    #     loss_op=loss_op,
    #     device=device,
    #     args=args,
    #     epoch=epoch
    # )
    if epoch % args.save_interval == 0:
        print('-------Saving, Uploading, and Generating Samples and Checkpoint-------')
        torch.save(model, f'{args.checkpoint}/model_{epoch}.pt')

        # Generate Sample Audio
        n_samples = args.sample_size
        with torch.no_grad():
            samples_x = model.sample(n_samples, device=device)

        # Save Waveform
        for i in range(samples_x.shape[0]):
            waveform = samples_x[i].detach().cpu()
            torchaudio.save(f'{args.sample}/sample_{epoch}_{i}.wav', waveform, args.sample_rate)


        # if args.wandb:
            # sample_np = sample.detach().cpu().numpy()
            # wandb.log({f'samples': wandb.Audio(sample.squeeze(), sample_rate=args.sample_rate)}, step=epoch+1)

        # Plot Waveform
        # plot_waveform(sample, args.sample_rate)
        # plot_specgram(sample, args.sample_rate)
        # input('Press Enter to Continue')

        print('-------Done-------')


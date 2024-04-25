import torch
import argparse
import wandb
import time
import os
from tqdm import tqdm
from pprint import pprint
from dataset import MusicData
from model import VariationalTransformerEncoder, VariationalTransformerDecoder, sample_latent, elbo_loss, sample_wav
import torchaudio
from utils import bool_string, plot_waveform, plot_specgram

def train_vae(encoder, decoder, data_loader, optimizer_en, optimizer_dec, loss_op, device, args, epoch):
    encoder.train()
    decoder.train()
    # loss_tracker = mean_tracker()

    for batch_idx, data in enumerate(tqdm(data_loader)):
        data = data.to(device)
        mu, sigma = encoder(data)
        z = sample_latent(mu, sigma)
        output = decoder(data, z)
        loss = loss_op(data, output, mu, sigma)
        # loss_tracker.update(loss.item())
        optimizer_en.zero_grad()
        optimizer_dec.zero_grad()
        loss.backward()
        optimizer_en.step()
        optimizer_dec.step()

    if args.wandb:
        wandb.log({'train_loss': loss.item()}, step=epoch)
        wandb.log({'train_epoch': epoch})


def test(model, data_loader, loss_op, device, args, epoch):
    model.eval()
    # loss_tracker = mean_tracker()

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(data_loader)):
            data = data.to(device)
            output = model(data)
            loss = loss_op(output, data)
            # loss_tracker.update(loss.item())

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
parser.add_argument('--enc_model', type=str, default=None,
                    help='location of the model to continue training')
parser.add_argument('--dec_model', type=str, default=None,
                    help='location of the model to continue training')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=5,
                    help='batch size')
parser.add_argument('--seq_len', type=int, default=10,
                    help='length of the sequence in seconds')
parser.add_argument('--sample_rate', type=int, default=100,
                    help='sample rate')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='dropout rate')
parser.add_argument('--nheads', type=int, default=1,
                    help='number of attention heads')
parser.add_argument('--channels', type=int, default=1,
                    help='number of channels')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed')
parser.add_argument('--wandb', type=bool_string, default=False,
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

train_dataset = MusicData(sample_length=args.seq_len, sample_rate=args.sample_rate, mode='train')
test_dataset = MusicData(sample_length=args.seq_len, sample_rate=args.sample_rate, mode='test')

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)


# load model if specified
encoder = VariationalTransformerEncoder(device=device, nheads=args.nheads, sequence_length=args.seq_len*args.sample_rate, channels=args.channels, dropout=args.dropout)
decoder = VariationalTransformerDecoder(device=device, nheads=args.nheads, sequence_length=args.seq_len*args.sample_rate, channels=args.channels, dropout=args.dropout)

if args.enc_model:
    enc_model = torch.load(torch.load(args.enc_model))
    encoder.load_state_dict(enc_model)
if args.dec_model:
    dec_model = torch.load(torch.load(args.dec_model))
    decoder.load_state_dict(dec_model)

encoder.to(device)
decoder.to(device)

loss_op = elbo_loss
sample_op = sample_wav
optimizer_en = torch.optim.Adam(encoder.parameters(), lr=args.lr)
optimizer_dec = torch.optim.Adam(decoder.parameters(), lr=args.lr)

lr_scheduler_en = torch.optim.lr_scheduler.StepLR(optimizer_en, step_size=1, gamma=0.1)
lr_scheduler_dec = torch.optim.lr_scheduler.StepLR(optimizer_dec, step_size=1, gamma=0.1)

for epoch in tqdm(range(args.epochs)):
    train_vae(
        encoder=encoder,
        decoder=decoder,
        data_loader=train_loader,
        optimizer_en=optimizer_en,
        optimizer_dec= optimizer_dec,
        loss_op=loss_op,
        device=device,
        args=args,
        epoch=epoch
    )
    lr_scheduler_en.step()
    lr_scheduler_dec.step()
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
        torch.save(encoder, f'{args.checkpoint}/encoder_{epoch}.pth')
        torch.save(decoder, f'{args.checkpoint}/decoder_{epoch}.pth')
        # Upload to wandb

        # Generate Sample Audio
        start_token = torch.zeros
        sample = sample_op(
            decoder=decoder,
            device=device,
            sample_length=args.seq_len*args.sample_rate,
            sample_size=args.sample_size,
            channels=args.channels)
        
        print(sample[0])    
        #Upload to wandb
        
        # Save to File
        for i in range(args.sample_size):
            print(sample[i].shape)
            wav_to_save = sample[i].detach().cpu()
            torchaudio.save(f'{args.sample}/sample_{epoch}_{i}.wav', wav_to_save, sample_rate=args.sample_rate, channels_first=False)
        
        #convert to numpy and upload
        if args.wandb:
            # sample_np = sample.detach().cpu().numpy()
            wandb.log({f'samples': wandb.Audio(sample.squeeze(), sample_rate=args.sample_rate)}, step=epoch+1)

        # # Plot Waveform
        # plot_waveform(sample, args.sample_rate)
        # plot_specgram(sample, args.sample_rate)
        # input('Press Enter to Continue')

        print('-------Done-------')


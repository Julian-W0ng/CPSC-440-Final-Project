import torch
import argparse
import wandb
import time
import os
from pprint import pprint

# Parse Arguments
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
                    help='location of the model to continue training')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch size')
parser.add_argument('--seq_len', type=int, default=256,
                    help='sequence length')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed')
parser.add_argument('--wandb', type=bool, default=True,
                    help='use wandb for logging')

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

# load model if specified
model = None #TODO: Load model
if args.model:
    model = torch.load(torch.load(args.model))

model = model.to(device)

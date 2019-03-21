import argparse
import os
import torch
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import json

from model import Generator

parser = argparse.ArgumentParser()
parser.add_argument('-load_path', default='checkpoints-pokedex/model_500.pth', help='Checkpoint to load path from')
parser.add_argument('-num_output', default=9, help='Number of generated outputs')
parser.add_argument('-tp1', default='fire', help='type 1')
parser.add_argument('-tp2', default='<UNK>', help='type 2')
parser.add_argument('-load_json', default='data/pokegan-params.json', help='Load path for params json.')
args = parser.parse_args()

# Load the checkpoint file.
state_dict = torch.load(args.load_path, map_location=lambda storage, loc: storage)

# Load color2ind dictionary from json parameter file.
with open(args.load_json, 'r') as info_file:
            info = json.load(info_file)
            tp2ind = info['tp2ind']

# Set the device to run on: GPU or CPU.
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
# Get the 'params' dictionary from the loaded state_dict.
params = state_dict['params']

# Create the generator network.
netG = Generator(params).to(device)
# Load the trained generator weights.
netG.load_state_dict(state_dict['generator'])
print(netG)

print(args.num_output)
# Get latent vector Z from unit normal distribution.
noise = torch.randn(int(args.num_output), params['nz'], 1, 1, device=device)

# To create onehot embeddings for the condition labels.
onehot = torch.zeros(params['vocab_size'], params['vocab_size'])
onehot = onehot.scatter_(1, torch.LongTensor([i for i in range(params['vocab_size'])]).view(params['vocab_size'], 1), 1).view(params['vocab_size'], params['vocab_size'], 1, 1)

fixed_noise = torch.randn(36, params['nz'], 1, 1, device=device)
#Contructing fixed conditions
frow1 = torch.cat((torch.ones(6, 1)*tp2ind['fire'], torch.ones(6, 1)*tp2ind['<UNK>']), dim=1)
frow2 = torch.cat((torch.ones(6, 1)*tp2ind['flying'], torch.ones(6, 1)*tp2ind['<UNK>']), dim=1)
frow3 = torch.cat((torch.ones(6, 1)*tp2ind['grass'], torch.ones(6, 1)*tp2ind['<UNK>']), dim=1)
frow4 = torch.cat((torch.ones(6, 1)*tp2ind['water'], torch.ones(6, 1)*tp2ind['<UNK>']), dim=1)
frow5 = torch.cat((torch.ones(6, 1)*tp2ind['dragon'], torch.ones(6, 1)*tp2ind['<UNK>']), dim=1)
frow6 = torch.cat((torch.ones(6, 1)*tp2ind['bug'], torch.ones(6, 1)*tp2ind['<UNK>']), dim=1)
fixed_condition = torch.cat((frow1, frow2, frow3, frow4, frow5,
                                frow6), dim=0).type(torch.LongTensor)


fixed_condition_ohe1 = onehot[fixed_condition[:, 0]].to(device)
fixed_condition_ohe2 = onehot[fixed_condition[:, 1]].to(device)

# Create input conditions vectors.
input_condition = torch.cat((torch.ones(int(args.num_output), 1)*tp2ind[args.tp1], 
                            torch.ones(int(args.num_output), 1)*tp2ind[args.tp2]),
                            dim=1).type(torch.LongTensor)

# Generate the onehot embeddings for the conditions.
tp1_ohe = onehot[input_condition[:, 0]].to(device)
tp2_ohe = onehot[input_condition[:, 1]].to(device)

# Turn off gradient calculation to speed up the process.
with torch.no_grad():
    # Get generated image from the noise vector using
    # the trained generator.
    generated_img = netG(fixed_noise, fixed_condition_ohe1, fixed_condition_ohe2).detach().cpu()

# Display the generated image.


result_dir = 'test_result/'
if not os.path.exists(result_dir):
    os.mkdir(result_dir)
    print("Directory " , result_dir ,  " Created ")

img_data = np.transpose(vutils.make_grid(generated_img, nrow=6, padding=2, normalize=True).cpu(), (1, 2, 0))
plt.imsave(result_dir + args.load_path.split('/')[1].split('.')[0] +'.png', img_data)

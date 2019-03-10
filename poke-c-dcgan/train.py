# import matplotlib
# matplotlib.use("TkAgg")
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import json
import os
import argparse

from torchvision import transforms
from torch.utils.data import DataLoader
from dataloader import PokemonDataset
from model import weights_init, Generator, Discriminator


# Set random seed for reproducibility.
seed = 8008
random.seed(seed)
torch.manual_seed(seed)
print("Random Seed: ", seed)

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64, help='Batch size during training')
parser.add_argument('--img_size', type=int, default=64, help='Spatial size of training images. All images will be resized to this size during preprocessing')
parser.add_argument('--nc', type=int, default=3, help='Number of channles in the training images. For coloured images this is 3')
parser.add_argument('--nz', type=int, default=100, help='Size of the Z latent vector (the input to the generator)')
parser.add_argument('--ngf', type=int, default=64, help='Size of feature maps in the generator. The depth will be multiples of this')
parser.add_argument('--ndf', type=int, default=64, help='Size of features maps in the discriminator. The depth will be multiples of this')
parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate for optimizers')
parser.add_argument('--beta1', type=float, default=0.5, help='Beta1 hyperparam for Adam optimizer')
parser.add_argument('--save_epoch', type=int, default=10, help='Save step')
parser.add_argument('--n_critic', type=int, default=5, help='Number of iterations to train discriminator before training generator')
parser.add_argument('--csv_file', type=str, default='data/pokemon-dex.csv', help='Path of csv file')
parser.add_argument('--root_dir', type=str, default='data/pokedex-images-rgb', help='Path of training images')
parser.add_argument('--param_file', type=str, default='data/pokegan-params.json', help='Path of json file')

args = parser.parse_args()

# Parameters to define the model.
params = {
    "bsize" : args.batch_size, # Batch size during training.
    'imsize' : args.img_size, # Spatial size of training images. All images will be resized to this size during preprocessing.
    'nc' : args.nc, # Number of channles in the training images. For coloured images this is 3.
    'nz' : args.nz, # Size of the Z latent vector (the input to the generator).
    'ngf' : args.ngf, # Size of feature maps in the generator. The depth will be multiples of this.
    'ndf' : args.ndf, # Size of features maps in the discriminator. The depth will be multiples of this.
    'num_epochs' : args.num_epochs, # Number of training epochs.
    'lr' : args.lr, # Learning rate for optimizers
    'beta1' : args.beta1, # Beta1 hyperparam for Adam optimizer
    'save_epoch' : args.save_epoch, # Save step.
    'n_critic' : args.n_critic} #Number of iterations to train discriminator before training generator.

# Directory path to store the result
result_dir = 'result/' + str(params['bsize']) + '-' + str(params['num_epochs'])
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

if not os.path.exists('checkpoints'):
    os.mkdir('checkpoints')

# Use GPU is available else use CPU.
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
print(device, " will be used.\n")

# Get the data.
transform = transforms.Compose([
    transforms.Resize(params['imsize']),
    transforms.CenterCrop(params['imsize']),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5))])

pokemon_dataset = PokemonDataset(csv_file=args.csv_file,
                             root_dir=args.root_dir,
                             param_file=args.param_file,
                             transform=transform)

dataloader = DataLoader(pokemon_dataset, batch_size=params['bsize'],
                        shuffle=True)

# # Plot the training images.
# sample_batch = next(iter(dataloader))
# plt.figure(figsize=(6, 6))
# plt.axis("off")
# plt.title("Training Images")
# plt.imshow(np.transpose(vutils.make_grid(
#     sample_batch['image'].to(device)[ : 36], nrow=6, padding=2, normalize=True).cpu(), (1, 2, 0)))

# plt.show()

# params['n_conditions'] = len(sample_batch['colors'][0])

with open('data/pokegan-params.json', 'r') as info_file:
            info = json.load(info_file)
            ind2tp = info['ind2tp']
            tp2ind = info['tp2ind']

params['vocab_size'] = len(ind2tp) + 1
params['embedding_size'] = params['vocab_size']

# Create the generator.
netG = Generator(params).to(device)
# Apply the weights_init() function to randomly initialize all
# weights to mean=0.0, stddev=0.2
netG.apply(weights_init)
# Print the model.
print(netG)

# Create the discriminator.
netD = Discriminator(params).to(device)
# Apply the weights_init() function to randomly initialize all
# weights to mean=0.0, stddev=0.2
netD.apply(weights_init)
# Print the model.
print(netD)

# Binary Cross Entropy loss function.
criterion = nn.BCELoss()

onehot = torch.zeros(params['vocab_size'], params['vocab_size'])
onehot = onehot.scatter_(1, torch.LongTensor([i for i in range(params['vocab_size'])]).view(params['vocab_size'], 1), 1).view(params['vocab_size'], params['vocab_size'], 1, 1)


fixed_noise = torch.randn(36, params['nz'], 1, 1, device=device)
#Contructing fixed conditions
frow1 = torch.cat((torch.ones(6, 1)*tp2ind['bug'], torch.ones(6, 1)*tp2ind['poison']), dim=1)
frow2 = torch.cat((torch.ones(6, 1)*tp2ind['normal'], torch.ones(6, 1)*tp2ind['flying']), dim=1)
frow3 = torch.cat((torch.ones(6, 1)*tp2ind['grass'], torch.ones(6, 1)*tp2ind['poison']), dim=1)
frow4 = torch.cat((torch.ones(6, 1)*tp2ind['rock'], torch.ones(6, 1)*tp2ind['ground']), dim=1)
frow5 = torch.cat((torch.ones(6, 1)*tp2ind['electric'], torch.ones(6, 1)*tp2ind['steel']), dim=1)
frow6 = torch.cat((torch.ones(6, 1)*tp2ind['ghost'], torch.ones(6, 1)*tp2ind['poison']), dim=1)
fixed_condition = torch.cat((frow1, frow2, frow3, frow4, frow5,
                                frow6), dim=0).type(torch.LongTensor)


fixed_condition_ohe1 = onehot[fixed_condition[:, 0]].to(device)
fixed_condition_ohe2 = onehot[fixed_condition[:, 1]].to(device)

real_label = 1
fake_label = 0

# Optimizer for the discriminator.
optimizerD = optim.Adam(netD.parameters(), lr=params['lr'], betas=(params['beta1'], 0.999))
# Optimizer for the generator.
optimizerG = optim.Adam(netG.parameters(), lr=params['lr'], betas=(params['beta1'], 0.999))

# Stores generated images as training progresses.
img_list = []
# Stores generator losses during training.
G_losses = []
# Stores discriminator losses during training.
D_losses = []

iters = 0

print("Starting Training Loop...")
print("-"*25)

for epoch in range(params['num_epochs']):
    for i, data in enumerate(dataloader, 0):
        # Transfer data tensor to GPU/CPU (device)
        real_images = data['image'].to(device)
        y = data['tps']
        # Correct Conditions
        r_condition1 = onehot[y[:, 0]].to(device)
        r_condition2 = onehot[y[:, 1]].to(device)

        # Wrong Conditions
        idx = torch.randperm(r_condition1.nelement())
        w_condition1 = r_condition1.view(-1)[idx].view(r_condition1.size())
        idx = torch.randperm(r_condition2.nelement())
        w_condition2 = r_condition2.view(-1)[idx].view(r_condition2.size())
        # Get batch size. Can be different from params['nbsize'] for last batch in epoch.
        b_size = real_images.size(0)
        
        # Make accumalated gradients of the discriminator zero.
        netD.zero_grad()
        # Create labels for the real data. (label=1)
        #label = torch.full((b_size, ), real_label, device=device)
        label = torch.rand((b_size, ), device=device)*(1.2 - 0.8) + 0.8

        # Real Image, Correct Conditions
        output = netD(real_images, r_condition1, r_condition2).view(-1)
        errD_real = criterion(output, label)
        # Calculate gradients for backpropagation.
        errD_real.backward()
        D_x = output.mean().item()
        
        # Sample random data from a unit normal distribution.
        noise = torch.randn(b_size, params['nz'], 1, 1, device=device)
        # Generate fake data (images).
        fake_data = netG(noise, r_condition1, r_condition2)
        # Create labels for fake data. (label=0)
        label.fill_(fake_label)
        #label = torch.rand((b_size, ), device=device)*(0.2 - 0.0) + 0.0
        # Calculate the output of the discriminator of the fake data.
        # As no gradients w.r.t. the generator parameters are to be
        # calculated, detach() is used. Hence, only gradients w.r.t. the
        # discriminator parameters will be calculated.
        # This is done because the loss functions for the discriminator
        # and the generator are slightly different.
        
        # Fake Image, Correct Condition
        output = netD(fake_data.detach(), r_condition1, r_condition2).view(-1)
        errD_fake1 = criterion(output, label) / 2
        # Calculate gradients for backpropagation.
        errD_fake1.backward()
        D_G_z1 = output.mean().item()

        # Real Image, Wrong Conditions.
        output = netD(real_images, w_condition1, w_condition2).view(-1)
        label.fill_(fake_label)
        errD_fake2 = criterion(output, label) / 2
        #Calculate gradients for backpropagation
        errD_fake2.backward()

        # Net discriminator loss.
        errD_fake = errD_fake1 + errD_fake2
        errD = errD_real + errD_fake
        # Update discriminator parameters.
        optimizerD.step()

        # Weight clipping for discriminator weights.
        #for p in netD.parameters():
        #    p.data.clamp_(-0.015, 0.015)
        
        if(True):
            # Make accumalted gradients of the generator zero.
            netG.zero_grad()
            # We want the fake data to be classified as real. Hence
            # real_label are used. (label=1)
            #label.fill_(real_label)
            label = torch.rand((b_size, ), device=device)*(1.2 - 0.8) + 0.8
            # No detach() is used here as we want to calculate the gradients w.r.t.
            # the generator this time.
            output = netD(fake_data, r_condition1, r_condition2).view(-1)
            errG = criterion(output, label)
            #errG = -torch.mean(output)
            # Gradients for backpropagation are calculated.
            # Gradients w.r.t. both the generator and the discriminator
            # parameters are calculated, however, the generator's optimizer
            # will only update the parameters of the generator. The discriminator
            # gradients will be set to zero in the next iteration by netD.zero_grad()
            errG.backward()

            D_G_z2 = output.mean().item()
            # Update generator parameters.
            optimizerG.step()

        # Check progress of training.
        if i%100 == 0:
            print(torch.cuda.is_available())
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, params['num_epochs'], i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save the losses for plotting.
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        iters += 1
    # Check how the generator is doing by saving G's output on a fixed noise.
    with torch.no_grad():
        fake_data = netG(fixed_noise, fixed_condition_ohe1, fixed_condition_ohe2).detach().cpu()
    # img_list.append(vutils.make_grid(fake_data, nrow=6, padding=2, normalize=True))

    # Save the model.
    if epoch % params['save_epoch'] == 0:
        torch.save({
            'generator' : netG.state_dict(),
            'discriminator' : netD.state_dict(),
            'optimizerG' : optimizerG.state_dict(),
            'optimizerD' : optimizerD.state_dict(),
            'params' : params
            }, 'checkpoints/model_epoch_{}.pth'.format(epoch))
        # plt.figure(figsize=(6, 6))
        # plt.axis("off")
        # plt.title("Epoch_{}".format(epoch))
        # Visualize output on fixed noise and conditions.
        with torch.no_grad():
            fake_data = netG(fixed_noise, fixed_condition_ohe1, fixed_condition_ohe2).detach().cpu()
            # print(fake_data.size())
            # vutils.save_image(fake_data, (result_dir +'/Epoch_{}.png'.format(epoch)), nrow=6, padding=2)
        # img_save = plt.imshow(np.transpose(vutils.make_grid(fake_data, nrow=6, padding=2, normalize=True).cpu(), (1, 2, 0)))
        # img_save.figure.savefig(result_dir +'/Epoch_{}'.format(epoch))
        img_data = np.transpose(vutils.make_grid(fake_data, nrow=6, padding=2, normalize=True).cpu(), (1, 2, 0))
        plt.imsave(result_dir +'/Epoch_{}.png'.format(epoch), img_data)

# Save the final trained model.
torch.save({
            'generator' : netG.state_dict(),
            'discriminator' : netD.state_dict(),
            'optimizerG' : optimizerG.state_dict(),
            'optimizerD' : optimizerD.state_dict(),
            'params' : params
            }, 'checkpoints/model_final.pth')
# plt.figure(figsize=(6, 6))
# plt.axis("off")
# plt.title("Epoch_{}".format(params['num_epochs']))
# Visualize output on fixed noise and conditions.
with torch.no_grad():
    fake_data = netG(fixed_noise, fixed_condition_ohe1, fixed_condition_ohe2).detach().cpu()
# vutils.save_image(fake_data, (result_dir +'/Epoch_{}.png'.format(epoch)), nrow=6, padding=2)
# img_save = plt.imshow(np.transpose(vutils.make_grid(fake_data, nrow=6, padding=2, normalize=True).cpu(), (1, 2, 0)))
# img_save.figure.savefig(result_dir +'/Epoch_final.png')
img_data = np.transpose(vutils.make_grid(fake_data, nrow=6, padding=2, normalize=True).cpu(), (1, 2, 0))
plt.imsave(result_dir +'/Epoch_{}.png'.format(epoch), img_data)


# Plot the training losses.
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig(result_dir +"/Loss_Curve")

# Animation showing the improvements of the generator.
# fig = plt.figure(figsize=(6,6))
# plt.axis("off")
# ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
# anim = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
# #plt.show()
# anim.save('poke.gif', dpi=100, writer='imagemagick')
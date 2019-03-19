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
from torchvision.utils import save_image
import glob, os


# batch_tensor = torch.randn(*(36, 3, 64, 64))
# batch_tensor = torch.randn(*(36, 4, 64, 64))
# batch_tensor = torch.split(batch_tensor, 1)[0]
# print(batch_tensor.size())

# image = torchvision.transforms.ToPILImage(mode='RGBA')(batch_tensor)
# img = plt.imshow(image)
# img.figure.savefig('lalala')


# grid_img = vutils.make_grid(batch_tensor, nrow=6)
# print(grid_img.shape)
# img = plt.imshow(grid_img.permute(1, 2, 0))
# img.figure.savefig('hahah')

i = 0
for infile in glob.glob('../data/evol_pairs_mini_rgb' + "/*.png"):
    i += 1

print(i)
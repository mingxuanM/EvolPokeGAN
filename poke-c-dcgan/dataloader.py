import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from six import iteritems

class PokemonDataset(Dataset):

    def __init__(self, csv_file, root_dir, param_file, transform):
        self.tp_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.param_file = param_file

        with open(self.param_file, 'r') as info_file:
            info = json.load(info_file)

            for key, value in iteritems(info):
                setattr(self, key, value)

        tp_count = len(self.tp2ind)
        self.tp2ind['<S>'] = tp_count + 1
        self.tp2ind['</S>'] = tp_count + 2
        self.start_token = self.tp2ind['<S>']
        self.end_token = self.tp2ind['</S>']

        self.ind2tp = {
            int(ind): tp
            for tp, ind in iteritems(self.tp2ind)
        }

    def __len__(self):
        return len(self.tp_frame)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir,
                                '/' + str('%03d' % self.tp_frame.iloc[idx, 32])+".png")

        image = Image.open(img_path)
        tps = self.tp_frame.iloc[idx, 36:38].values
        tps = [self.tp2ind.get(tp, self.tp2ind['<UNK>']) for tp in tps]
        tps = torch.LongTensor(tps)

        transformed_images = self.transform(image)

        sample = {'image': transformed_images, 'tps': tps}

        return sample 

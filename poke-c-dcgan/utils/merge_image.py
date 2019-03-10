import sys
from PIL import Image
import pandas as pd
import math
import os


save_dir = '../data/evol_pairs_images'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

csv_file = pd.read_csv('../data/pokemon-dex-ev.csv')
ev_idxs = csv_file.iloc[:, 32].values
pre_idxs = csv_file.iloc[:, 33].values

idx_pairs = list(zip(ev_idxs, pre_idxs))
img_dir = '../data/pokedex-images-rgb/'
for p in idx_pairs:
    if not math.isnan(p[1]):
        im_pair = []
        im_pair.append(Image.open(img_dir + '%03d' % int(p[0]) + '.png'))
        im_pair.append(Image.open(img_dir + '%03d' % int(p[1]) + '.png'))
        widths, heights = zip(*(i.size for i in im_pair))
        total_width = sum(widths)
        max_height = max(heights)

        new_im = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for im in im_pair:
            new_im.paste(im, (x_offset,0))
            x_offset += im.size[0]
        new_im.save(save_dir + '/' + '%03d' % int(p[0]) + '-ev' + '.png')


# images = []
# images.append(Image.open('001.png'))
# images.append(Image.open('002.png'))
# widths, heights = zip(*(i.size for i in images))
# total_width = sum(widths)
# max_height = max(heights)

# new_im = Image.new('RGB', (total_width, max_height))

# x_offset = 0
# for im in images:
#     new_im.paste(im, (x_offset,0))
#     x_offset += im.size[0]

# new_im.save('test.png')
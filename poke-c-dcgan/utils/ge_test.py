import sys
from PIL import Image
import pandas as pd
import math
import os, glob
import shutil




save_dir = '../data/evol_pairs_mini_rgb_r10'
img_dir = '../data/pokemon-mini-rgb-r10/'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)


def crop_im(im):
    crop_rectangle = (5, 0, 35, 30)
    return im.crop(crop_rectangle)


csv_file = pd.read_csv('../data/pokemon-dex-ev.csv', nrows=721)
ev_idxs = csv_file.iloc[:, 32].values
pre_idxs = csv_file.iloc[:, 33].values


idx_pairs = list(zip(ev_idxs, pre_idxs))

idxs = []
for p in idx_pairs:
    if not math.isnan(p[1]):
        idxs.append(int(p[0]))
        idxs.append(int(p[1]))
uni_idxs = list(dict.fromkeys(idxs))
print(type(uni_idxs[0]))

test_idxs = []

for i in range(1, 722):
    i = '%03d' % i
    if not int(i) in uni_idxs:
        test_idxs.append(i)
print(len(test_idxs))

test_idxs.append('133')

for infile in glob.glob('../data/pokemon-mini-rgb' + "/*.png"):
    file, ext = os.path.splitext(infile)
    file_name = file.split('/')[-1]
    print(infile,type(infile))
    if file_name in test_idxs:
        shutil.copyfile(infile, '../data/no_evol/' + file_name + '.png') 


def blank_image():
    im = Image.new('RGB', (120,120), color=0)
    return im

# Merge test pairs

for infile in glob.glob('../data/no_evol' + "/*.png"):
    file, ext = os.path.splitext(infile)
    file_name = file.split('/')[-1]
    im_pair = []
    im2 = Image.open(infile)
    im1 = blank_image()
    im_pair.append(im1)
    im_pair.append(im2)
    widths, heights = zip(*(i.size for i in im_pair))
    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for im in im_pair:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]
    new_im.save('../data/test_pairs/' + file_name + '-test.png')


        
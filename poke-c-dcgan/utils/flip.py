from PIL import Image
import glob, os


src_dir = '../data/pokemon-mini-rgb'
save_dir = src_dir + '-f'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
    print("Directory " , save_dir ,  " Created ")

for infile in glob.glob(src_dir + "/*.png"):
    file, ext = os.path.splitext(infile)
    file_name = file.split('/')[-1]
    im = Image.open(infile)
    rotated_im = im.transpose(Image.FLIP_LEFT_RIGHT)
    rotated_im.save(save_dir + '/' + file_name + '.png', 'PNG')
print("Finished")

# for infile in glob.glob(src_dir + "/*.png"):
#     file, ext = os.path.splitext(infile)
#     file_name = file.split('/')[-1]
#     im = Image.open(infile)
#     rotated_im = im.rotate(5)
#     rotated_im.save(save_dir + '/' + file_name + '.png', 'PNG')
# print("Finished")

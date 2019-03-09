import requests
import os

img_nums = []
for i in range(1, 810):
    img_nums.append('%03d' % i)

save_dir = 'data/pokedex-images'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

for i in img_nums:
    img_data = requests.get('https://assets.pokemon.com/assets/cms2/img/pokedex/full/'+i+'.png').content
    with open(save_dir+'/'+i+'.png', 'wb') as handler:
        handler.write(img_data)
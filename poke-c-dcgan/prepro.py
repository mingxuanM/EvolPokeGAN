import csv
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-load_path', default='data/tags_clean.csv', help='Path to csv data file.')
parser.add_argument('-save_path', default='data/clean_labels.csv', help="Path to save cleaned data.")
parser.add_argument('-json_save_path', default='data/animegan_params.json', help="Path to save json.")
parser.add_argument('-color_count_threshold', default=25, help="Word count threshold.")

args = parser.parse_args()

color_counts = {}

with open(args.save_path, 'w') as g:
    writer = csv.writer(g)
    writer.writerow(["image_name", "eyes", "hair"])

    with open(args.load_path, 'r') as f:
        for ridx, row in enumerate(csv.reader(f)):
            tags = row[1].split('\t')
            attrib = {'eyes': '<UNK>', 'hair': '<UNK>'}

            for t in tags:
                if t != '':
                    tag = t.split(':')[0].strip()
                    s_tag = tag.split()
                    
                    if len(s_tag) != 2:
                        continue

                    w = s_tag[1]
                    if (s_tag[0] == 'long' or s_tag[0] == 'short'):
                        continue
                    if(w == 'eyes' or w == 'hair'):
                        if(attrib[w] != '<UNK>'):
                            continue

                        attrib[w] = s_tag[0]
                        color_counts[s_tag[0]] = color_counts.get(s_tag[0], 0) + 1
            
            writer.writerow([ridx, attrib['eyes'], attrib['hair']])

# Store the mapping for the words as a json
color_counts['<UNK>'] = args.color_count_threshold
color_counts_sorted = sorted(color_counts.items(), key=lambda kv: kv[1], reverse=True)

vocab = [color[0] for color in color_counts_sorted if color[1] >= args.color_count_threshold]

color2ind = {color: color_ind + 1 for color_ind, color in enumerate(vocab)}
ind2color = {color_ind: color for color, color_ind in color2ind.items()}

out = {}
out['ind2color'] = ind2color
out['color2ind'] = color2ind

with open(args.json_save_path, 'w') as fp:
    json.dump(out, fp)

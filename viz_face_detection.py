import argparse
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import numpy as np
import seaborn as sns
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_path', type=str, help='Path where the image is located')
    parser.add_argument('-e', '--embed_path', type=str, help='Path to the .npy file with embeddings and bounding boxes')
    return parser.parse_args()


def main(opts):
    img_path = Path(opts.image_path)
    if not img_path.exists():
        print('ERROR: image file does not exist or cannot be accessed')
        sys.out(-1)

    embed_path = Path(opts.embed_path)
    if not embed_path.exists():
        print('ERROR: embeddings file does not exist or cannot be accessed')
        sys.out(-1)

    embeddings = np.load(embed_path, allow_pickle=True).item()
    if not img_path.name in embeddings:
        print('ERROR: could not find the image filename in the embeddings npy')
        sys.out(0)

    img = Image.open(img_path)
    colors = sns.color_palette('Set2')
    fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 30)


    for idx, face in enumerate(embeddings[img_path.name]['faces']):
        canvas = ImageDraw.Draw(img)
        color = tuple([int(255*c) for c in colors[idx]])
        canvas.rectangle(face['bbox'], fill=None, outline=color)
        canvas.text((face['bbox'][0], face['bbox'][1]),
                    str(idx),
                    font=fnt,
                    fill=(color[0], color[1], color[2], 255))

    img.show()


if __name__ == "__main__":
    opts = parse_args()
    main(opts)
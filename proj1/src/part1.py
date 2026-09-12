import numpy as np
import skimage as sk
import skimage.io as skio
import os
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import time
from pathlib import Path

os.makedirs('out_path', exist_ok=True)
OUTPUT_ROOT = Path("out_path")


def save_file(fname, part, img, as_ubyte=True):
    out_dir = OUTPUT_ROOT / part
    out_dir.mkdir(exist_ok=True)

    path = out_dir / fname

    if as_ubyte and img.dtype != "uint8":
        img = sk.img_as_ubyte(img)

    skio.imsave(str(path), img)
    print(f"Saved: {path}")
    return path


def ncc_score(img1, img2):
    # calculate normalised cross-correlation

    # minus average brightness
    a = img1 - img1.mean()
    b = img2 - img2.mean()

    # compute L2 length, to normalise the sum of product
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return -np.inf
    return np.sum(a * b) / denom

def align(img, ref, border=0.1, window=15):
    # crop borders before scoring so edge artifacts don't dominate the metric

    # height & width of reference image
    h, w = ref.shape[:2]

    # compute how many pixels trim off each edge
    bh, bw = int(h * border), int(w * border)

    best_score = -np.inf
    best_dx, best_dy = 0, 0

    # iterate through every combination of vertical and horizontal shift
    for dy in range(-window, window + 1):
        for dx in range(-window, window + 1):
            shifted = np.roll(img, (dy, dx), axis=(0, 1))

            # crop both to the same interior region for scoring
            crop_shifted = shifted[bh:h-bh, bw:w-bw]
            crop_ref = ref[bh:h-bh, bw:w-bw]
            score = ncc_score(crop_shifted, crop_ref)

            # keep track of scores
            if score > best_score:
                best_score = score
                best_dx, best_dy = dx, dy

    # apply best shift to the full, uncropped img
    aligned = np.roll(img, (best_dy, best_dx), axis=(0, 1))
    return aligned, (best_dx, best_dy)

def align_channels(imname):
    """
    Loads a plate image, splits into B/G/R, aligns G and R to B.
    Returns (im_out, g_shift, r_shift) — im_out is float in [0,1], NOT cropped.
    """
    im = skio.imread(imname)
    print(f"Aligning image: {imname}")

    im = sk.img_as_float(im)
    height = np.floor(im.shape[0] / 3.0).astype(int)

    # separate color channels
    b = im[:height]
    g = im[height: 2*height]
    r = im[2*height: 3*height]

    # align the images
    ag, g_shift = align(g, b)
    ar, r_shift = align(r, b)
    print(f"G shift: {g_shift}")
    print(f"R shift: {r_shift}")

    # create a color image
    im_out = np.dstack([ar, ag, b])
    im_out = np.clip(im_out, 0, 1)

    return im_out


def single_scale_alignment(image_paths):
    for imname in image_paths:
        im_out = align_channels(imname)
        crop = 0.06
        h, w = im_out.shape[:2]
        bh, bw = int(h*crop), int(w*crop)
        im_out_cropped = im_out[bh:h-bh, bw:w-bw]

        img = sk.img_as_ubyte(im_out_cropped)
        
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    
        # save
        base = os.path.splitext(os.path.basename(imname))[0]
        fname = f'{base}_out.jpg'
        save_file(fname, "part1", im_out_cropped)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image_paths = [
        'CS180_fa2026_proj1_data/cathedral.jpg',
        'CS180_fa2026_proj1_data/monastery.jpg',
        'CS180_fa2026_proj1_data/tobolsk.jpg',
    ]

    single_scale_alignment(image_paths)
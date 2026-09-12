from skimage.filters import sobel
import numpy as np
import skimage as sk
import skimage.io as skio
import os
import matplotlib.pyplot as plt
from pathlib import Path
from part1 import *
from part2 import pyramid_align

os.makedirs('out_path', exist_ok=True)
OUTPUT_ROOT = Path("out_path")


def gradient_magnitude(img):
    if img.ndim == 3:
        img = img.mean(axis=2)
    gx = sobel(img, axis=1)
    gy = sobel(img, axis=0)
    return np.sqrt(gx**2 + gy**2)


def align_channels_pyramid_gradient(imname):
    im = skio.imread(imname)
    print(f"Aligning image: {imname}")

    im = sk.img_as_float(im)
    height = np.floor(im.shape[0] / 3.0).astype(int)

    b = im[:height]
    g = im[height:2*height]
    r = im[2*height:3*height]

    b_grad = gradient_magnitude(b)
    g_grad = gradient_magnitude(g)
    r_grad = gradient_magnitude(r)

    _, g_shift = pyramid_align(g_grad, b_grad)
    _, r_shift = pyramid_align(r_grad, b_grad)
    print(f"G shift: {g_shift}")
    print(f"R shift: {r_shift}")

    ag = np.roll(g, (g_shift[1], g_shift[0]), axis=(0, 1))
    ar = np.roll(r, (r_shift[1], r_shift[0]), axis=(0, 1))

    im_out = np.dstack([ar, ag, b])
    im_out = np.clip(im_out, 0, 1)

    return im_out, g_shift, r_shift

if __name__ == "__main__":
    imname = 'CS180_fa2026_proj1_data/emir.tif'
    im_out, g_shift, r_shift = align_channels_pyramid_gradient(imname)

    h, w = im_out.shape[:2]
    img_uncropped = sk.img_as_ubyte(im_out)

    crop = 0.07
    top = bottom = int(h * crop)
    left = right = int(w * crop)
    im_out_cropped = im_out[top:h-bottom, left:w-right]
    img_cropped = sk.img_as_ubyte(im_out_cropped)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_uncropped)
    axes[0].set_title(f"Uncropped: {imname}", fontsize=10)
    axes[0].axis('off')

    axes[1].imshow(img_cropped)
    axes[1].set_title(f"Cropped: (t{top},b{bottom},l{left},r{right})", fontsize=10)
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

    base = os.path.splitext(os.path.basename(imname))[0]
    fname = f'{base}_out.jpg'
    save_file(fname, "bw", img_cropped)
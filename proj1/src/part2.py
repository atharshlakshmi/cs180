import numpy as np
import skimage as sk
import skimage.io as skio
import os
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import time
from pathlib import Path
from part1 import *

os.makedirs('out_path', exist_ok=True)
OUTPUT_ROOT = Path("out_path")


def pyramid_align(img, ref, coarse_window=15, refine_window=4, border=0.1, min_size=100):
    h, w = ref.shape[:2]

    if min(h, w) < min_size:
        return align(img, ref, border=border, window=coarse_window)

    img_small = img[::2, ::2]
    ref_small = ref[::2, ::2]

    _, (dx_coarse, dy_coarse) = pyramid_align(
        img_small, ref_small, coarse_window, refine_window, border, min_size
    )

    dx_coarse, dy_coarse = dx_coarse * 2, dy_coarse * 2

    img_shifted = np.roll(img, (dy_coarse, dx_coarse), axis=(0, 1))
    _, (dx_refine, dy_refine) = align(img_shifted, ref, border=border, window=refine_window)

    total_dx, total_dy = dx_coarse + dx_refine, dy_coarse + dy_refine
    aligned = np.roll(img, (total_dy, total_dx), axis=(0, 1))
    return aligned, (total_dx, total_dy)


def align_channels_pyramid(imname):
    im = skio.imread(imname)
    print(f"Aligning image: {imname}")

    im = sk.img_as_float(im)
    height = np.floor(im.shape[0] / 3.0).astype(int)

    b = im[:height]
    g = im[height:2*height]
    r = im[2*height:3*height]

    ag, g_shift = pyramid_align(g, b)
    ar, r_shift = pyramid_align(r, b)
    print(f"G shift: {g_shift}")
    print(f"R shift: {r_shift}")

    im_out = np.dstack([ar, ag, b])
    im_out = np.clip(im_out, 0, 1)

    return im_out, g_shift, r_shift


def multi_scale_alignment(image_paths, crop_ratio = 0.07):
    results = {}

    for imname in tqdm(image_paths):
        start_time = time.perf_counter()
        im_out, g_shift, r_shift = align_channels_pyramid(imname)
        h, w = im_out.shape[:2]

        img_uncropped = sk.img_as_ubyte(im_out)
        top = bottom = int(h * crop_ratio)
        left = right = int(w * crop_ratio)


        im_out_cropped = im_out[top:h-bottom, left:w-right]
        img_cropped = sk.img_as_ubyte(im_out_cropped)

        end_time = time.perf_counter()
        print(f"{end_time - start_time} seconds to run.")

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(img_uncropped)
        axes[0].set_title(f"Uncropped: {imname}", fontsize=10)
        axes[0].axis('off')

        axes[1].imshow(img_cropped)
        axes[1].set_title(
            f"Cropped: (t{top},b{bottom},l{left},r{right})", fontsize=10
        )
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()

        base = os.path.splitext(os.path.basename(imname))[0]
        fname = f'{base}_out.jpg'
        save_file(fname, "part2", img_cropped)
        results[imname] = {"G": g_shift, "R": r_shift}

    return results


def get_image_paths(folder, extensions=(".jpg", ".jpeg", ".tif", ".tiff", ".png")):
    paths = []
    for ext in extensions:
        paths.extend(glob.glob(os.path.join(folder, f"*{ext}")))
    return sorted(paths)


if __name__ == "__main__":
    image_folder = 'CS180_fa2026_proj1_data'
    image_paths = get_image_paths(image_folder)

    results = multi_scale_alignment(image_paths, crop_ratio=0.07)

    print("\n=== Offset summary ===")
    print(f"{'image':>25} | {'G shift':>12} | {'R shift':>12}")
    for imname, shifts in results.items():
        print(f"{os.path.basename(imname):>25} | {str(shifts['G']):>12} | {str(shifts['R']):>12}")
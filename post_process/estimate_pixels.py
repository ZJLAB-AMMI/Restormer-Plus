import os
import pickle
from glob import glob
from pathlib import Path

import numpy as np
from PIL import Image
from natsort import natsorted

# ==========config
f"""
    test_median_res_dir: the directory of the median result of test data, achieved by running median_derain.py
    train_median_res_dir: the directory of the median result of train data, achieved by running median_derain.py
    pixels_file: a .pkl file where contains the position info of the pixels whose values require to be estimated. 
        Format: a dict, the key is the scene name, the value is a list of pixel-position.
    save_dir: where to save the similar patches.
    patch_size: the size of the patch.
    min_dis: the threshold used to select similar patches.
"""
test_median_res_dir = '/gt-rain/result/test_median'
train_median_res_dir = '/gt-rain/result/train_median'
pixels_file = '/gt-rain/result/pixels.pkl'
save_dir = '/gt-rain/result/similar_patch'
patch_size = 8
min_dis = 6
# ==========

pixels = pickle.load(open(pixels_file, 'rb'))


def get_img_paths(data_dir):
    scene_names = []
    for sc in list(os.walk(data_dir))[0][1]:
        scene_names.append(sc)
    img_paths = []
    for scene in scene_names:
        img_paths.append(
            (
                natsorted(glob(os.path.join(data_dir, scene, '*-R-*.png')))[0],
                natsorted(glob(os.path.join(data_dir, scene, '*-C-*.png')))[0]
            )
        )
    return img_paths


for scene_name, pixels_pos in pixels.items():
    test_median_res = np.asarray(Image.open(os.path.join(test_median_res_dir, scene_name, '1_r.png')))
    hts, wts, cts = test_median_res.shape
    train_img_paths = get_img_paths(train_median_res_dir)

    for pixel_pos in pixels_pos:
        save_path = f"{save_dir}/{scene_name}/{str(pixel_pos[0]) + '_' + str(pixel_pos[1])}"
        Path(save_path).mkdir(parents=True, exist_ok=True)

        # test patch
        hts1 = np.clip(pixel_pos[0] - patch_size // 2, 0, hts)
        hts2 = np.clip(pixel_pos[0] + patch_size // 2, 0, hts)
        wts1 = np.clip(pixel_pos[1] - patch_size // 2, 0, wts)
        wts2 = np.clip(pixel_pos[1] + patch_size // 2, 0, wts)
        test_patch = test_median_res[hts1: hts2, wts1: wts2, :]
        Image.fromarray(test_patch).save(f"{save_path}/test_patch.png")

        h_patch_size = hts2 - hts1
        w_patch_size = wts2 - wts1

        # search and save similar patch in train data
        for train_median_res_file, train_clean_file in train_img_paths:
            train_median_res = np.asarray(Image.open(train_median_res_file))
            train_clean = np.asarray(Image.open(train_clean_file))
            htr, wtr, ctr = train_median_res.shape
            h_gap = (htr - h_patch_size) // 30
            w_gap = (wtr - w_patch_size) // 30
            for h_idx in range(0, htr - h_patch_size, h_gap):
                for w_idx in range(0, wtr - w_patch_size, w_gap):
                    train_median_patch = train_median_res[h_idx: (h_idx + h_patch_size), w_idx: (w_idx + w_patch_size),
                                         :]
                    train_clean_patch = train_clean[h_idx: (h_idx + h_patch_size), w_idx: (w_idx + w_patch_size), :]

                    distance = np.median(
                        np.abs(test_patch.flatten() - train_median_patch.flatten())
                    )

                    if distance <= min_dis:
                        pred_val = np.mean(train_clean_patch, axis=(0, 1))
                        Image.fromarray(train_median_patch).save(
                            os.path.join(save_path,
                                         'train_median_patch_{}_{}_{}.png'.format(h_idx, w_idx, np.round(distance, 3))))
                        Image.fromarray(train_clean_patch).save(os.path.join(save_path,
                                                                             'train_clean_patch_{}_{}_{}.png'.format(
                                                                                 h_idx,
                                                                                 w_idx,
                                                                                 np.round(pred_val, 3))))

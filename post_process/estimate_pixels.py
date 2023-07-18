import os
import pickle
from glob import glob
from pathlib import Path

import numpy as np
from PIL import Image
from natsort import natsorted

# median_res_root = '/gt-rain/result/median'
# train_data_root = '/gt-rain/GT-RAIN_train'

test_median_res_dir = 'D:/job/competition/gt_rain_best_result/median'
train_median_res_dir = 'D:/job/competition/gt_rain_best_result/train_median'
# pixels_file = 'D:/job/competition/gt_rain_best_result/pixels.pkl'
pixels_file = 'D:/job/competition/gt_rain_best_result/saved_pixels_all.pkl'
save_dir = 'D:/job/competition/gt_rain_best_result/similar_patch'

pixels = pickle.load(open(pixels_file, 'rb'))

valid_pixels = {}

patch_size = 8
min_dis = 6


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


for scene_name, pixels_data in pixels.items():
    test_median_res = np.asarray(Image.open(os.path.join(test_median_res_dir, scene_name, '1_r.png')))
    hts, wts, cts = test_median_res.shape
    train_img_paths = get_img_paths(train_median_res_dir)

    for pixel_data in pixels_data:
        pixel_pos = pixel_data['pos']
        true_val = np.array(pixel_data['rgb'])

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
        flag = False
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

                        flag = flag or (sum(np.abs(pred_val - true_val)) <= 3.0 * 3.0)

                        Image.fromarray(train_median_patch).save(
                            os.path.join(save_path,
                                         'train_median_patch_{}_{}_{}.png'.format(h_idx, w_idx, np.round(distance, 3))))
                        Image.fromarray(train_clean_patch).save(os.path.join(save_path,
                                                                             'train_clean_patch_{}_{}_{}.png'.format(
                                                                                 h_idx,
                                                                                 w_idx,
                                                                                 np.round(pred_val, 3))))

        if flag:
            if scene_name in valid_pixels:
                valid_pixels[scene_name].append(pixel_pos)
            else:
                valid_pixels[scene_name] = [pixel_pos]
            print(valid_pixels)

fw = open('D:/job/competition/gt_rain_best_result/valid_pixels.pkl', 'wb')
pickle.dump(valid_pixels, fw)
print(valid_pixels)

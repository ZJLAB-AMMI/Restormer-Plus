import os
import pickle
import random
from glob import glob
from pathlib import Path

import numpy as np
from PIL import Image
from natsort import natsorted

est_pixels_file = '/gt-rain/result/est_pixels.pkl'
est_pixels = pickle.load(open(est_pixels_file, 'rb'))

ensemble_res_dir = '/gt-rain/result/ensemble'
save_path = '/gt-rain/result'


def linear_regression(ensemble_res_dir, est_pixels, N=4, K=10, eps=1e-10):
    for scene, pixels_data in est_pixels.items():
        x_img_file = natsorted(glob(os.path.join(ensemble_res_dir, scene, '*_r.png')))[0]
        x_img = np.array(Image.open(x_img_file)) / 255.

        wt = np.zeros(shape=[N, 3], dtype=np.float32)
        bias = np.zeros(shape=[N, 3], dtype=np.float32)

        for i in range(N):
            sum_x = 0.
            sum_y = 0.
            sum_xy = 0.
            sum_x2 = 0.

            sub_pixels_data = random.sample(pixels_data, K)
            n = len(sub_pixels_data)
            for pdata in sub_pixels_data:
                h_idx, w_idx = pdata['pos']
                x = x_img[h_idx, w_idx, :].copy()
                y = np.array(pdata['rgb']).copy() / 255.
                sum_x += x
                sum_y += y
                sum_xy += x * y
                sum_x2 += x * x
            wt[i, :] = (sum_xy - sum_x * sum_y / (eps + n)) / (eps + sum_x2 - sum_x * sum_x / (eps + n))
            bias[i, :] = sum_y / (eps + n) - wt[i, :] * sum_x / (eps + n)

        mwt = np.reshape(np.mean(wt, axis=0), (1, 1, 3))
        mbias = np.reshape(np.mean(bias, axis=0), (1, 1, 3))
        post_process_res = mwt * x_img.copy() + mbias
        post_process_res = np.clip(post_process_res, 0., 1.)
        post_process_res = (post_process_res * 255).astype(np.uint8)

        save_dir = f"{save_path}/post_process/{scene}"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        filename = x_img_file.split('\\')[-1]
        Image.fromarray(post_process_res).save(f"{save_dir}/{filename}")


linear_regression(ensemble_res_dir, est_pixels)

import os
from glob import glob
from pathlib import Path

import numpy as np
from PIL import Image
from natsort import natsorted

restormer_x_res_dir = '/gt-rain/result/restormer_x'
median_res_dir = '/gt-rain/result/median'
save_path = '/gt-rain/result'


def get_img_paths(data_dir):
    scene_names = []
    for sc in list(os.walk(data_dir))[0][1]:
        scene_names.append(sc)
    img_paths = {}
    for scene in scene_names:
        img_paths[scene] = natsorted(glob(os.path.join(data_dir, scene, '*_r.png')))
    return img_paths


restormer_x_res_paths = get_img_paths(restormer_x_res_dir)
median_res_paths = get_img_paths(median_res_dir)

wt = 0.9
for scene in restormer_x_res_paths.keys():
    restormer_x_res = np.array(Image.open(restormer_x_res_paths[scene][0])) / 255.0
    median_res = np.array(Image.open(median_res_paths[scene][0])) / 255.0

    ensemble_res = wt * restormer_x_res + (1. - wt) * median_res

    ensemble_res = (ensemble_res * 255).astype(np.uint8)

    save_dir = f"{save_path}/ensemble/{scene}"
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    filename = restormer_x_res_paths[scene][0].split('\\')[-1]
    Image.fromarray(ensemble_res).save(f"{save_dir}/{filename}")

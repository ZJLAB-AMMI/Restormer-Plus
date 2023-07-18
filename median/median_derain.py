import os
import shutil
from glob import glob
from pathlib import Path

import numpy as np
from PIL import Image
from natsort import natsorted


is_train = True
if is_train:
    data_dir = '/gt-rain/GT-RAIN_train'
    save_path = '/gt-rain/result'
else:
    data_dir = '/gt-rain/GT-RAIN_test'
    save_path = '/gt-rain/result'


def get_img_paths(data_dir, is_train=False):
    scene_names = []
    for sc in list(os.walk(data_dir))[0][1]:
        scene_names.append(sc)
    img_paths = {}
    clean_img_path = {} if is_train else None
    for scene in scene_names:
        if is_train:
            img_paths[scene] = natsorted(glob(os.path.join(data_dir, scene, '*-R-*.png')))
            clean_img_path[scene] = natsorted(glob(os.path.join(data_dir, scene, '*-C-*.png')))[0]
        else:
            img_paths[scene] = natsorted(glob(os.path.join(data_dir, scene, '*_r.png')))
    return img_paths, clean_img_path


img_paths, clean_img_path = get_img_paths(data_dir, is_train)

for scene, scene_img_paths in img_paths.items():

    img_list = []
    for img_path in scene_img_paths:
        img = Image.open(img_path)
        img = np.array(img) / 255.0
        img_list.append(img)
    median_res = np.median(np.stack(img_list, axis=-1), axis=-1)
    median_res = (median_res * 255).astype(np.uint8)
    if is_train:
        save_dir = f"{save_path}/train_median/{scene}"
    else:
        save_dir = f"{save_path}/test_median/{scene}"
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    filename = scene_img_paths[0].split('\\')[-1]
    Image.fromarray(median_res).save(f"{save_dir}/{filename}")
    if is_train:
        filename = clean_img_path[scene].split('\\')[-1]
        shutil.copyfile(clean_img_path[scene], f"{save_dir}/{filename}")

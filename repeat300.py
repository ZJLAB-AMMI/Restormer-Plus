import os
import shutil
from glob import glob

from natsort import natsorted

root_dir = '/gt-rain/result/post_process'
scene_names = []
for sc in list(os.walk(root_dir))[0][1]:
    scene_names.append(sc)

img_paths = {}
for scene in scene_names:
    scene_path = os.path.join(root_dir, scene)
    scene_img_paths = natsorted(glob(os.path.join(scene_path, '*_r.png')))
    img_paths[scene] = scene_img_paths

for scene_name, im_paths in img_paths.items():
    print(scene_name)
    origin_file = im_paths[0]
    for idx in range(2, 301):
        new_file = origin_file[:-7] + '{}_r.png'.format(idx)
        shutil.copyfile(origin_file, new_file)

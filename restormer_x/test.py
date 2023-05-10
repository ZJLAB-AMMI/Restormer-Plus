import logging
import os
from pathlib import Path

import torch

from restormer_x.model.restormer import get_model
from restormer_x.utils.log import set_logger
from restormer_x.utils.trainutil import predict

os.environ["CUDA_VISIBLE_DEVICES"] = '6'

# CONFIG
params = {
    # general
    'save_dir': '/gt-rain/model',  # Dir to save the model weights
    'result_dir': '/gt-rain/result',
    'method_name': 'restormer_x',

    # data
    'val_dir_list': ['/gt-rain/GT-RAIN_val'],  # Dir for the val data
    'test_dir_list': ['/gt-rain/GT-RAIN_test'],  # Dir for the val data

    # model
    'model_version': 'base',
    'resume_epoch': 11,  # begin training using loaded checkpoint
}

# INIT
save_path = os.path.join(params['save_dir'], params['method_name'])
Path(save_path).mkdir(parents=True, exist_ok=True)
set_logger(save_path, 'test.log')
logging.info(str(params))

# MODEL

model = get_model(model_version=params['model_version'])

resume_epoch = params['resume_epoch']
resume_file = os.path.join(save_path, f'model_epoch_{resume_epoch}.pth')
checkpoint = torch.load(resume_file)
model.load_state_dict(checkpoint['state_dict'], strict=False)

# EVALUATE OR TEST

is_test = True
psnr_res = predict(
    model,
    params['test_dir_list'][0] if is_test else params['val_dir_list'][0],
    is_test=is_test,
    save_path=params['result_dir'],
    method_name=params['method_name']
)
logging.info(psnr_res)

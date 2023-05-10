import logging
import os
import time
from pathlib import Path

import tabulate
import torch
import torch.nn as nn

from restormer_x.dataset.gt_rain_dataset import get_datasets
from restormer_x.model.restormer import get_model
from restormer_x.utils.log import set_logger
from restormer_x.utils.loss import ShiftMSSSIM
from restormer_x.utils.trainutil import get_train_settings, train

os.environ["CUDA_VISIBLE_DEVICES"] = '6'

# CONFIG
params = {
    # general
    'method_name': 'restormer_x',
    # data
    'train_dir_list': ['/gt-rain/GT-RAIN_train'],  # Dir for the training data
    'rain_mask_dir': '/gt-rain/Streaks_Garg06',  # Dir for the rain masks
    'img_size': 256,  # the size of image input
    'zoom_min': .06,  # the minimum zoom for RainMix
    'zoom_max': 1.8,  # the maximum zoom for RainMix
    'batch_size': 2,  # batch size

    # model
    'model_version': 'base',
    'pretrained_model': '/pre-train-model/gt_rain/restormer_deraining.pth',

    # train
    'ssim_kernel_size': 11,  # img_size >= (kernel_size - 1) * 16 + 1
    'initial_lr': 3e-4,  # initial learning rate used by scheduler
    'weight_decay': 1e-4,
    'num_epochs': 20,  # number of epochs to train
    'warmup_epochs': 4,  # number of epochs for warmup
    'min_lr': 1e-6,  # minimum learning rate used by scheduler
    'mixmethod': 'mixup',
    'mix_prob': 0.5,
    'ssim_loss_weight': 0.0,  # weight for the ssim loss
    'acc_grad_step': 4,
    'save_freq': 1,
    'save_dir': '/gt-rain/model',  # Dir to save the model weights
}

# INIT

save_path = os.path.join(params['save_dir'], params['method_name'])
Path(save_path).mkdir(parents=True, exist_ok=True)
set_logger(save_path, 'train.log')
logging.info(str(params))

# DATA

train_loader = get_datasets(params)

# MODEL

model = get_model(model_version=params['model_version'])

if params['pretrained_model'] is not None:
    model.load_state_dict(torch.load(params['pretrained_model'])['params'], strict=False)

# LOSS

criterion_l1 = nn.L1Loss().cuda()
criterion_ssim = ShiftMSSSIM(ssim_kernel_size=params['ssim_kernel_size']).cuda()

# TRAIN

optimizer, scheduler = get_train_settings(model, params)

start_epoch = 0

for epoch in range(start_epoch, params['num_epochs']):
    time_ep = time.time()

    train_res = train(model, train_loader, optimizer, scheduler, criterion_l1, criterion_ssim, params)

    if ((epoch + 1) % params['save_freq'] == 0) or ((epoch + 1) == params['num_epochs']):
        torch.save(
            {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            },
            os.path.join(save_path, f'model_epoch_{epoch}.pth')
        )

    time_ep = time.time() - time_ep
    columns = ["epoch", "learning_rate",
               "train_loss", "train_ssim_loss", "train_l1_loss",
               "cost_time"]

    values = [epoch + 1, optimizer.param_groups[0]['lr'],
              train_res["total_loss"], train_res["ssim_loss"], train_res["l1_loss"],
              time_ep]

    table = tabulate.tabulate([values], columns, tablefmt="simple", floatfmt="8.4f")
    if epoch % 50 == 0:
        table = table.split("\n")
        table = "\n".join([table[1]] + table)
    else:
        table = table.split("\n")[2]

    logging.info(table)

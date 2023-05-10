import os
from glob import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms.functional as TF
from PIL import Image
from natsort import natsorted
from skimage.metrics import peak_signal_noise_ratio as psnr
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau

from restormer_x.utils.mixmethod import mixup
from restormer_x.utils.loss import AverageMeter


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                    self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                         self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


def get_train_settings(model, params):
    optimizer = optim.AdamW(
        model.parameters(),
        lr=params['initial_lr'],
        weight_decay=params['weight_decay']
    )

    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        params['num_epochs'] - params['warmup_epochs'],
        eta_min=params['min_lr'])

    scheduler = GradualWarmupScheduler(
        optimizer,
        multiplier=1.0,
        total_epoch=params['warmup_epochs'],
        after_scheduler=scheduler_cosine
    )

    optimizer.zero_grad()
    optimizer.step()
    scheduler.step()  # To start warmup

    return optimizer, scheduler


def train(model, train_loader, optimizer, scheduler, criterion_l1, criterion_ssim, params):
    model.train()

    total_losses = AverageMeter()
    l1_losses = AverageMeter()
    ssim_losses = AverageMeter()
    num_batchs = len(train_loader.dataset) // params['batch_size']
    for batch_idx, batch_data in enumerate(train_loader):
        input_img = batch_data['input_img'].cuda()
        target_img = batch_data['target_img'].cuda()

        if (params['mixmethod'] == 'mixup') and (np.random.rand(1) <= params['mix_prob']):
            input_img, target_img = mixup(input_img, target_img)

        output_img = model(input_img)

        l1_loss = criterion_l1(output_img, target_img)
        loss = l1_loss
        l1_losses.add(l1_loss.item(), input_img.size(0))

        if params['ssim_loss_weight'] > 0:
            ssim_loss = criterion_ssim(output_img.clip(0., 1.), target_img)
            loss += params['ssim_loss_weight'] * ssim_loss
            ssim_losses.add(ssim_loss.item(), input_img.size(0))

        total_losses.add(loss.item(), input_img.size(0))

        acc_grad_step = params['acc_grad_step']
        loss = loss / acc_grad_step
        loss.backward()

        if (((batch_idx + 1) % acc_grad_step) == 0) or ((batch_idx + 1) == num_batchs):
            optimizer.step()
            optimizer.zero_grad()

    scheduler.step()

    return {
        'total_loss': total_losses.value(),
        'ssim_loss': ssim_losses.value(),
        'l1_loss': l1_losses.value()
    }


def predict(model, root_dir, is_test=False, eta=8, save_path=None, method_name=None):
    model.eval()
    scene_names = []
    for sc in list(os.walk(root_dir))[0][1]:
        scene_names.append(sc)

    img_paths = {}
    for scene in scene_names:
        scene_path = os.path.join(root_dir, scene)
        if is_test:
            scene_img_paths = natsorted(glob(os.path.join(scene_path, '*_r.png')))
        else:
            scene_img_paths = natsorted(glob(os.path.join(scene_path, '*R-*.png')))
        img_paths[scene] = scene_img_paths

    mean_output = {}
    with torch.no_grad():
        for scene_name, im_paths in img_paths.items():
            print(scene_name)
            if scene_name not in mean_output:
                mean_output[scene_name] = {'sum_im': 0.0, 'num_im': 0}
            for im_path in im_paths:
                img = Image.open(im_path)
                img = np.array(img)
                img = TF.to_tensor(img)  # [c, h, w]
                h, w = img.shape[1:]
                padw = eta - (w % eta) if (w % eta) != 0 else 0
                padh = eta - (h % eta) if (h % eta) != 0 else 0
                if padw != 0 or padh != 0:
                    img = F.pad(img, (0, padw, 0, padh), mode='reflect')

                input = torch.unsqueeze(img, 0).cuda()
                output = model(input)
                output = output.squeeze().permute((1, 2, 0))
                output = output.detach().cpu().numpy()[:h, :w, :]

                mean_output[scene_name]['sum_im'] += output
                mean_output[scene_name]['num_im'] += 1

    psnr_res = {'scene_psnr': {}, 'psnr': [0.0]}
    for scene_name, res in mean_output.items():
        output = res['sum_im'] / res['num_im']
        output = np.clip(output, 0.0, 1.0)
        if not is_test:
            tmp = img_paths[scene_name][0]
            tar_path = tmp[:-9] + 'C-000.png'
            if 'Gurutto_1-2' in im_path:
                tar_path = tmp[:-9] + 'C' + tmp[-8:]
            tar_img = Image.open(tar_path)
            tar_img = np.array(tar_img, dtype=np.float32)
            tar_img = tar_img / 255  # [h, w, c]

            psnr_val = psnr(tar_img, output)
            psnr_res['scene_psnr'][scene_name] = psnr_val
            psnr_res['psnr'] += psnr_val
        else:
            save_dir = f"{save_path}/{method_name}/test/{scene_name}"
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            output = (output * 255).astype(np.uint8)
            filename = img_paths[scene_name][0].split('/')[-1]
            Image.fromarray(output).save(f"{save_dir}/{filename}")
    psnr_res['psnr'][0] /= len(mean_output.keys())
    return psnr_res

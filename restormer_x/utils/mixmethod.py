import numpy as np
import torch


def rand_bbox(size, lam):
    H = size[2]
    W = size[3]

    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def mixup(input_image, target_image, alpha=1.0):
    """

    :param alpha:
    :param input_image: [bs, c, h, w]
    :param target_image:
    :return:
    """
    image_shape = input_image.shape
    rand_index = torch.randperm(image_shape[0]).to(input_image.device)
    lam = np.random.beta(alpha, alpha)

    input_image = lam * input_image + (1.0 - lam) * input_image[rand_index]
    target_image = lam * target_image + (1.0 - lam) * target_image[rand_index]

    return input_image, target_image


def cutmix(input_image, target_image, alpha=1.0):
    image_shape = input_image.shape
    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(image_shape, lam)

    rand_index = torch.randperm(image_shape[0]).to(input_image.device)

    input_image[:, :, bby1: bby2, bbx1: bbx2] = input_image[rand_index][:, :, bby1: bby2, bbx1: bbx2]
    target_image[:, :, bby1: bby2, bbx1: bbx2] = target_image[rand_index][:, :, bby1: bby2, bbx1: bbx2]
    return input_image, target_image

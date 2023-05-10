import torch
import torch.nn as nn
import torch.nn.functional as F
from piq import MultiScaleSSIMLoss


class ShiftMSSSIM(torch.nn.Module):
    """Shifted SSIM Loss """

    def __init__(self, ssim_kernel_size=11):
        super(ShiftMSSSIM, self).__init__()
        self.ssim = MultiScaleSSIMLoss(kernel_size=ssim_kernel_size, data_range=1.)

    def forward(self, est, gt):
        # shift images back into range (0, 1)
        # est = est * 0.5 + 0.5
        # gt = gt * 0.5 + 0.5
        return self.ssim(est, gt)


class RainRobustLoss(torch.nn.Module):
    """Rain Robust Loss"""

    def __init__(self, batch_size, n_views, device, temperature=0.07):
        super(RainRobustLoss, self).__init__()
        self.batch_size = batch_size
        self.n_views = n_views
        self.temperature = temperature
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

    def forward(self, features):
        logits, labels = self.info_nce_loss(features)
        return self.criterion(logits, labels)

    def info_nce_loss(self, features):
        labels = torch.cat([torch.arange(self.batch_size) for i in range(self.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.temperature
        return logits, labels


def rain_robust_loss(params):
    return RainRobustLoss(
        batch_size=params['batch_size'],
        n_views=2,
        device=torch.device("cuda"),
        temperature=params['temperature']
    ).cuda()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def add(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def value(self):
        return self.sum / self.count if self.count > 0 else 0.0


class AverageAccMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def add(self, output, target):
        n = output.size(0)
        self.val = self.accuracy(output, target).item()
        self.sum += self.val * n
        self.count += n

    def value(self):
        if self.sum == 0:
            return 0
        else:
            return self.sum / self.count

    def accuracy(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res[0]

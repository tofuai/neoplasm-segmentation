import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def split_mask(neo_mask):
    ignore_mask = neo_mask[:, [0], :, :]
    # sum of ignore, neo and non-neo
    polyp_mask = ignore_mask + neo_mask[:, [1], :, :] + neo_mask[:, [2], :, :]
    # neo, non-neo and background
    neo_mask = neo_mask[:, [1, 2, 3], :, :]
    
    return polyp_mask, neo_mask, ignore_mask


def IoUScore(inputs, targets, smooth=1, ignore=None):
    if inputs.shape[1] == 1:
        inputs = torch.sigmoid(inputs)
        inputs = (inputs > 0.5).type(inputs.dtype)
    else:
        inputs = torch.argmax(inputs, dim=1, keepdims=True)
        neo = torch.all(inputs == 0, axis=1).type(inputs.dtype)
        non = torch.all(inputs == 1, axis=1).type(inputs.dtype)
        bg = torch.all(inputs == 2, axis=1).type(inputs.dtype)
        inputs = torch.stack([neo, non, bg], axis=1)

    if ignore is None:
        tp = (inputs * targets).sum(dim=(0, 2, 3))
        fp = (inputs).sum(dim=(0, 2, 3)) - tp
        fn = (targets).sum(dim=(0, 2, 3)) - tp
    else:
        ignore = (1-ignore).expand(-1, targets.shape[1], -1, -1)
        tp = (inputs * targets * ignore).sum(dim=(0, 2, 3))
        fp = (inputs * ignore).sum(dim=(0, 2, 3)) - tp
        fn = (targets * ignore).sum(dim=(0, 2, 3)) - tp
        
    iou = (tp + smooth) / (tp + fp + fn + smooth)

    return iou.mean()


def DiceScore(inputs, targets, smooth=1, ignore=None):
    if inputs.shape[1] == 1:
        inputs = torch.sigmoid(inputs)
        inputs = (inputs > 0.5).type(inputs.dtype)
    else:
        inputs = torch.argmax(inputs, dim=1, keepdims=True)
        neo = torch.all(inputs == 0, axis=1).type(inputs.dtype)
        non = torch.all(inputs == 1, axis=1).type(inputs.dtype)
        bg = torch.all(inputs == 2, axis=1).type(inputs.dtype)
        inputs = torch.stack([neo, non, bg], axis=1)

    if ignore is None:
        tp = (inputs * targets).sum(dim=(0, 2, 3))
        fp = (inputs).sum(dim=(0, 2, 3)) - tp
        fn = (targets).sum(dim=(0, 2, 3)) - tp
    else:
        ignore = (1-ignore).expand(-1, targets.shape[1], -1, -1)
        tp = (inputs * targets * ignore).sum(dim=(0, 2, 3))
        fp = (inputs * ignore).sum(dim=(0, 2, 3)) - tp
        fn = (targets * ignore).sum(dim=(0, 2, 3)) - tp
        
    iou = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)

    return iou.mean()


class IoU(nn.Module):
    __name__ = 'iou_score'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, y_prs, mask):
        _, neo_mask, ignore_mask = split_mask(mask)

        if isinstance(y_prs, tuple):
            y_pr = y_prs[-1]
        else:
            y_pr = y_prs

        score = IoUScore(y_pr, neo_mask, ignore=ignore_mask)

        return score


class Dice(nn.Module):
    __name__ = 'dice_score'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, y_prs, mask):
        _, neo_mask, ignore_mask = split_mask(mask)

        if isinstance(y_prs, tuple):
            y_pr = y_prs[-1]
        else:
            y_pr = y_prs

        score = DiceScore(y_pr, neo_mask, ignore=ignore_mask)

        return score
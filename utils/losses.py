import torch
import torch.nn as nn
import torch.nn.functional as F


def split_mask(neo_mask):
    ignore_mask = neo_mask[:, [0], :, :]
    # sum of ignore, neo and non-neo
    polyp_mask = ignore_mask + neo_mask[:, [1], :, :] + neo_mask[:, [2], :, :]
    # neo, non-neo and background
    neo_mask = neo_mask[:, [1, 2, 3], :, :]
    
    return polyp_mask, neo_mask, ignore_mask


def CELoss(inputs, targets, ignore=None):
    if inputs.shape[1] == 1:
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    else:
        ce_loss = F.cross_entropy(inputs, torch.argmax(targets, axis=1), reduction='none')

    if ignore is not None:
        ignore = 1 - ignore.squeeze()
        ce_loss = ce_loss * ignore

    return ce_loss.mean()


def FocalTverskyLoss(inputs, targets, alpha=0.7, beta=0.3, gamma=4/3, smooth=1, ignore=None):
    if inputs.shape[1] == 1:
        inputs = torch.sigmoid(inputs)
    else:
        inputs = torch.softmax(inputs, dim=1)

    if ignore is None:
        tp = (inputs * targets).sum(dim=(0, 2, 3))
        fp = (inputs).sum(dim=(0, 2, 3)) - tp
        fn = (targets).sum(dim=(0, 2, 3)) - tp
    else:
        ignore = (1-ignore).expand(-1, targets.shape[1], -1, -1)
        tp = (inputs * targets * ignore).sum(dim=(0, 2, 3))
        fp = (inputs * ignore).sum(dim=(0, 2, 3)) - tp
        fn = (targets * ignore).sum(dim=(0, 2, 3)) - tp
    
    ft_score = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    ft_loss = (1 - ft_score) ** gamma
    
    return ft_loss.mean()


class BlazeNeoLoss(nn.Module):
    __name__ = 'blazeneo_loss'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, y_prs, mask):
        polyp_mask, neo_mask, ignore_mask = split_mask(mask)

        ce_loss = CELoss(y_prs[0], polyp_mask)
        ft_loss = FocalTverskyLoss(y_prs[0], polyp_mask)
        aux_loss = ce_loss + ft_loss

        ce_loss = CELoss(y_prs[-1], neo_mask, ignore=ignore_mask)
        ft_loss = FocalTverskyLoss(y_prs[-1], neo_mask, ignore=ignore_mask)
        main_loss = ce_loss + ft_loss

        return aux_loss + main_loss


class UNetLoss(nn.Module):
    __name__ = 'unet_loss'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, y_pr, mask):
        polyp_mask, neo_mask, ignore_mask = split_mask(mask)

        ce_loss = CELoss(y_pr, neo_mask, ignore=ignore_mask)
        ft_loss = FocalTverskyLoss(y_pr, neo_mask, ignore=ignore_mask)
        main_loss = ce_loss + ft_loss

        return main_loss


class HarDNetMSEGLoss(nn.Module):
    __name__ = 'hardnetmseg_loss'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, y_pr, mask):
        polyp_mask, neo_mask, ignore_mask = split_mask(mask)

        ce_loss = CELoss(y_pr, neo_mask, ignore=ignore_mask)
        ft_loss = FocalTverskyLoss(y_pr, neo_mask, ignore=ignore_mask)
        main_loss = ce_loss + ft_loss

        return main_loss


class PraNetLoss(nn.Module):
    __name__ = 'pranet_loss'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, y_prs, mask):
        main_loss = 0
        for y_pr in y_prs:
            polyp_mask, neo_mask, ignore_mask = split_mask(mask)

            ce_loss = CELoss(y_pr, neo_mask, ignore=ignore_mask)
            ft_loss = FocalTverskyLoss(y_pr, neo_mask, ignore=ignore_mask)
            main_loss += ce_loss + ft_loss

        return main_loss


class NeoUNetLoss(nn.Module):
    __name__ = 'neounet_loss'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, y_prs, mask):
        main_loss = 0

        neo_gt = mask[:, [1], :, :]
        non_gt = mask[:, [2], :, :]
        polyp_gt = neo_gt + non_gt
        
        for y_pr in y_prs:
            neo_pr = y_pr[:, [0], :, :]
            non_pr = y_pr[:, [1], :, :]
            # neo_pr = (neo_pr > non_pr) * neo_pr
            # non_pr = (non_pr > neo_pr) * non_pr
            polyp_pr = (neo_pr > non_pr) * neo_pr + (non_pr > neo_pr) * non_pr

            main_loss += (CELoss(neo_pr, neo_gt) + FocalTverskyLoss(neo_pr, neo_gt) + \
                    CELoss(non_pr, non_gt) + FocalTverskyLoss(non_pr, non_gt) + \
                    CELoss(polyp_pr, polyp_gt) + FocalTverskyLoss(polyp_pr, polyp_gt)) / len(y_prs)

        return main_loss
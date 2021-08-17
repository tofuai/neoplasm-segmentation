import torch
import warnings
import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWarmupLR(_LRScheduler):
    def __init__(self,
        optimizer: Optimizer,
        T_max: int, 
        eta_min: float = 0,
        last_epoch: int = -1,
        warmup_epochs=5
    ) -> None:
        self.T_max = T_max
        self.eta_min = eta_min
        self.warmup_epochs = warmup_epochs

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        offset_epoch = self.last_epoch - self.warmup_epochs
        if self.last_epoch < self.warmup_epochs:
            return [
                lr * (self.last_epoch + 1) / self.warmup_epochs
                for lr in self.base_lrs
            ]
        elif self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        elif (offset_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [group['lr'] + (base_lr - self.eta_min) *
                    (1 - math.cos(math.pi / self.T_max)) / 2
                    for base_lr, group in
                    zip(self.base_lrs, self.optimizer.param_groups)]
        return [(1 + math.cos(math.pi * offset_epoch / self.T_max)) /
                (1 + math.cos(math.pi * (offset_epoch - 1) / self.T_max)) *
                (group['lr'] - self.eta_min) + self.eta_min
                for group in self.optimizer.param_groups]
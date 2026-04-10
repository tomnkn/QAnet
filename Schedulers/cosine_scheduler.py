import math

from torch.optim.lr_scheduler import LRScheduler


class CosineAnnealingLR(LRScheduler):
    """Cosine annealing learning rate scheduler.

    Decays the learning rate of each param group from the initial lr down to
    eta_min following a cosine curve over T_max steps:

        lr_t = eta_min + 0.5 * (lr_0 - eta_min) * (1 + cos(π * t / T_max))

    At t=0   → lr_0  (initial learning rate)
    At t=T_max → eta_min
    """

    def __init__(self, optimizer, T_max, eta_min=0.0, last_epoch=-1):
        if T_max <= 0:
            raise ValueError(f"T_max must be positive, got {T_max}")
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        t = self.last_epoch
        return [
            self.eta_min + 0.5 * (base_lr - self.eta_min) * (1 + math.cos(math.pi * t / self.T_max))
            for base_lr in self.base_lrs
        ]

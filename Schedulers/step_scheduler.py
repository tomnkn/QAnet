from torch.optim.lr_scheduler import LRScheduler


class StepLR(LRScheduler):
    """Step decay learning rate scheduler.

    Multiplies the learning rate of each param group by gamma every
    step_size steps:

        lr_t = lr_0 * gamma ^ floor(t / step_size)
    """

    def __init__(self, optimizer, step_size, gamma=0.5, last_epoch=-1):
        if step_size <= 0:
            raise ValueError(f"step_size must be positive, got {step_size}")
        if not 0.0 < gamma < 1.0:
            raise ValueError(f"gamma must be in (0, 1), got {gamma}")
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        t = self.last_epoch
        return [
            base_lr * (self.gamma ** (t // self.step_size))
            for base_lr in self.base_lrs
        ]

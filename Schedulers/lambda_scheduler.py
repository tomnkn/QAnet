from torch.optim.lr_scheduler import LRScheduler


class LambdaLR(LRScheduler):
    """Multiplies the learning rate of each param group by the output of a
    user-supplied function of the current step:

        lr_t = base_lr * lr_lambda(t)

    Args:
        optimizer:  wrapped optimizer
        lr_lambda:  a function that takes the current step (int) and returns
                    a multiplicative factor (float)
    """

    def __init__(self, optimizer, lr_lambda, last_epoch=-1):
        self.lr_lambda = lr_lambda
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        t = self.last_epoch
        factor = self.lr_lambda(t)
        return [base_lr * factor for base_lr in self.base_lrs]

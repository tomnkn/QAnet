import torch
from torch.optim import Optimizer


class SGD(Optimizer):
    """Vanilla Stochastic Gradient Descent (no momentum).

    Update rule:
        p = p - lr * grad
    """

    def __init__(self, params, lr, weight_decay=0.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # Weight decay: equivalent to L2 regularisation
                if wd != 0.0:
                    grad = grad.add(p, alpha=wd)

                p.add_(grad, alpha=-lr)

        return loss

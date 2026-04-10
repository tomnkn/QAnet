import torch
from torch.optim import Optimizer


class SGDMomentum(Optimizer):
    """Stochastic Gradient Descent with momentum.

    Update rule:
        v = momentum * v + grad
        p = p - lr * v
    """

    def __init__(self, params, lr, momentum=0.9, weight_decay=0.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # Weight decay
                if wd != 0.0:
                    grad = grad.add(p, alpha=wd)

                state = self.state[p]

                # Initialise velocity buffer on first step
                if "velocity" not in state:
                    state["velocity"] = torch.zeros_like(p)

                v = state["velocity"]

                # v = momentum * v + grad
                v.mul_(mu).add_(grad)

                p.add_(v, alpha=-lr)

        return loss

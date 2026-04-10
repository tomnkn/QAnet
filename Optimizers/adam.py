import math

import torch
from torch.optim import Optimizer


class Adam(Optimizer):
    """Adam optimiser (Kingma & Ba, 2015).

    Update rule:
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad^2
        m_hat = m / (1 - beta1^t)
        v_hat = v / (1 - beta2^t)
        p = p - lr * m_hat / (sqrt(v_hat) + eps)
    """

    def __init__(self, params, lr=1.0, betas=(0.9, 0.999), eps=1e-7, weight_decay=0.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # Weight decay
                if wd != 0.0:
                    grad = grad.add(p, alpha=wd)

                state = self.state[p]

                # Initialise moment buffers and step counter on first step
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)  # 1st moment (mean)
                    state["v"] = torch.zeros_like(p)  # 2nd moment (variance)

                m, v = state["m"], state["v"]
                state["step"] += 1
                t = state["step"]

                # Update biased moment estimates
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                v.mul_(beta2).add_(grad ** 2, alpha=1.0 - beta2)

                # Bias correction
                bias_correction1 = 1.0 - beta1 ** t
                bias_correction2 = 1.0 - beta2 ** t
                m_hat = m / bias_correction1
                v_hat = v / bias_correction2

                p.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)

        return loss

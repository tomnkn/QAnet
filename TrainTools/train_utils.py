"""
train_utils.py — Low-level training utilities used by train().
"""

import os

import numpy as np
import torch
from tqdm import tqdm


class ExponentialMovingAverage:
    """Track parameter EMA and support temporary shadow-weight evaluation."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in [0, 1)")
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name not in self.shadow:
                self.shadow[name] = param.detach().clone()
                continue
            self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_shadow(self, model: torch.nn.Module):
        self.backup = {}
        for name, param in model.named_parameters():
            if not param.requires_grad or name not in self.shadow:
                continue
            self.backup[name] = param.detach().clone()
            param.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name in self.backup:
                param.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        return {
            "decay": self.decay,
            "shadow": {k: v.clone() for k, v in self.shadow.items()},
        }


def _global_grad_norm(parameters) -> float:
    """Compute L2 norm over all available parameter gradients."""
    total_sq = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        grad = p.grad.detach()
        total_sq += float(torch.sum(grad * grad).item())
    return total_sq ** 0.5


def _named_grad_norm(named_parameters, include_name_fn) -> float:
    """Compute L2 norm over gradients of selected parameters by name."""
    total_sq = 0.0
    for name, p in named_parameters:
        if not include_name_fn(name):
            continue
        if p.grad is None:
            continue
        grad = p.grad.detach()
        total_sq += float(torch.sum(grad * grad).item())
    return total_sq ** 0.5


def train_single_epoch(model, optimizer, scheduler, data_iter,
                       steps, grad_clip, loss_fn, device,
                       global_step: int = 0,
                       warmup_steps: int = 0,
                       log_every_steps: int = 10,
                       track_cq_stats: bool = True,
                       track_conv_stats: bool = True,
                       accumulate_grad_steps: int = 1,
                       ema: ExponentialMovingAverage = None):
    """
    Run one block of `steps` training iterations consuming from `data_iter`.
    Returns
    -------
    mean_loss: float
        Mean loss over this block.
    step_metrics: list[dict]
        Step-level diagnostics sampled every `log_every_steps`.
    """
    model.train()
    loss_list = []
    step_metrics = []

    if accumulate_grad_steps < 1:
        raise ValueError("accumulate_grad_steps must be >= 1")

    base_lrs = [group.get("base_lr", group["lr"]) for group in optimizer.param_groups]

    for local_step in tqdm(range(steps), total=steps):
        step_id = global_step + local_step + 1

        # Optional linear warmup to avoid unstable early updates.
        if warmup_steps > 0 and step_id <= warmup_steps:
            warm = step_id / float(warmup_steps)
            for group, base_lr in zip(optimizer.param_groups, base_lrs):
                group["lr"] = base_lr * warm

        optimizer.zero_grad(set_to_none=True)

        micro_loss_sum = 0.0
        for _ in range(accumulate_grad_steps):
            Cwid, Ccid, Qwid, Qcid, y1, y2, _ = next(data_iter)
            Cwid, Ccid = Cwid.to(device), Ccid.to(device)
            Qwid, Qcid = Qwid.to(device), Qcid.to(device)
            y1, y2     = y1.to(device),   y2.to(device)

            p1, p2 = model(Cwid, Ccid, Qwid, Qcid)
            loss   = loss_fn(p1, p2, y1, y2)
            micro_loss_sum += float(loss.item())

            # Keep gradient scale equivalent to non-accumulated training.
            (loss / accumulate_grad_steps).backward()

        step_loss = micro_loss_sum / accumulate_grad_steps
        loss_list.append(step_loss)
        grad_norm_before = _global_grad_norm(model.parameters())
        conv_grad_norm = None
        if track_conv_stats:
            conv_grad_norm = _named_grad_norm(
                model.named_parameters(),
                lambda n: ("depthwise_conv" in n) or ("pointwise_conv" in n),
            )

        cq_weight_var = None
        cq_grad_norm = None
        if track_cq_stats and hasattr(model, "cq_att") and hasattr(model.cq_att, "w"):
            cq_weight_var = float(model.cq_att.w.detach().var(unbiased=False).item())
            if model.cq_att.w.grad is not None:
                cq_grad_norm = float(model.cq_att.w.grad.detach().norm(2).item())

        has_nan_loss = bool(np.isnan(step_loss))
        has_inf_loss = bool(np.isinf(step_loss))
        has_nonfinite_grad = False
        for p in model.parameters():
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad).all():
                has_nonfinite_grad = True
                break

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        grad_norm_after = _global_grad_norm(model.parameters())
        optimizer.step()
        if scheduler is not None and (warmup_steps == 0 or step_id > warmup_steps):
            scheduler.step()
        if ema is not None:
            ema.update(model)

        if log_every_steps > 0 and step_id % log_every_steps == 0:
            current_lr = optimizer.param_groups[0]["lr"]

            step_metrics.append({
                "step": step_id,
                "loss": float(step_loss),
                "lr": float(current_lr),
                "grad_norm_before_clip": grad_norm_before,
                "conv_grad_norm": conv_grad_norm,
                "grad_norm_after_clip": grad_norm_after,
                "cq_weight_var": cq_weight_var,
                "cq_grad_norm": cq_grad_norm,
                "nan_loss": has_nan_loss,
                "inf_loss": has_inf_loss,
                "nonfinite_grad": has_nonfinite_grad,
            })

    mean_loss = float(np.mean(loss_list))
    print(f"STEP {global_step + steps:8d}  loss {mean_loss:8f}\n")
    return mean_loss, step_metrics


def save_checkpoint(save_dir, ckpt_name, model, optimizer, scheduler,
                    step, best_f1, best_em, config, ema_state=None):
    """Save model, optimizer, scheduler state to a checkpoint file."""
    os.makedirs(save_dir, exist_ok=True)
    payload = {
        "model":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "step":            step,
        "best_f1":         best_f1,
        "best_em":         best_em,
        "config":          config,
        "ema_state":       ema_state,
    }
    torch.save(payload, os.path.join(save_dir, ckpt_name))

"""
train_utils.py — Low-level training utilities used by train().
"""

import os

import numpy as np
import torch
from tqdm import tqdm


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
                       log_every_steps: int = 10,
                       track_cq_stats: bool = True,
                       track_conv_stats: bool = True):
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

    for local_step in tqdm(range(steps), total=steps):
        optimizer.zero_grad(set_to_none=True)

        Cwid, Ccid, Qwid, Qcid, y1, y2, _ = next(data_iter)
        Cwid, Ccid = Cwid.to(device), Ccid.to(device)
        Qwid, Qcid = Qwid.to(device), Qcid.to(device)
        y1, y2     = y1.to(device),   y2.to(device)

        p1, p2 = model(Cwid, Ccid, Qwid, Qcid)
        loss   = loss_fn(p1, p2, y1, y2)
        loss_list.append(float(loss.item()))

        loss.backward()
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

        has_nan_loss = bool(torch.isnan(loss).item())
        has_inf_loss = bool(torch.isinf(loss).item())
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
        if scheduler is not None:
            scheduler.step()

        step_id = global_step + local_step + 1
        if log_every_steps > 0 and step_id % log_every_steps == 0:
            if scheduler is not None:
                current_lr = scheduler.get_last_lr()[0]
            else:
                current_lr = optimizer.param_groups[0]["lr"]

            step_metrics.append({
                "step": step_id,
                "loss": float(loss.item()),
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
                    step, best_f1, best_em, config):
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
    }
    torch.save(payload, os.path.join(save_dir, ckpt_name))

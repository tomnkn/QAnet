import math

from Schedulers.cosine_scheduler import CosineAnnealingLR
from Schedulers.lambda_scheduler import LambdaLR
from Schedulers.step_scheduler import StepLR


# ── Scheduler factories ──────────────────────────────────────────────────────

def cosine_scheduler(optimizer, args):
    """Cosine annealing over the full training run."""
    return CosineAnnealingLR(
        optimizer,
        T_max=args.num_steps,
    )


def step_scheduler(optimizer, args):
    """Step decay: multiply LR by gamma every lr_step_size steps."""
    return StepLR(
        optimizer,
        step_size=getattr(args, "lr_step_size", 10000),
        gamma=getattr(args, "lr_gamma", 0.5),
    )


def warmup_lambda(t):
    """Inverse exponential warmup from 0.0 to 1.0 in first 250 steps, then constant.
    
    From QANet paper (https://arxiv.org/pdf/1804.09541):
    - Warmup: inverse exponential increase from 0.0 to 0.001 in the first 250 steps
    - After 250 steps: maintain constant learning rate (og paper is 1000 steps)
    
    When paired with an optimizer with base_lr=0.001, this produces:
    - LR(0) = 0.001 * 0 ≈ 0.0
    - LR(250) ≈ 0.001 * 1.0 = 0.001
    - LR(>250) = 0.001 * 1.0 = 0.001
    """
    if t < 250:
        # Inverse exponential approach: 1 - exp(-t/tau)
        # tau ≈ 62.5 ensures ~98% saturation by t=250
        tau = 62.5
        return 1.0 - math.exp(-t / tau)
    else:
        # Constant learning rate for remainder of training
        return 1.0


def lambda_scheduler(optimizer, args):
    """LambdaLR with inverse exponential warmup (from QANet paper).
    
    Requires: optimizer must have base_lr=0.001 for correct behavior.
    """
    return LambdaLR(optimizer, lr_lambda=warmup_lambda)

# ── Registry ─────────────────────────────────────────────────────────────────

schedulers = {
    "cosine":  cosine_scheduler,
    "step":    step_scheduler,
    "lambda":  lambda_scheduler,
}

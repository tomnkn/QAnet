from Optimizers.adam import Adam
from Optimizers.sgd import SGD
from Optimizers.sgd_momentum import SGDMomentum


# ── Optimizer factories ──────────────────────────────────────────────────────
#
# NOTE: All optimizers use args.learning_rate as base_lr.
# When paired with a scheduler:
#   - `cosine` and `step` schedulers: work with any optimizer
#   - `lambda` scheduler: multiplies base_lr by warmup_lambda(t)
#     For QANet warmup, use with learning_rate=0.001 and warmup_lambda provides 0→1 factor

def adam(params, args):
    return Adam(
        params=params,
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=getattr(args, "eps", 1e-7),
        weight_decay=args.weight_decay,
    )


def sgd(params, args):
    return SGD(
        params=params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )


def sgd_momentum(params, args):
    return SGDMomentum(
        params=params,
        lr=args.learning_rate,
        momentum=getattr(args, "momentum", 0.9),
        weight_decay=args.weight_decay,
    )


# ── Registry ─────────────────────────────────────────────────────────────────

optimizers = {
    "adam":          adam,
    "sgd":           sgd,
    "sgd_momentum":  sgd_momentum,
}

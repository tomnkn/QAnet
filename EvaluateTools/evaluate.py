"""
evaluate.py — Evaluation entry point for QANet / Assignment 1.

Usage (from Assignment1.ipynb):
    from EvaluateTools.evaluate import evaluate
    metrics = evaluate()
    metrics = evaluate(save_dir="_model", ckpt_name="model.pt")

Returns
-------
dict with keys: f1, exact_match, loss
"""

import argparse
import os

import torch
import ujson as json

from Data import SQuADDataset, load_dev_eval, load_word_char_mats
from Losses import losses
from Models import QANet
from EvaluateTools.eval_utils import run_eval


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(
    # ── Data paths ────────────────────────────────────────────────────────────
    dev_npz:        str   = "_data/dev.npz",
    word_emb_json:  str   = "_data/word_emb.json",
    char_emb_json:  str   = "_data/char_emb.json",
    dev_eval_json:  str   = "_data/dev_eval.json",
    save_dir:       str   = "_model",
    log_dir:        str   = "_log",
    ckpt_name:      str   = "model.pt",

    # ── Eval settings ─────────────────────────────────────────────────────────
    batch_size:         int   = 8,
    test_num_batches:   int   = -1,       # -1 = full dev set
    max_answer_len:     int   = 30,
    loss_name:          str   = "qa_nll",

    # ── Model architecture (must match the checkpoint) ────────────────────────
    para_limit:     int   = 400,
    ques_limit:     int   = 50,
    char_limit:     int   = 16,
    d_model:        int   = 96,
    num_heads:      int   = 8,
    glove_dim:      int   = 300,
    char_dim:       int   = 64,
    dropout:        float = 0.1,
    dropout_char:   float = 0.05,
    pretrained_char: bool = False,
    freeze_word:    bool = True,
) -> dict:
    """Evaluate a saved QANet checkpoint on the SQuAD v1.1 dev set.

    Parameters
    ----------
    dev_npz:
        Path to the preprocessed dev record file.
    word_emb_json, char_emb_json:
        Paths to the embedding matrices produced by ``preprocess()``.
    dev_eval_json:
        Path to the dev evaluation metadata (contexts, gold answers).
    save_dir:
        Directory containing the checkpoint file.
    log_dir:
        Directory where ``answers.json`` will be written.
    ckpt_name:
        Filename of the checkpoint inside ``save_dir``.
    batch_size:
        Number of examples per batch.
    test_num_batches:
        Number of batches to evaluate (-1 = entire dev set).
    max_answer_len:
        Maximum decoded span length used by constrained span search.
    loss_name:
        Loss function key from the registry (default ``"qa_nll"``).
    para_limit, ques_limit, char_limit, d_model, num_heads,
    glove_dim, char_dim, dropout, dropout_char, pretrained_char:
        Model architecture parameters — must match the values used during
        training.
    freeze_word:
        Whether the pretrained word embedding table is frozen.

    Returns
    -------
    dict
        ``{"f1": float, "exact_match": float, "loss": float}``
    """
    os.makedirs(log_dir, exist_ok=True)

    if loss_name not in losses:
        raise ValueError(f"Unknown loss '{loss_name}'. Available: {list(losses.keys())}")

    ckpt_path = os.path.join(save_dir, ckpt_name)
    # PyTorch 2.6 defaults torch.load(..., weights_only=True), which rejects
    # scheduler lambda objects serialized in our training checkpoints.
    # These checkpoints are locally produced by this project, so loading with
    # weights_only=False is expected here.
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = ckpt.get("config", {})
    state = ckpt.get("model", ckpt.get("model_state"))
    if state is None:
        raise KeyError("Checkpoint missing both 'model' and 'model_state' keys")

    # Infer key architecture dims from checkpoint weights when available.
    inferred_d_model = int(state["conv.weight"].shape[0]) if "conv.weight" in state else int(cfg.get("d_model", d_model))
    inferred_glove_dim = int(state["word_emb.weight"].shape[1]) if "word_emb.weight" in state else int(cfg.get("glove_dim", glove_dim))
    inferred_char_dim = int(state["char_emb.weight"].shape[1]) if "char_emb.weight" in state else int(cfg.get("char_dim", char_dim))

    # Build a lightweight namespace so existing helpers can consume it.
    # Checkpoint config takes precedence over function defaults to avoid
    # shape mismatches when evaluating models trained with different settings.
    args = argparse.Namespace(
        dev_npz=dev_npz,
        word_emb_json=word_emb_json,
        char_emb_json=char_emb_json,
        dev_eval_json=dev_eval_json,
        para_limit=int(cfg.get("para_limit", para_limit)),
        ques_limit=int(cfg.get("ques_limit", ques_limit)),
        char_limit=int(cfg.get("char_limit", char_limit)),
        d_model=inferred_d_model,
        num_heads=int(cfg.get("num_heads", num_heads)),
        glove_dim=inferred_glove_dim,
        char_dim=inferred_char_dim,
        dropout=float(cfg.get("dropout", dropout)),
        dropout_char=float(cfg.get("dropout_char", dropout_char)),
        pretrained_char=bool(cfg.get("pretrained_char", pretrained_char)),
        freeze_word=bool(cfg.get("freeze_word", freeze_word)),
        norm_name=str(cfg.get("norm_name", "layer_norm")),
        norm_groups=int(cfg.get("norm_groups", 8)),
        activation=str(cfg.get("activation", "relu")),
        init_name=str(cfg.get("init_name", "kaiming")),
    )

    word_mat, char_mat = load_word_char_mats(args)
    model = QANet(word_mat, char_mat, args).to(DEVICE)
    model.load_state_dict(state)

    dev_eval = load_dev_eval(args)
    dev_dataset = SQuADDataset(dev_npz)

    metrics, ans = run_eval(
        model, dev_dataset, dev_eval,
        num_batches=test_num_batches,
        batch_size=batch_size,
        use_random_batches=False,
        device=DEVICE,
        loss_fn=losses[loss_name],
        max_answer_len=max_answer_len,
    )

    with open(os.path.join(log_dir, "answers.json"), "w") as f:
        json.dump(ans, f)

    print("TEST  loss {loss:.6f}  F1 {f1:.6f}  EM {exact_match:.6f}".format(**metrics))
    return {"f1": metrics["f1"], "exact_match": metrics["exact_match"], "loss": metrics["loss"]}

"""
eval_utils.py — Low-level evaluation utilities used by evaluate().
"""

import itertools
import re
import string
from collections import Counter
import numpy as np
import torch
from tqdm import tqdm

from Data import make_loader
from Losses import qa_nll_loss


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    return max(metric_fn(prediction, gt) for gt in ground_truths)


def squad_evaluate(eval_file, answer_dict):
    f1 = exact_match = total = 0.0
    for key, pred in answer_dict.items():
        total += 1.0
        ground_truths = eval_file[key]["answers"]
        exact_match += metric_max_over_ground_truths(exact_match_score, pred, ground_truths)
        f1 += metric_max_over_ground_truths(f1_score, pred, ground_truths)
    if total == 0.0:
        return {"exact_match": 0.0, "f1": 0.0}
    return {"exact_match": 100.0 * exact_match / total, "f1": 100.0 * f1 / total}


def convert_tokens(eval_file, qa_id, pp1, pp2):
    answer_dict = {}
    remapped_dict = {}
    for qid, p1, p2 in zip(qa_id, pp1, pp2):
        context = eval_file[str(qid)]["context"]
        spans = eval_file[str(qid)]["spans"]
        uuid = eval_file[str(qid)]["uuid"]
        length = len(spans)
        if p1 >= length or p2 >= length:
            ans = ""
        else:
            start_idx = spans[p1][0]
            end_idx = spans[p2][1]
            ans = context[start_idx:end_idx]
        answer_dict[str(qid)] = ans
        remapped_dict[uuid] = ans
    return answer_dict, remapped_dict


def decode_best_spans(p1: torch.Tensor, p2: torch.Tensor, max_answer_len: int = 30):
    """Jointly decode best valid span by maximizing p(start) + p(end).

    p1 and p2 are log-probabilities with shape [B, L].
    Constraints:
      - start <= end
      - (end - start + 1) <= max_answer_len
    """
    if p1.ndim != 2 or p2.ndim != 2:
        raise ValueError("p1 and p2 must have shape [B, L]")
    if p1.shape != p2.shape:
        raise ValueError("p1 and p2 must have the same shape")
    if max_answer_len < 1:
        raise ValueError("max_answer_len must be >= 1")

    batch_size, length = p1.shape

    scores = p1.unsqueeze(2) + p2.unsqueeze(1)  # [B, L, L]

    valid = torch.ones(length, length, dtype=torch.bool, device=p1.device)
    valid = torch.triu(valid, diagonal=0) & torch.tril(valid, diagonal=max_answer_len - 1)
    scores = scores.masked_fill(~valid.unsqueeze(0), -1e30)

    flat_idx = scores.view(batch_size, -1).argmax(dim=1)
    ystart = flat_idx // length
    yend = flat_idx % length
    return ystart, yend


@torch.no_grad()
def run_eval(model, dataset, eval_file, num_batches, batch_size,
             use_random_batches, device, loss_fn=qa_nll_loss, max_answer_len: int = 30):
    loader = make_loader(dataset, batch_size, shuffle=use_random_batches)
    # num_batches=-1 means evaluate the full dataset
    batch_limit = None if num_batches < 0 else num_batches
    total_display = len(loader) if num_batches < 0 else num_batches

    model.eval()
    answer_dict = {}
    losses = []

    for Cwid, Ccid, Qwid, Qcid, y1, y2, ids in tqdm(
        itertools.islice(loader, batch_limit), total=total_display
    ):
        Cwid, Ccid, Qwid, Qcid = (
            Cwid.to(device), Ccid.to(device), Qwid.to(device), Qcid.to(device)
        )
        y1, y2 = y1.to(device), y2.to(device)

        p1, p2 = model(Cwid, Ccid, Qwid, Qcid)
        loss = loss_fn(p1, p2, y1, y2)
        losses.append(float(loss.item()))

        ymin, ymax = decode_best_spans(p1, p2, max_answer_len=max_answer_len)

        answer_dict_, _ = convert_tokens(eval_file, ids.tolist(), ymin.tolist(), ymax.tolist())
        answer_dict.update(answer_dict_)

    metrics = squad_evaluate(eval_file, answer_dict)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics, answer_dict

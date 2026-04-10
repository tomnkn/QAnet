# QANet Model and System Architecture (Implementation Guide)

This document describes the complete implementation architecture of the project so a new user or AI agent can understand, run, and modify the system confidently.

## 1. What This Repository Implements

- Task: Extractive question answering on SQuAD v1.1.
- Model: QANet-style architecture with:
  - word + character embedding fusion,
  - encoder blocks (depthwise separable conv + self-attention + feed-forward),
  - context-question attention,
  - pointer head for start/end span prediction.
- Entrypoint workflow is notebook-driven via `assignment1.ipynb`:
  - Section 0: environment setup
  - Section 1: optional data download
  - Section 2: preprocessing
  - Section 3: training
  - Section 4: evaluation

## 2. High-Level Pipeline

1. Raw SQuAD JSON + embedding files are preprocessed into cached arrays and metadata.
2. Training loads cached tensors, constructs QANet from config, trains in blocks, evaluates periodically, and saves checkpoint + answers.
3. Evaluation reloads the model/checkpoint and computes dev metrics (F1, EM, loss).

Data flow:

`Raw JSON/GloVe -> preprocess() -> _data/*.npz + *.json -> train() -> _model/model.pt + _log/answers.json -> evaluate()`

## 3. Directory and Module Responsibilities

- `Tools/`
  - `preproc.py`: full SQuAD preprocessing pipeline.
  - `utils.py`: utility helpers (`set_seed`).
  - `download.py`: dataset download helper.
- `Data/`
  - `squad.py`: dataset class and cache sanity checks.
  - `loader.py`: DataLoader factory.
  - `io.py`: read embeddings/eval metadata from JSON.
- `Models/`
  - `qanet.py`: top-level QANet graph.
  - `embedding.py`: character + word embedding fusion and highway network.
  - `encoder.py`: positional encoding, multi-head attention, encoder block.
  - `attention.py`: context-question attention (trilinear style).
  - `heads.py`: pointer output head for start/end logits.
  - `conv.py`: custom Conv1d/Conv2d and depthwise separable conv.
  - `dropout.py`: custom inverted dropout.
  - `Initializations/`: Kaiming/Xavier initialization registry.
  - `Normalizations/`: LayerNorm/GroupNorm registry.
  - `Activations/`: activation registry.
- `Optimizers/`
  - `adam.py`, `sgd.py`, `sgd_momentum.py`: custom optimizer implementations.
  - `optimizer.py`: optimizer registry/factories.
- `Schedulers/`
  - `cosine_scheduler.py`, `step_scheduler.py`, `lambda_scheduler.py`: custom schedulers.
  - `scheduler.py`: scheduler registry/factories.
- `Losses/`
  - `loss.py`: span loss functions (`qa_nll`, `qa_ce`) + registry.
- `TrainTools/`
  - `train.py`: public training API.
  - `train_utils.py`: step loop and checkpoint writer.
- `EvaluateTools/`
  - `evaluate.py`: public evaluation API.
  - `eval_utils.py`: SQuAD metric logic and batch evaluation.

## 4. Preprocessing Architecture

Implemented in `Tools/preproc.py`.

### 4.1 Inputs

- `train_file`, `dev_file`: SQuAD JSON.
- `glove_word_file` (or optional fastText).
- optional `glove_char_file` if `pretrained_char=True`.

### 4.2 Core processing stages

1. Parse SQuAD articles/paragraphs/QAs.
2. Tokenize context/question via regex tokenizer.
3. Build character-level token forms.
4. Map answer character spans to token spans (`y1`,`y2`).
5. Build vocab counters (word and char).
6. Build embedding matrices and token-index maps.
7. Vectorize into fixed-size arrays with limits:
   - context length: `para_limit`
   - question length: `ques_limit`
   - per-token char length: `char_limit`
8. Save all outputs under `_data` (by default).

### 4.3 Output artifacts

- `train.npz`, `dev.npz` containing:
  - `context_idxs`, `context_char_idxs`, `ques_idxs`, `ques_char_idxs`, `y1s`, `y2s`, `ids`
- `word_emb.json`, `char_emb.json`
- `train_eval.json`, `dev_eval.json`
- `word2idx.json`, `char2idx.json`
- `dev_meta.json`

## 5. Dataset and DataLoader Contracts

Implemented in `Data/squad.py` and `Data/loader.py`.

### 5.1 Sample tuple shape contract

Each dataset item returns:

`(Cwid, Ccid, Qwid, Qcid, y1, y2, id)`

with tensors:
- `Cwid`: `[para_limit]`
- `Ccid`: `[para_limit, char_limit]`
- `Qwid`: `[ques_limit]`
- `Qcid`: `[ques_limit, char_limit]`
- `y1`, `y2`: scalar span indices

### 5.2 Training batch contract

After collation (`batch_size = B`):
- `Cwid`: `[B, Lc]`
- `Ccid`: `[B, Lc, char_limit]`
- `Qwid`: `[B, Lq]`
- `Qcid`: `[B, Lq, char_limit]`

### 5.3 Cache sanity checks

`sanity_check_cache()` ensures:
- required files exist,
- required NPZ keys exist,
- non-empty train set,
- no invalid span where `y1 > y2`.

## 6. QANet Model Architecture (Implementation-Level)

Top-level in `Models/qanet.py`.

### 6.1 Embedding stage

- Word embedding: `nn.Embedding.from_pretrained(word_mat, freeze=False)`.
- Char embedding: `nn.Embedding.from_pretrained(char_mat, freeze=pretrained_char)`.
- Character branch in `Embedding`:
  - 2D depthwise separable conv over char dimension,
  - activation,
  - max over char axis.
- Word branch:
  - dropout,
  - channel transpose.
- Fusion:
  - concatenate char+word channels,
  - 2-layer Highway network.

Output of embedding module: `[B, d_word + d_char, L]`.

### 6.2 Input projection

- `context_conv` and `question_conv`: depthwise separable 1D conv
- project fused channels to model dim `d_model`.

Outputs:
- `C`: `[B, d_model, Lc]`
- `Q`: `[B, d_model, Lq]`

### 6.3 Embedding encoders

- Separate `EncoderBlock` for context and question:
  - positional encoding (sinusoidal buffer),
  - repeated depthwise separable conv stack,
  - multi-head self-attention,
  - feed-forward linear layer,
  - residual + dropout + normalization structure.

Outputs:
- `Ce`: `[B, d_model, Lc]`
- `Qe`: `[B, d_model, Lq]`

### 6.4 Context-question attention

`CQAttention` (`Models/attention.py`):
- Builds similarity tensor using concatenated trilinear-style features `[C, Q, C*Q]`.
- Computes attention both context->question and question-aware context.
- Returns concatenated tensor `[C, A, C*A, C*B]`.

Output: `[B, 4*d_model, Lc]`.

### 6.5 Model encoder stack

- `cq_resizer` projects `[B, 4*d_model, Lc] -> [B, d_model, Lc]` as `M1` seed.
- A 7-block `ModuleList` is reused in three passes to produce `M1`, `M2`, `M3`:
  - pass 1: `M1`
  - pass 2: `M2`
  - pass 3: `M3`

### 6.6 Output pointer head

`Pointer` in `Models/heads.py`:
- Concatenate `M1/M2` and `M1/M3` to get two `[B, 2C, L]` tensors.
- Project with learned vectors to start/end logits.
- Apply masking on PAD locations.
- Return log-probabilities with `log_softmax`:
  - `p1`: start log-probs `[B, L]`
  - `p2`: end log-probs `[B, L]`

## 7. Masking Semantics

Mask convention is important and consistent:
- `cmask = (Cwid == 0)` and `qmask = (Qwid == 0)`.
- `True` means PAD / invalid position.
- `mask_logits(...)` fills masked positions with `-1e30` before softmax.

## 8. Training System Architecture

Public API: `TrainTools/train.py::train(...)`.

### 8.1 Config and reproducibility

- All function args are packed into an `argparse.Namespace` called `args`.
- Seed is set via `set_seed(seed)` (Python, NumPy, torch, CUDA).
- Full config is persisted to `_model/run_config.json`.

### 8.2 Dynamic component selection (registry-based)

String keys select components at runtime:
- Optimizers: `adam`, `sgd`, `sgd_momentum`
- Schedulers: `cosine`, `step`, `lambda`, or `none`
- Losses: `qa_nll`, `qa_ce`
- Normalizations: `layer_norm`, `group_norm` (used by model construction)

Invalid keys raise explicit `ValueError` with available options.

### 8.3 Main training loop

Loop is block-based by `checkpoint` steps:
1. Run `train_single_epoch(...)` for the next block size.
2. Evaluate sampled train batches and ordered dev batches.
3. Track learning rate (`scheduler.get_last_lr()` if scheduler exists, else optimizer LR).
4. Append metrics to `history`.
5. Early-stop check (patience on both dev F1 and dev EM degrading).
6. Save checkpoint every block.
7. Save dev answers to `_log/answers.json`.

### 8.4 Per-step operations (`train_utils.py`)

For each batch:
- zero grads,
- forward pass,
- compute span loss,
- backward pass,
- gradient clipping (`clip_grad_norm_`, default 5.0),
- optimizer step,
- scheduler step (if present).

### 8.5 Checkpoint payload

Saved via `torch.save` with keys:
- `model`
- `optimizer_state`
- `scheduler_state` (or `None`)
- `step`
- `best_f1`, `best_em`
- `config`

## 9. Evaluation System Architecture

Public API: `EvaluateTools/evaluate.py::evaluate(...)`.

1. Load embeddings and reconstruct QANet with architecture args.
2. Load checkpoint and apply `model.load_state_dict(...)`.
3. Run `run_eval(...)` over dev set (or limited batches).
4. Save answer dict to `_log/answers.json`.
5. Return metrics dict:
   - `f1`
   - `exact_match`
   - `loss`

### 9.1 Metric computation details

`EvaluateTools/eval_utils.py`:
- text normalization removes punctuation/articles/case differences,
- computes exact match and token-level F1,
- takes max score over all gold answers per question,
- converts predicted start/end token indices back to text span via saved character spans.

## 10. Optimizer, Scheduler, and Loss Implementations

### 10.1 Optimizers

- `Adam`: custom Adam with bias correction and optional L2-style weight decay.
- `SGD`: vanilla SGD.
- `SGDMomentum`: SGD with velocity buffer.

All optimizer factories in `Optimizers/optimizer.py` use `args.learning_rate` as base LR.

### 10.2 Schedulers

- `cosine`: cosine annealing to `eta_min` over `num_steps`.
- `step`: multiplicative decay by `gamma` every `lr_step_size` steps.
- `lambda`: warmup scheduler based on QANet-style inverse exponential ramp for first 1000 steps, then constant factor 1.0.
- `none`: special value handled in `train.py` to disable scheduler entirely.

Effective LR formula with scheduler:

`effective_lr_t = base_lr * scheduler_factor_t`

If `scheduler_name="none"`, LR is fixed at optimizer param group LR.

### 10.3 Losses

- `qa_nll`: expects log-probabilities (matches current pointer output because pointer uses `log_softmax`).
- `qa_ce`: expects raw logits (not ideal unless pointer output is changed accordingly).

## 11. Shape and Interface Summary (Quick Reference)

- Model input tensors:
  - `Cwid`: `[B, Lc]`
  - `Ccid`: `[B, Lc, char_limit]`
  - `Qwid`: `[B, Lq]`
  - `Qcid`: `[B, Lq, char_limit]`
- Model output:
  - `p1`, `p2`: `[B, Lc]` log-probabilities over start/end indices.
- Gold labels:
  - `y1`, `y2`: `[B]`.

## 12. Configuration Parameters That Must Stay Consistent

Training/evaluation architecture parameters must match checkpoint config:
- `para_limit`, `ques_limit`, `char_limit`
- `d_model`, `num_heads`
- `glove_dim`, `char_dim`
- `dropout`, `dropout_char`
- `pretrained_char`

If these differ between train and evaluate, checkpoint loading or behavior may break.

## 13. Runtime Outputs and Files

Generated by training:
- `_model/model.pt` (checkpoint)
- `_model/run_config.json`
- `_log/answers.json` (dev predictions from latest eval)

Generated by evaluation:
- `_log/answers.json` (overwritten with current run predictions)

Generated by preprocessing:
- all cached `_data/*.npz` and `_data/*.json` listed earlier.

## 14. Extension Points for New Users/Agents

1. Swap optimizer/scheduler/loss by key (no train-loop rewrite needed).
2. Add new registry entry in:
   - `Optimizers/optimizer.py`,
   - `Schedulers/scheduler.py`,
   - `Losses/loss.py`,
   - `Models/Normalizations/normalization.py`,
   - `Models/Activations/activation_function.py`.
3. Adjust model width/depth through `train()` args.
4. Modify encoder internals in `Models/encoder.py`.
5. Change answer head behavior in `Models/heads.py`.

## 15. Known Practical Notes

- Scheduler can be disabled by setting `scheduler_name="none"`.
- Pointer currently returns log-probs; `qa_nll` is the natural loss pairing.
- Early stopping is conservative: patience increments only when both dev F1 and dev EM degrade.
- Checkpoint serialization includes scheduler state; scheduler functions used by LambdaLR must be picklable module-level functions.

## 16. Minimal Reproduction (Programmatic)

```python
from TrainTools.train import train
from EvaluateTools.evaluate import evaluate

results = train(
    num_steps=6000,
    batch_size=8,
    optimizer_name="adam",
    scheduler_name="lambda",
    loss_name="qa_nll",
    learning_rate=1e-3,
)

metrics = evaluate(
    ckpt_name="model.pt",
    loss_name="qa_nll",
)
```

This uses the full architecture described above and produces best-dev training stats plus final dev metrics.

# QANet Model and System Architecture (Current Implementation)

This document describes the architecture implemented in this repository as it exists now, including training/evaluation control flow, module responsibilities, registries, tensor contracts, and checkpoint behavior.

## 1. Repository Scope

- Task: extractive question answering on SQuAD v1.1.
- Main workflow: notebook-driven via assignment1.ipynb (setup, preprocess, train, evaluate).
- Core model: QANet-style architecture with word+char embedding fusion, encoder blocks, context-question attention, and span pointer head.

## 2. End-to-End Pipeline

1. Preprocessing parses SQuAD and embedding files into cached arrays and metadata.
2. Training loads caches, builds QANet from args/config, trains in checkpoint blocks, evaluates periodically, and writes checkpoint/log artifacts.
3. Evaluation restores a saved model checkpoint, infers architecture settings when needed, evaluates on dev, and writes predictions.

Data flow:

Raw SQuAD + embeddings -> preprocess() -> _data/*.npz + _data/*.json -> train() -> _model/model.pt + _log/*.json -> evaluate()

## 3. Module Responsibilities

### Data

- Data/squad.py
  - SQuADDataset loads NPZ arrays and returns:
    - (Cwid, Ccid, Qwid, Qcid, y1, y2, id)
  - sanity_check_cache validates required cache files, NPZ keys, non-empty train set, and y1 <= y2.
- Data/loader.py
  - make_loader builds DataLoader (num_workers=0).
- Data/io.py
  - load_word_char_mats, load_train_dev_eval, load_dev_eval.

### Models

- Models/qanet.py
  - Top-level QANet graph and forward pass.
- Models/embedding.py
  - Word+char fusion with depthwise separable char conv + Highway network.
- Models/encoder.py
  - Positional encoding, multi-head self-attention, encoder block, mask_logits.
- Models/attention.py
  - Context-question attention (trilinear features [C, Q, C*Q]).
- Models/heads.py
  - Pointer head for start/end logits.
- Models/conv.py
  - Custom Conv1d/Conv2d and depthwise separable conv implementations.
- Models/dropout.py
  - Custom inverted dropout layer.
- Models/Initializations/*
  - Initialization registry and implementations.
- Models/Normalizations/*
  - Layer norm and group norm registry/factory.
- Models/Activations/*
  - ReLU and LeakyReLU registry/factory.

### Training/Evaluation/Loss

- TrainTools/train.py
  - Public training API and orchestration.
- TrainTools/train_utils.py
  - Block step loop, EMA, gradient metrics, checkpoint save.
- EvaluateTools/evaluate.py
  - Public evaluation API, checkpoint loading, config inference.
- EvaluateTools/eval_utils.py
  - Batched evaluation, constrained span decoding, SQuAD EM/F1 metrics.
- Losses/loss.py
  - qa_nll and qa_ce span losses.

### Optimizers/Schedulers/Utilities

- Optimizers/optimizer.py
  - Optimizer registry/factories.
- Schedulers/scheduler.py
  - Scheduler registry/factories and warmup lambda.
- Tools/preproc.py
  - Full preprocessing pipeline.
- Tools/utils.py
  - set_seed helper.

## 4. Preprocessing Architecture

Implemented in Tools/preproc.py.

### Inputs

- train_file, dev_file (SQuAD JSON)
- glove_word_file (or fastText if fasttext=True)
- optional glove_char_file if pretrained_char=True

### Processing Steps

1. Parse SQuAD into examples and eval metadata.
2. Tokenize text via regex tokenizer.
3. Build token character sequences.
4. Convert answer char spans to token spans.
5. Build word/char frequency counters.
6. Build embeddings + token2idx maps.
7. Vectorize with limits:
   - para_limit (default 400)
   - ques_limit (default 50)
   - ans_limit (default 30)
   - char_limit (default 16)
8. Save NPZ and JSON outputs.

### Outputs

- train.npz, dev.npz with keys:
  - context_idxs
  - context_char_idxs
  - ques_idxs
  - ques_char_idxs
  - y1s
  - y2s
  - ids
- word_emb.json, char_emb.json
- train_eval.json, dev_eval.json
- word2idx.json, char2idx.json
- dev_meta.json

## 5. Dataset and Tensor Contracts

Single sample from dataset:

- Cwid: [Lc]
- Ccid: [Lc, char_limit]
- Qwid: [Lq]
- Qcid: [Lq, char_limit]
- y1, y2: scalar indices
- id: scalar example id

Batch shapes:

- Cwid: [B, Lc]
- Ccid: [B, Lc, char_limit]
- Qwid: [B, Lq]
- Qcid: [B, Lq, char_limit]
- y1, y2: [B]

Mask semantics used throughout model:

- cmask = (Cwid == 0)
- qmask = (Qwid == 0)
- True means PAD/invalid position.

mask_logits fills masked positions with -1e30 before softmax/argmax steps.

## 6. QANet Implementation Details

Top-level in Models/qanet.py.

### Embedding Stage

- Word embedding: nn.Embedding.from_pretrained(word_mat, freeze=freeze_word).
- Char embedding: nn.Embedding.from_pretrained(char_mat, freeze=pretrained_char).
- Embedding module:
  - char branch: depthwise separable conv + activation + max over char axis
  - word branch: dropout + transpose
  - concat + 2-layer Highway

### Input Projection and Embedding Encoder

- Shared 1x1 projection conv for context and question.
- Shared embedding encoder block for both streams:
  - Ce = emb_enc(C, cmask)
  - Qe = emb_enc(Q, qmask)

### Context-Question Attention

- CQAttention builds similarity on concatenated [C, Q, C*Q].
- Computes attended representations A and B.
- Output shape: [B, 4*d_model, Lc].

### Model Encoder Stacks

- cq_resizer projects 4*d_model -> d_model.
- Three separate model encoder stacks (not shared weights):
  - model_enc_blks_1: produces M1
  - model_enc_blks_2: produces M2 (from M1)
  - model_enc_blks_3: produces M3 (from M2)
- Each stack contains 7 EncoderBlock modules.

### Pointer Head

- Pointer concatenates [M1, M2] and [M1, M3].
- Projects with learned vectors w1/w2.
- Applies mask_logits on PAD positions.
- Returns raw masked logits (not log-softmax):
  - p1: [B, Lc]
  - p2: [B, Lc]

## 7. Encoder Block Design

Implemented in Models/encoder.py.

- Positional encoding: sinusoidal buffer (non-trainable).
- Conv stack:
  - DepthwiseSeparableConv repeated conv_num times.
  - Stochastic-depth-style conv dropout with increasing probability by depth.
- Self-attention:
  - Multi-head scaled dot-product attention.
- Feed-forward:
  - Linear(d_model -> 4*d_model) -> activation -> Linear(4*d_model -> d_model).
- Residual + normalization structure around each stage.

Normalization and activation are selected via registries (args.norm_name, args.activation).

## 8. Registries and Selectable Components

### Optimizers

From Optimizers/optimizer.py:

- adam
- sgd
- sgd_momentum

### Schedulers

From Schedulers/scheduler.py:

- cosine
- step
- lambda
- none (handled in train.py by disabling scheduler)

lambda uses warmup_lambda(t): inverse-exponential rise for t < 1000, then factor 1.0.

### Losses

From Losses/loss.py:

- qa_nll
  - Applies log_softmax internally, then NLL loss for start/end.
- qa_ce
  - Uses cross_entropy directly on raw logits.

### Normalizations

From Models/Normalizations/normalization.py:

- layer_norm
- group_norm

### Activations

From Models/Activations/activation_function.py:

- relu
- leaky_relu

### Initializations

From Models/Initializations/initialization.py:

- kaiming
- kaiming_normal
- kaiming_uniform
- xavier
- xavier_normal
- xavier_uniform

## 9. Training System Architecture

Public API: TrainTools/train.py::train(...)

### Setup

- set_seed(seed) for Python, NumPy, torch (and CUDA where available).
- Full args persisted to _model/run_config.json.
- Cache validated via sanity_check_cache.
- Model/data/eval metadata loaded.

### Parameter Grouping

Parameters are split into:

- decay_params (weight decay applied)
- no_decay_params (weight decay = 0) for:
  - bias params
  - normalization params (name contains "norm")
  - 1D tensors

### Main Loop (Checkpoint Blocks)

For each block of checkpoint steps:

1. train_single_epoch runs the step loop.
2. If EMA enabled, shadow weights are applied for evaluation.
3. run_eval on sampled train batches and ordered dev batches.
4. EMA weights restored back to live model weights.
5. History record appended with train/dev metrics + LR.
6. Checkpoint saved only on improvement:
   - better dev F1, or
   - equal dev F1 and better dev EM.
7. Early stopping based on consecutive non-improving blocks.
8. _log/answers.json updated each eval block.

Optional step diagnostics are saved to:

- _log/step_metrics_<optimizer>_<scheduler>_seed<seed>.json
  - or custom step_metrics_file.

### Per-Step Behavior (train_utils.py)

- Optional linear warmup (for Adam path in train.py).
- Gradient accumulation via accumulate_grad_steps.
- Backprop on scaled micro-batch loss.
- Gradient clipping with clip_grad_norm_.
- optimizer.step().
- scheduler.step() (outside warmup window).
- EMA update (if enabled).
- Optional metrics sampled every log_every_steps:
  - loss, lr
  - grad norm before/after clip
  - conv grad norm
  - CQ weight variance + grad norm
  - NaN/Inf/non-finite gradient flags

## 10. Checkpoint Format

Saved payload keys (train_utils.save_checkpoint):

- model
- optimizer_state
- scheduler_state
- step
- best_f1
- best_em
- config
- ema_state

Notes:

- scheduler_state is None when scheduler is disabled.
- ema_state is None when EMA is disabled.

## 11. Evaluation System Architecture

Public API: EvaluateTools/evaluate.py::evaluate(...)

Flow:

1. Validate loss_name registry key.
2. Load checkpoint with torch.load(..., weights_only=False).
3. Read model state from:
   - ckpt["model"], or fallback ckpt["model_state"].
4. Infer key dims from state dict when available:
   - d_model from conv.weight
   - glove_dim from word_emb.weight
   - char_dim from char_emb.weight
5. Build args namespace where checkpoint config takes precedence.
6. Rebuild QANet and load state_dict.
7. Run run_eval over dev batches.
8. Save predictions to _log/answers.json.
9. Return dict: {f1, exact_match, loss}.

## 12. Evaluation and Decoding Semantics

Implemented in EvaluateTools/eval_utils.py.

- Model outputs raw start/end logits.
- Loss computed via selected loss function.
- decode_best_spans chooses span maximizing start+end score with constraints:
  - start <= end
  - span length <= max_answer_len
- Predicted token spans converted back to answer text using stored context spans.
- SQuAD metrics:
  - normalization removes punctuation/articles/case differences
  - EM and token-level F1
  - max score over all gold answers per example

## 13. Runtime Artifacts

Preprocessing:

- _data/train.npz
- _data/dev.npz
- _data/word_emb.json
- _data/char_emb.json
- _data/train_eval.json
- _data/dev_eval.json
- _data/word2idx.json
- _data/char2idx.json
- _data/dev_meta.json

Training:

- _model/model.pt
- _model/run_config.json
- _log/answers.json
- _log/step_metrics_*.json (optional)

Evaluation:

- _log/answers.json (current eval predictions)

## 14. Practical Notes and Caveats

- evaluate() prioritizes checkpoint config over function defaults to reduce shape mismatch risks.
- evaluate() uses weights_only=False because locally saved checkpoints may include scheduler lambda state.
- Scheduler key "none" is supported at train API level (not part of schedulers registry map).
- Early stopping counts non-improving checkpoint blocks, not individual steps.
- Word embedding freezing is configurable via freeze_word (default True).
- Character embedding freezing is controlled by pretrained_char.

## 15. Minimal Programmatic Usage

```python
from TrainTools.train import train
from EvaluateTools.evaluate import evaluate

train_results = train(
    num_steps=6000,
    batch_size=8,
    optimizer_name="adam",
    scheduler_name="lambda",
    loss_name="qa_nll",
    learning_rate=1e-3,
)

eval_metrics = evaluate(
    ckpt_name="model.pt",
    loss_name="qa_nll",
)
```

This runs the same architecture and pipeline described above.

# QANet Deep Learning Mechanisms — Complete Error Identification

## Executive Summary

I've identified **17 critical errors** in your QANet implementation across optimization, initialization, scheduling, normalization, architectural components, and evaluation compatibility. Two of these are **CRITICAL SEVERITY** (will prevent training from working at all), eleven are **HIGH SEVERITY** (will cause training instability, evaluation failures, or poor generalization), two are MEDIUM, and two are LOW priority.

---

## 12 April 2026 Update — Newly Applied Fixes

This section records three additional fixes that were implemented after the earlier audit.

### 0) Training Run Log (10k-step cosine, qa_nll)

Recorded here for reproducibility and direct comparison against the stronger 3k-run recipe.

Run configuration:

```python
# -- training loop -----------------------------------------------------------
num_steps  = 10000
checkpoint = 500
batch_size = 8
accumulate_grad_steps = 4
seed = 42
val_num_batches = 0  # train-monitoring disabled (prints 0 metrics)
test_num_batches = 150
early_stop = 10

# -- optimizer / scheduler ---------------------------------------------------
optimizer_name = "adam"
scheduler_name = "cosine"
loss_name = "qa_nll"

learning_rate = 1e-3
beta1 = 0.9
beta2 = 0.999
eps = 1e-8
weight_decay = 3e-5
grad_clip = 1.0
warmup_steps = 1000
dropout = 0.15
d_model = 128
```

Observed logs (excerpt through step 3500):

```text
STEP      500  loss 20.547813
TEST        loss 27.719023  F1 6.363199  EM 0.250000
Learning rate: [0.0005, 0.0005]

STEP     1000  loss 7.917776
TEST        loss 9.229487  F1 8.414490  EM 2.416667
Learning rate: [0.001, 0.001]

STEP     1500  loss 7.490897
TEST        loss 7.583587  F1 12.842754  EM 7.833333
Learning rate: [0.0009938441702975688, 0.0009938441702975688]

STEP     2000  loss 8.809056
TEST        loss 7.173117  F1 17.916788  EM 12.500000
Learning rate: [0.0009755282581475768, 0.0009755282581475768]

STEP     2500  loss 17.629601
TEST        loss 7.423115  F1 16.312539  EM 9.166667
Learning rate: [0.0009455032620941839, 0.0009455032620941839]

STEP     3000  loss 30.962723
TEST        loss 8.109672  F1 10.433804  EM 4.166667
Learning rate: [0.0009045084971874737, 0.0009045084971874737]

STEP     3500  loss 31.202969
TEST        loss 12.451104  F1 6.500154  EM 0.750000
```

Quick interpretation:

- This run improves until about step 2000, then degrades notably by 3000-3500.
- With `val_num_batches=0`, the printed `VALID(train)` line is expected to stay at zeros and is not informative.
- This behavior supports using best-checkpoint selection and cautious extension beyond the early-gain window.

### 1) Best-checkpoint model selection corrected (Applied)

- **File**: `TrainTools/train.py`
- **What changed**:
  - Best metrics now initialize to `-1.0` so the first evaluation is always treated as a valid baseline.
  - Improvement rule is now unified and deterministic:
    - improve if `dev_f1` is higher, or
    - if `dev_f1` is equal and `dev_em` is higher.
  - Checkpoint save now occurs **only on improvement**.
  - Patience now increments only on non-improving checkpoints and early-stop triggers on `patience >= early_stop`.
- **Why**:
  - Prevents overwriting better checkpoints with worse later ones.
  - Aligns saved checkpoint selection with dev-metric quality.

### 2) Weight-decay parameter grouping added (Applied)

- **File**: `TrainTools/train.py`
- **What changed**:
  - Parameters are split into two optimizer groups before optimizer creation:
    - **decay group**: regular multi-dimensional weights, uses configured `weight_decay`
    - **no-decay group**: bias terms, norm-related params, and 1D tensors, uses `weight_decay=0.0`
  - Optimizer is instantiated with grouped parameter dictionaries.
- **Why**:
  - Avoids over-regularizing normalization and bias parameters, which commonly harms QA optimization stability and EM.

### 3) Max answer length is now wired through train/eval APIs (Applied)

- **Files**: `TrainTools/train.py`, `EvaluateTools/evaluate.py`
- **What changed**:
  - Added `max_answer_len` argument to `train()` and `evaluate()` (default `30`).
  - Forwarded `max_answer_len` into `run_eval(...)` calls.
- **Why**:
  - Makes decode constraints configurable at API level and consistent with constrained span decoding logic.
  - Reduces risk of train/eval mismatch from hardcoded decoding limits.

### Validation status

- Static diagnostics report no errors introduced in:
  - `TrainTools/train.py`
  - `EvaluateTools/evaluate.py`

### 4) Added word-embedding freeze control and enabled safe default (Applied)

- **Files**: `Models/qanet.py`, `TrainTools/train.py`, `EvaluateTools/evaluate.py`
- **What changed**:
  - Added a new boolean config flag: `freeze_word`.
  - `QANet` now reads `freeze_word` from args and applies it to:
    - `nn.Embedding.from_pretrained(..., freeze=freeze_word)` for word embeddings.
  - Exposed `freeze_word` in both public APIs:
    - `train(..., freeze_word: bool = True)`
    - `evaluate(..., freeze_word: bool = True)`
- **Why**:
  - Freezing pretrained GloVe is typically more stable in small-data regimes and helps prevent early overfitting/drift in lexical representations.
  - This also aligns runtime behavior with standard QANet practice of freezing pretrained word vectors by default.

### 5) Fixed preprocessing supervision consistency for multi-answer spans (Applied)

- **File**: `Tools/preproc.py`
- **What changed**:
  - Added canonical span selection per example (`choose_answer_span`):
    - choose shortest span,
    - tie-break by earliest start index.
  - Applied the same chosen span consistently for:
    - answer-length filtering,
    - saved training targets (`y1s`, `y2s`).
  - Updated answer-length filtering to inclusive span length:
    - from non-inclusive `(y2 - y1)` style behavior
    - to inclusive `(y2 - y1 + 1)`.
  - Skips malformed examples that have no valid answer span annotations.
- **Why**:
  - Removes label-noise caused by using different spans for filtering vs supervision.
  - Aligns preprocessing constraints with extractive QA span semantics (inclusive token boundaries).
  - Improves consistency for EM/F1 learning signals near max-answer-length boundary cases.

### Action required

- Regenerate preprocessed data to apply this fix to training:
  - rerun preprocessing so `_data/train.npz` and `_data/dev.npz` are rebuilt.

### 6) Aligned QANet architecture to intended topology (Applied)

- **File**: `Models/qanet.py`
- **What changed**:
  - Replaced separate context/question projection convolutions with a **shared 1x1 projection conv**:
    - from two `DepthwiseSeparableConv(..., k=5)` layers
    - to one shared `Conv1d(..., kernel_size=1)`.
  - Replaced separate context/question embedding encoders with a **shared embedding encoder**:
    - from `c_emb_enc` + `q_emb_enc`
    - to one shared `emb_enc` using `length=max(len_c, len_q)`.
  - Replaced `cq_resizer` with a **1x1 conv projection**:
    - from `DepthwiseSeparableConv(..., k=5)`
    - to `Conv1d(..., kernel_size=1)`.
  - Added explicit initialization for new 1x1 conv layers via `initializations[init_name]` and zero bias via `constant_`.
- **Why**:
  - Matches intended architectural assumptions more closely.
  - Reduces early-stage feature blurring from wider-kernel projection layers.
  - Improves architectural consistency so reused hyperparameter recipes are more likely to behave similarly.

### Action required (architecture change)

- Retrain from scratch after this model change.
- Do not reuse checkpoints produced by the previous architecture.

### 7) Warmup + scheduler behavior update (Applied)

- **Files**: `TrainTools/train.py`, `TrainTools/train_utils.py`
- **What changed**:
  - Added `warmup_steps` to `train(...)` public arguments (default `1000`).
  - `train_single_epoch(...)` now receives warmup for all Adam runs:
    - `warmup_steps=warmup_steps if optimizer_name == "adam" else 0`
  - Added stable per-group base LR capture in `train.py`:
    - `group.setdefault("base_lr", group["lr"])`
  - Added linear warmup in `train_utils.py` before each optimizer step:
    - scales LR by `step / warmup_steps` for early steps.
  - Added scheduler-step guard in `train_utils.py` so scheduler does not overwrite warmup:
    - scheduler steps only when `warmup_steps == 0` or after warmup completes.
  - Updated LR printout in `train.py` to read directly from optimizer param groups so logs reflect effective warmup LR.
- **Why**:
  - Aligns behavior with the intended training setup where Adam warmup is used even with cosine scheduling.
  - Prevents warmup/scheduler race conditions that can erase warmup effects in early training.

### 8) Syntax regression from mixed indentation fixed (Applied)

- **File**: `TrainTools/train.py`
- **Issue**:
  - Mixed indentation introduced around optimizer/scheduler setup caused parse errors (`Unexpected indentation` / `Unindent amount does not match previous indent`).
- **Fix**:
  - Rewrote the affected block with consistent indentation.
- **Validation**:
  - Static diagnostics now report no errors in `TrainTools/train.py`.

### Behavioral note

- Current behavior now supports warmup with **every scheduler** for **Adam**.
- Warmup is still intentionally disabled for non-Adam optimizers (`sgd`, `sgd_momentum`) unless explicitly changed.

### 9) QANet model-encoder stack sharing removed (Applied)

- **File**: `Models/qanet.py`
- **What changed**:
  - Replaced a single shared 7-block model-encoder stack with three independent stacks:
    - `model_enc_blks_1` for the `M1` pass
    - `model_enc_blks_2` for the `M2` pass
    - `model_enc_blks_3` for the `M3` pass
  - Updated forward pass so each stage uses its own parameter set instead of reusing one shared stack.
- **Why**:
  - Restores stage-specific representational capacity during iterative refinement.
  - Better matches intended QANet-style multi-pass model encoder behavior.
- **Validation**:
  - Static diagnostics report no errors in `Models/qanet.py` after this change.

---

## Encoder Patch Log (Applied)

This section documents the exact `Models/encoder.py` fixes that were implemented during debugging.

### 0) EncoderBlock math/order parity fix (Applied, high impact)

- **Location**: `Models/encoder.py` (`EncoderBlock.forward`)
- **What changed**:
  - Updated convolution sublayer sequence to match intended parity ordering:
    - `conv -> act -> (stochastic-depth drop on even convs) -> residual add -> norm`
  - Updated attention sublayer sequence to:
    - `out = self.self_att(out, mask) + res`
    - removed the extra post-attention dropout that was previously applied before entering FFN.
  - Updated FFN tail ordering to:
    - `norm -> ffn1 -> act -> ffn2 -> dropout -> residual add`
- **Why**:
  - The previous ordering changed residual and normalization placement inside the block, which changes optimization dynamics and representation flow.
  - This was identified as a high-impact implementation mismatch for `qa_nll + layer_norm` runs and can directly affect EM/F1.
- **Validation**:
  - Static diagnostics report no errors after the patch in `Models/encoder.py`.
- **Observed training impact**:
  - Recent post-fix runs show major improvement in both F1 and EM.
  - Improvement is especially visible in the 3000-step run, where metric jumps are noticeably larger than before this fix.
  - Compact 3000-step run snapshot (TEST metrics at each checkpoint):
    - 300: loss 33.14, F1 6.21, EM 0.16
    - 600: loss 12.14, F1 8.69, EM 1.88
    - 900: loss 8.42, F1 11.60, EM 6.09
    - 1200: loss 7.54, F1 14.65, EM 9.84
    - 1500: loss 6.89, F1 20.01, EM 14.06
    - 1800: loss 6.32, F1 25.66, EM 19.53
    - 2100: loss 5.93, F1 30.94, EM 23.12
    - 2400: loss 5.70, F1 31.70, EM 24.84
    - 2700: loss 5.55, F1 36.42, EM 29.38
    - 3000: loss 5.46, F1 39.87, EM 33.12
  - End-of-run summary:
    - Best F1: 39.8673
    - Best EM: 33.1250
    - Net gain from step 300 to step 3000: F1 +33.66, EM +32.97
  - Run configuration used for this result:
    - num_steps=3000, checkpoint=300, val_num_batches=0, test_num_batches=80
    - accumulate_grad_steps=4, batch_size=8, seed=42, early_stop=8
    - optimizer=adam, scheduler=cosine, loss=qa_nll
    - learning_rate=1.5e-3, beta1=0.9, beta2=0.999, eps=1e-8
    - weight_decay=3e-7, grad_clip=1.0, warmup_steps=250
    - dropout=0.15, d_model=128

### 1) Multi-head ordering normalized and made consistent

- **Location**: `Models/encoder.py`, lines 66-88
- **What changed**:
  - `q/k/v` are now packed in a clear batch-major convention:
    - `[B, L, H, d_k] -> [B, H, L, d_k] -> [B*H, L, d_k]`
  - `attn_mask` expansion now follows the same batch-major head packing before reshape to `[B*H, L, L]`.
  - Attention output is unpacked with the matching inverse order:
    - `[B*H, L, d_k] -> [B, H, L, d_k] -> [B, L, H, d_k] -> [B, L, d_model]`
- **Why**:
  - Prevents accidental batch/head mixing from mismatched pack/unpack conventions.
  - Removes ordering ambiguity for future edits.

### 2) Residual connections fixed to use per-sublayer inputs

- **Location**: `Models/encoder.py`, lines 116-129
- **What changed**:
  - Inside each convolution sublayer, `res = out` is set immediately before the sublayer computation.
  - Before self-attention, `res = out` is set again so attention uses the current tensor as skip input.
- **Why**:
  - Ensures each residual addition uses the immediate sublayer input (standard residual behavior), not a stale tensor from earlier in the block.

### 3) Notes on current state

- There is still a redundant `res = out` before `self.normb(out)` at line 113. It is harmless because `res` is overwritten in the conv loop before use.
- No static analysis errors were reported after these encoder edits.

### Expected impact

- More stable and semantically correct attention computation.
- Cleaner residual flow through conv and attention sublayers.
- These fixes are intended to improve representation quality and help recover low F1/EM behavior seen previously.

---

## Evaluation Patch Log (Applied)

This section documents the `EvaluateTools/eval_utils.py` fixes that were implemented during debugging.

### 1) Joint constrained span decoding added

- **Location**: `EvaluateTools/eval_utils.py` (`decode_best_spans` + `run_eval` decode path)
- **What changed**:
  - Replaced independent start/end argmax + `min/max` reordering with joint decoding.
  - Added `decode_best_spans(p1, p2, max_answer_len=30)` that maximizes:
    - `score(i, j) = p1[i] + p2[j]`
  - Enforced decoding constraints:
    - `start <= end`
    - `(end - start + 1) <= max_answer_len`
- **Why**:
  - Independent argmax can pick inconsistent boundaries and then clamp to a suboptimal span.
  - Joint constrained decoding is standard for extractive QA and improves span validity, which typically helps EM/F1.

### 2) Empty-evaluation safeguard added

- **Location**: `EvaluateTools/eval_utils.py` (`squad_evaluate`)
- **What changed**:
  - Added guard for `total == 0` to return `{exact_match: 0.0, f1: 0.0}`.
- **Why**:
  - Prevents `ZeroDivisionError` when evaluation is invoked with zero sampled batches.

### 3) Checkpoint-driven architecture loading in evaluate entrypoint (Applied)

- **Location**: `EvaluateTools/evaluate.py` (`evaluate`)
- **What changed**:
  - `evaluate()` now loads checkpoint metadata/state before model construction.
  - Added support for both checkpoint schemas:
    - `ckpt["model"]`
    - `ckpt["model_state"]`
  - Added architecture inference from checkpoint weights to prevent shape drift:
    - `d_model` inferred from `state["conv.weight"].shape[0]` when available
    - `glove_dim` inferred from `state["word_emb.weight"].shape[1]`
    - `char_dim` inferred from `state["char_emb.weight"].shape[1]`
  - `args` now prioritizes checkpoint `config` values (with function defaults as fallback) before building `QANet`.
  - Then loads state dict into a shape-compatible model instance.
- **Why**:
  - Fixes runtime `size mismatch` errors when a checkpoint was trained with different architecture settings (for example `d_model=128`) than evaluate defaults (`d_model=96`).
  - Makes notebook and script evaluation robust across experiments without manual architecture argument syncing.

### Expected impact

- Better decoding quality at evaluation time via globally better valid spans.
- More robust metric computation for edge cases (no eval samples).

### Observed results after eval update

- In recent runs, the reported training/evaluation loss starts much lower (around ~10) instead of the previously observed extremely large starting value (~20000).
- EM and F1 are now trending higher than before, consistent with improved span decoding quality.
- Interpretation note: this change is expected because joint constrained decoding produces more valid/high-quality answer spans than independent start/end argmax with post-hoc reordering.

---

## Training Patch Log (Applied)

This section documents the EMA-related training updates applied to stabilize checkpoint-time evaluation.

### 1) EMA (Exponential Moving Average) added to training loop

- **Files**: `TrainTools/train_utils.py`, `TrainTools/train.py`
- **What changed**:
  - Added `ExponentialMovingAverage` utility class (default decay `0.999`) in `train_utils.py`.
  - EMA shadow weights are updated after every optimizer step.
  - `train_single_epoch(...)` now accepts an optional `ema` object and updates it each step.
  - `train(...)` exposes new args:
    - `use_ema: bool = True`
    - `ema_decay: float = 0.999`
  - At each checkpoint boundary:
    - Apply EMA shadow weights.
    - Run train-monitor and dev evaluation using EMA weights.
    - Save checkpoint with EMA-applied model weights.
    - Restore live (non-EMA) weights before continuing optimization.
  - Checkpoints now include optional `ema_state` metadata.

### 2) Why this helps

- QANet training can be noisy step-to-step; EMA smooths short-term parameter variance.
- Evaluating and saving with EMA weights often yields more stable and higher EM/F1 than raw weights at the same step.
- This does not replace optimizer updates; it only changes which weights are used for evaluation/checkpointing.

### 3) Adam denominator status

- The Adam denominator in `Optimizers/adam.py` is currently in standard form (`sqrt(v_hat) + eps`), so no additional denominator patch was required in this pass.

---

## Embedding Patch Log (Applied)

This section documents the character-convolution axis bug fix in the embedding layer.

### 1) Character convolution changed to token-local 1D Char-CNN

- **File**: `Models/embedding.py`
- **Bug**:
  - Char features were previously computed with 2D convolution over `[L, char_len]`, which can mix signal across neighboring token positions.
- **What changed**:
  - Replaced 2D conv path with token-local 1D convolution over character axis only.
  - New flow:
    - Input char tensor: `[B, L, char_len, d_char]`
    - Reshape to per-token batches: `[B*L, d_char, char_len]`
    - Apply depthwise-separable 1D conv over `char_len`
    - Max-pool over character axis
    - Reshape back to `[B, d_char, L]`
- **Why**:
  - Char-CNN should model subword patterns within each token, not fuse neighboring token positions at this stage.

### Expected impact

- Cleaner token-local character representations.
- Reduced contextual leakage in embedding stage.
- Potential improvements in span boundary quality, often visible in EM/F1.

---

## Additional Architecture Patches (Applied)

### 1) Encoder FFN upgraded from 1 layer to 2-layer feed-forward network

- **File**: `Models/encoder.py`
- **What changed**:
  - Replaced single projection `d_model -> d_model` with Transformer-style FFN:
    - `ffn1: d_model -> 4*d_model`
    - activation
    - `ffn2: 4*d_model -> d_model`
  - Residual and dropout structure around the FFN sublayer is preserved.
- **Why**:
  - A wider two-layer FFN increases non-linear modeling capacity and is generally stronger than a single linear projection in encoder blocks.

### 2) Highway gate bias initialized to -1.0

- **File**: `Models/embedding.py`
- **What changed**:
  - Set each highway gate layer bias to `-1.0` at initialization.
- **Why**:
  - Encourages early carry behavior (`1 - gate`) so token representations pass through more easily at the start of training, which can stabilize optimization.

### Expected impact

- Better representational power in encoder FFN sublayers.
- More stable early-stage embedding flow through highway layers.
- Potential EM/F1 gains after full retraining from scratch.

---

## Loss/Output Contract Patch (Applied)

### Pointer outputs changed to raw logits, and NLL now handles normalization internally

- **Files**: `Models/heads.py`, `Losses/loss.py`
- **Bug**:
  - Pointer head previously returned `log_softmax` outputs, while training allowed switching between `qa_nll` and `qa_ce` losses.
  - `qa_ce` expects raw logits, so using `qa_ce` with already-normalized log-probabilities created a contract mismatch.
- **What changed**:
  - `Models/heads.py`: removed `log_softmax` from pointer output; now returns raw masked logits.
  - `Losses/loss.py`: `qa_nll_loss` now applies `F.log_softmax` internally before `F.nll_loss`.
  - `qa_ce_loss` remains unchanged and now receives the correct logits input.
- **Why**:
  - Establishes a single clean interface: model returns logits; each loss function owns its required normalization.
  - Makes loss swapping (`qa_nll` vs `qa_ce`) safe and mathematically consistent.

### Expected impact

- Eliminates loss/output mismatch risk when experimenting with cross-entropy.
- Improves training robustness and reduces chances of unstable optimization caused by incompatible loss contracts.

---

## The 17 Errors (Priority Order)

### 🚨 CRITICAL ERRORS

#### **Error 3: StepLR Scheduler — Learning Rate Becomes Zero**
- **File**: `Schedulers/step_scheduler.py`, line 26
- **Problem**: Uses multiplication instead of exponentiation for decay
- **Current**: `base_lr * gamma * (t // step_size)` = 0 at step 0
- **Fix**: `base_lr * (gamma ** (t // step_size))`
- **Impact**: Training completely breaks; optimizer makes no progress

#### **Error 7: EncoderBlock — Broken Residual Connection + Duplicate Conv**
- **File**: `Models/encoder.py`, lines 99 & 111
- **Problems**:
  1. Convolution applied twice (line 99): `out = conv(out); out = conv(out)` should be once
  2. Self-attention output discarded (line 111): `out = res` should be `out = out + res`
- **Impact**: All self-attention gradients are dead; model can't use attention mechanism
- **Result**: 8 attention heads are wasted; model architecture is broken

---

### 🔴 HIGH SEVERITY ERRORS

#### **Error 1: Xavier Uniform Initialization**
- **File**: `Models/Initializations/xavier.py`, line 33
- **Problem**: `sqrt(2.0 / (fan_in * fan_out))` should be `sqrt(2.0 / (fan_in + fan_out))`
- **Impact**: Weights initialized 7× too small → vanishing gradients, slow convergence

#### **Error 2: Kaiming Uniform Initialization**
- **File**: `Models/Initializations/kaiming.py`, line 31
- **Problem**: `sqrt(1.0 / fan)` should be `sqrt(2.0 / fan)` (missing factor of 2)
- **Impact**: Dead ReLU problem; many neurons stuck at 0

#### **Error 5: Adam Optimizer — Weight Decay Sign**
- **File**: `Optimizers/adam.py`, line 54
- **Problem**: `grad = grad.add(p, alpha=-wd)` should be `alpha=wd` (positive)
- **Impact**: L2 regularization works backwards; encourages weights to grow instead of shrink

#### **Error 6: Multi-Head Attention — Missing Scaling Factor**
- **File**: `Models/encoder.py`, line 62
- **Problem**: `torch.bmm(q, k.transpose(1, 2))` should be `* self.scale`
- **Missing**: `1/sqrt(d_k)` scaling required by Attention is All You Need
- **Impact**: Attention logits too large → softmax becomes one-hot → vanishing gradients

#### **Error 9: Lambda Scheduler — Addition Instead of Multiplication**
- **File**: `Schedulers/lambda_scheduler.py`, line 22
- **Problem**: `[base_lr + factor for ...]` should be `[base_lr * factor for ...]`
- **Current Formula**: `lr_t = base_lr + lr_lambda(t)` (wrong: addition)
- **Correct Formula**: `lr_t = base_lr * lr_lambda(t)` (correct: multiplication)
- **Impact**: Learning rate scaling is completely wrong; if lr_lambda returns 0.5, LR becomes base_lr + 0.5 instead of base_lr * 0.5

#### **Error 10: Adam Optimizer — Bias Correction Uses Multiplication Instead of Exponentiation**
- **File**: `Optimizers/adam.py`, lines 70–71
- **Problem**: Uses `*` (multiplication) instead of `**` (exponentiation)
- **Current**: `bias_correction1 = 1.0 - beta1 * t` (wrong: linear)
- **Correct**: `bias_correction1 = 1.0 - beta1 ** t` (correct: exponential)
- **Impact**: At step 2+, bias correction becomes negative (e.g., `1 - 0.9*2 = -0.8`), flipping parameter update signs and breaking training

#### **Error 11: Adam Optimizer — Second Moment Missing Gradient Squaring**
- **File**: `Optimizers/adam.py`, line 73
- **Problem**: Second moment accumulator doesn't square the gradient
- **Current**: `v.mul_(beta2).add_(grad, alpha=1.0 - beta2)` (missing `**2`)
- **Correct**: `v.mul_(beta2).add_(grad ** 2, alpha=1.0 - beta2)`
- **Impact**: Variance tracking broken; adaptive per-parameter learning rates won't work (all parameters treated equally)

#### **Error 12: SGDMomentum Optimizer — Velocity Update Uses Subtraction Instead of Addition**
- **File**: `Optimizers/sgd_momentum.py`, line 49
- **Problem**: Momentum accumulator subtracts gradient instead of adding it
- **Current**: `v.mul_(mu).sub_(grad)` (wrong: subtraction)
- **Correct**: `v.mul_(mu).add_(grad)` (correct: addition)
- **Impact**: Optimizer does gradient ascent instead of descent; parameters move in opposite direction

#### **Error 13: GroupNorm — Incorrect Reshape Dimension Order**
- **File**: `Models/Normalizations/groupnorm.py`, line 34
- **Problem**: Reshape dimensions are swapped; normalizing over wrong axes
- **Current**: `x.view(B, C // self.G, self.G, *spatial)` (wrong: [B, C//G, G, ...])
- **Correct**: `x.view(B, self.G, C // self.G, *spatial)` (correct: [B, G, C//G, ...])
- **Impact**: Each group doesn't normalize independently over channels; statistics computed wrongly

#### **Error 14: Pointer Network — Incorrect Matmul Dimension Order**
- **File**: `Models/heads.py`, lines 25-26
- **Problem**: Weight vector and feature matrix have incompatible dimensions for matmul
- **Current**: `torch.matmul(self.w1, X1)` where w1: [2C], X1: [B, 2C, L]
- **Correct**: `torch.einsum('c,bcl->bl', self.w1, X1)` or `torch.matmul(X1.transpose(1, 2), self.w1)`
- **Impact**: Shape mismatch error; forward pass crashes, no predictions computed

#### **Error 15: Lambda Scheduler — Unpicklable Lambda Function**
- **File**: `Schedulers/scheduler.py`, line 24
- **Problem**: Anonymous lambda function can't be pickled during checkpoint saving
- **Current**: `LambdaLR(optimizer, lr_lambda=lambda _: 1.0)`
- **Correct**: Use a named function instead:
  ```python
  def constant_lambda(t):
      return 1.0
  LambdaLR(optimizer, lr_lambda=constant_lambda)
  ```
- **Impact**: AttributeError when training tries to save checkpoints; forces use of "cosine" or "step" scheduler

#### **Error 16: Lambda Scheduler — Missing Warmup Implementation**
- **Files**: `Optimizers/optimizer.py` (line 14) AND `Schedulers/scheduler.py` (lines 25-31)
- **Root Problem**: Misaligned base learning rate and scheduler function
  1. Adam was hardcoded to `lr=1.0` (intended to be controlled by scheduler)
  2. Lambda scheduler only returned 1.0 (no actual warmup)
  3. Result: Final LR = 1.0 × 1.0 = 1.0 ❌ (way too high)
  
- **Current (Wrong)**:
  ```python
  # Optimizers/optimizer.py
  Adam(params, lr=1.0, ...)  # Hardcoded, ignores args.learning_rate
  
  # Schedulers/scheduler.py
  def constant_lambda(t): return 1.0  # No warmup
  ```

- **Correct**:
  ```python
  # Optimizers/optimizer.py
  Adam(params, lr=args.learning_rate, ...)  # Use actual learning rate
  
  # Schedulers/scheduler.py
  def warmup_lambda(t):  # Implements QANet paper warmup
      if t < 1000:
          tau = 250.0
          return 1.0 - math.exp(-t / tau)
      else:
          return 1.0
  ```

- **Impact**: 
  - With original code: Final LR = 1.0 (catastrophic; gradient updates way too large)
  - With fix: Final LR = learning_rate × warmup_factor = 0.001 × (0→1) ✅
  - Without fix, Adam + lambda scheduler combo produces immediate training divergence
  - Violates QANet paper specification

  #### **Error 17: Evaluation Checkpoint Loading Fails on PyTorch 2.6**
  - **File**: `EvaluateTools/evaluate.py`, line 118
  - **Problem**: `torch.load` now defaults to `weights_only=True` in PyTorch 2.6
  - **Failure**: Checkpoints trained with lambda scheduler include `warmup_lambda` in serialized scheduler state, causing:
    - `UnpicklingError: Weights only load failed`
    - `Unsupported global: Schedulers.scheduler.warmup_lambda`
  - **Fix**: Load trusted local checkpoints with `weights_only=False`
  - **Impact**: Experiment evaluation crashes even when training succeeded

---

### 🟡 MEDIUM SEVERITY ERRORS

#### **Error 4: Cosine Annealing Scheduler**
- **File**: `Schedulers/cosine_scheduler.py`, line 25
- **Problem**: Missing `0.5 *` factor in formula
- **Current**: `eta_min + (base_lr - eta_min) * (1 + cos(...))` ranges learning rate [eta_min, 2*base_lr]
- **Correct**: `eta_min + 0.5 * (base_lr - eta_min) * (1 + cos(...))` ranges [eta_min, base_lr]
- **Impact**: Wrong learning rate schedule; too high at start

---

### 🟢 LOW SEVERITY ERRORS

#### **Error 8: Early Stopping Logic**
- **File**: `TrainTools/train.py`, line 196
- **Current**: Uses `and` operator — stops only when BOTH metrics degrade
- **Alternative**: Use `or` operator — stops when EITHER metric stops improving
- **Impact**: May train longer than necessary, but not a logical bug per se
- **Recommendation**: Low priority; current logic is conservative but acceptable

---

## Error Details with Formulas

### Error 1: Xavier Uniform Initialization
**Theory**: Initialize weights uniformly in `[-bound, bound]` where bound depends on layer fan-in/fan-out
$$\text{std} = \text{gain} \cdot \sqrt{\frac{2}{\text{fan\_in} + \text{fan\_out}}}$$
$$\text{bound} = \sqrt{3} \cdot \text{std}$$

**Current Wrong Formula** (multiplies instead of adds):
$$\text{std} = \text{gain} \cdot \sqrt{\frac{2}{\text{fan\_in} \times \text{fan\_out}}}$$

**Example Impact** (d_model=96, num_heads=8, glove_dim=300):
- fan_in=300, fan_out=96
- Wrong: `std = sqrt(2/28800) ≈ 0.0083` (weights ≈ ±0.014)
- Correct: `std = sqrt(2/396) ≈ 0.071` (weights ≈ ±0.123)
- Ratio: **7-8× smaller**

---

### Error 2: Kaiming Uniform Initialization  
**Theory**: He initialization optimized for ReLU networks
$$\text{std} = \sqrt{\frac{2}{\text{fan}}}$$

**Current Wrong Formula** (missing factor 2):
$$\text{std} = \sqrt{\frac{1}{\text{fan}}}$$

**Impact**: Weights are 1.41× smaller; ReLU neurons receive smaller activations → many stay at 0 (dead ReLU)

---

### Error 3: StepLR Scheduler — CRITICAL

**Theory**: Decay learning rate exponentially at regular intervals
$$\text{lr}_t = \text{base\_lr} \cdot \gamma^{\lfloor t / \text{step\_size} \rfloor}$$

**Current Wrong Formula**:
$$\text{lr}_t = \text{base\_lr} \cdot \gamma \cdot \lfloor t / \text{step\_size} \rfloor$$

**Timeline** (base_lr=1e-3, gamma=0.5, step_size=10000):
| Step | Correct | Wrong |
|------|---------|-------|
| 0 | 1e-3 | 0 ❌ |
| 5000 | 1e-3 | 0 ❌ |
| 10000 | 0.5e-3 | 0.5e-3 ✓ (but only by accident) |
| 20000 | 0.25e-3 | 1e-3 ❌ |

**The bug starts immediately** — learning rate is zero at step 0!

---

### Error 4: Cosine Annealing Scheduler

**Theory**: Smooth learning rate decay using cosine function
$$\text{lr}_t = \eta_{\min} + \frac{1}{2}(\text{base\_lr} - \eta_{\min})\left(1 + \cos\left(\frac{\pi t}{T_{\max}}\right)\right)$$

**Current Wrong Formula** (missing 0.5):
$$\text{lr}_t = \eta_{\min} + (\text{base\_lr} - \eta_{\min})\left(1 + \cos\left(\frac{\pi t}{T_{\max}}\right)\right)$$

**Impact** (with eta_min=0, base_lr=1e-3, T_max=60000):
- At t=0: Wrong gives 2×1e-3, Correct gives 1e-3
- At t=30000 (halfway): Wrong gives 1e-3, Correct gives 0.5e-3
- At t=60000: Both give eta_min=0

The wrong version starts too high and doesn't decay properly in the middle.

---

### Error 5: Adam Weight Decay — L2 Regularization

**Theory**: L2 regularization adds penalty term to loss
$$\mathcal{L}_{\text{regularized}} = \mathcal{L} + \lambda \|w\|^2_2$$

**Gradient with respect to w**:
$$\frac{\partial \mathcal{L}_{\text{reg}}}{\partial w} = \frac{\partial \mathcal{L}}{\partial w} + 2\lambda w$$

We should **ADD** the weight term to gradient (with positive coefficient).

**Current Wrong Code**:
```python
grad = grad.add(p, alpha=-wd)  # Negative: subtracts the weight!
```
This does the OPPOSITE of regularization — encourages weights to grow!

**Correct Code**:
```python
grad = grad.add(p, alpha=wd)  # Positive: adds the weight penalty
```

---

### Error 6: Multi-Head Attention — Scaled Dot-Product Attention

**Theory** (from "Attention is All You Need"):
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

The scaling factor `1/sqrt(d_k)` is **essential** for stable gradients.

**Why it matters**:
- Dot product between random vectors has variance ~ $d_k$
- Without scaling: attention logits have large variance
- Large logits → softmax becomes nearly one-hot
- One-hot softmax → gradient through softmax ≈ 0
- Result: attention weights don't learn; attention mechanism is dead

**Current Wrong Code** (missing scaling):
```python
attn = torch.bmm(q, k.transpose(1, 2))  # No 1/sqrt(d_k)
```

**Correct Code**:
```python
attn = torch.bmm(q, k.transpose(1, 2)) * self.scale  # Apply 1/sqrt(d_k)
```
where `self.scale = 1.0 / math.sqrt(self.d_k)` is already computed in `__init__`.

---

### Error 7: EncoderBlock — Broken Residual Connections (CRITICAL)

The forward pass has two critical bugs:

**Bug 1: Duplicate Convolution** (line 99)
```python
out = conv(out)
out = self.norms[i](out)
out = conv(out)  # ❌ Applied TWICE! Should apply once then activate
out = self.act(out)
```

Should be:
```python
out = conv(out)
out = self.norms[i](out)
out = self.act(out)  # Apply activation, not another conv
```

**Bug 2: Overwriting Self-Attention Output** (line 111)
```python
out = self.self_att(out, mask)  # Compute self-attention
out = res  # ❌ OVERWRITE WITH OLD RESIDUAL!
```

This **completely discards** the self-attention computation!

Should be:
```python
out = self.self_att(out, mask)  # Compute self-attention
out = out + res  # Add to residual, don't replace
```

**Consequences**:
- All 8 attention heads' gradients are dead
- Model cannot learn long-range dependencies
- Architecture is broken; residual connection is bypassed
- Massive waste of computation

---

### Error 8: Early Stopping Logic

**Current Logic**:
```python
if dev_f1 < best_f1 and dev_em < best_em:  # Both must degrade
    patience += 1
```

This uses AND: patience only increments when **BOTH** metrics degrade.

**Scenario**:
- Checkpoint 1: F1 improves +0.1, EM improves +0.05 → patience=0 ✓
- Checkpoint 2: F1 stays same, EM stays same → patience=0 (still) ✗
- Checkpoint 3: F1 stays same, EM stays same → patience=0 (still) ✗
- Never stops, even though metrics plateaued!

**More typical approach**:
```python
if dev_f1 >= best_f1 or dev_em >= best_em:  # Either improves
    patience = 0
else:
    patience += 1
```

This increments patience only when **BOTH** fail to improve.

**Recommendation**: Low priority; current logic is defensible (conservative). Only change if desired.

---

### Error 9: Lambda Scheduler — Addition Instead of Multiplication

**Location**: `Schedulers/lambda_scheduler.py`, line 22

**Theory**: Lambda scheduler multiplies the base learning rate by a custom function
$$\text{lr}_t = \text{base\_lr} \cdot f(t)$$

where $f(t)$ is a user-supplied function that returns a multiplicative factor.

**Current Wrong Code**:
```python
def get_lr(self):
    t = self.last_epoch
    factor = self.lr_lambda(t)
    return [base_lr + factor for base_lr in self.base_lrs]  # WRONG: addition
```

**Correct Code**:
```python
def get_lr(self):
    t = self.last_epoch
    factor = self.lr_lambda(t)
    return [base_lr * factor for base_lr in self.base_lrs]  # CORRECT: multiplication
```

**Impact** (example with base_lr = 0.001 and lambda function returning 0.5):
- **Wrong**: `0.001 + 0.5 = 0.501` (learning rate increases by 500×!)
- **Correct**: `0.001 * 0.5 = 0.0005` (proper learning rate decay)

This completely breaks any custom learning rate scheduling that uses the lambda scheduler.

---

### Error 10: Adam Optimizer — Bias Correction Uses Multiplication Instead of Exponentiation

**Location**: `Optimizers/adam.py`, lines 70–71

**Theory**: Adam uses bias correction to unbias the moment estimates in early training steps
$$m\_\text{hat} = \frac{m}{1 - \beta_1^t}$$
$$v\_\text{hat} = \frac{v}{1 - \beta_2^t}$$

These should use **exponential decay** $(1 - \beta^t)$, not **linear** $(1 - \beta \cdot t)$.

**Current Wrong Code**:
```python
# Bias correction
bias_correction1 = 1.0 - beta1 * t        # WRONG: multiplication
bias_correction2 = 1.0 - beta2 * t        # WRONG: multiplication
m_hat = m / bias_correction1
v_hat = v / bias_correction2
```

**Correct Code**:
```python
# Bias correction
bias_correction1 = 1.0 - beta1 ** t       # CORRECT: exponentiation
bias_correction2 = 1.0 - beta2 ** t       # CORRECT: exponentiation
m_hat = m / bias_correction1
v_hat = v / bias_correction2
```

**Example** (with default beta1 = 0.9):

| Step | Correct (β^t) | Wrong (β*t) | Bias Corr | Effect |
|------|---------------|------------|-----------|--------|
| 1 | 1 - 0.9^1 = 0.1 | 1 - 0.9*1 = 0.1 | ~1.0 | OK |
| 2 | 1 - 0.9^2 = 0.19 | 1 - 0.9*2 = -0.8 | **negative** | Signs flip ⚠️ |
| 10 | 1 - 0.9^10 ≈ 0.65 | 1 - 0.9*10 = -8.0 | **very negative** | Catastrophic |

**Impact**:
- At step 2+, dividing by a negative number flips parameter update signs
- Gradient descent becomes gradient ascent
- Loss increases instead of decreases
- Training becomes unstable or completely fails

---

### Error 11: Adam Optimizer — Second Moment Missing Gradient Squaring

**Location**: `Optimizers/adam.py`, line 73

**Theory**: Adam optimizer maintains two moment estimates:
- **First moment (m)**: exponential moving average of gradients
- **Second moment (v)**: exponential moving average of **squared gradients**

$$m = \beta_1 m + (1 - \beta_1) \cdot \text{grad}$$
$$v = \beta_2 v + (1 - \beta_2) \cdot \text{grad}^2$$

The second moment squared is crucial for adaptive learning rates.

**Current Wrong Code**:
```python
# Update biased moment estimates
m.mul_(beta1).add_(grad, alpha=1.0 - beta1)      # ✓ Correct
v.mul_(beta2).add_(grad, alpha=1.0 - beta2)      # ❌ WRONG: missing ** 2
```

**Correct Code**:
```python
# Update biased moment estimates
m.mul_(beta1).add_(grad, alpha=1.0 - beta1)           # ✓ Correct
v.mul_(beta2).add_(grad ** 2, alpha=1.0 - beta2)      # ✅ Square the gradient
```

**Impact** (example with gradient = 0.1):
- **Wrong**: `v = 0.999 * v + 0.001 * 0.1 = ... + 0.0001`
- **Correct**: `v = 0.999 * v + 0.001 * (0.1^2) = ... + 0.00001`

The wrong implementation:
- Accumulates raw gradients instead of variance
- Breaks the adaptive learning rate mechanism
- `1 / sqrt(v)` becomes meaningless
- Optimizer loses its primary advantage of adapting to per-parameter gradient magnitudes

---

### Error 12: SGDMomentum Optimizer — Velocity Update Uses Subtraction Instead of Addition

**Location**: `Optimizers/sgd_momentum.py`, line 49

**Theory**: SGD with momentum accumulates velocity to accelerate convergence in consistent gradient directions
$$v = \mu \cdot v + \text{grad}$$
$$p = p - \text{lr} \cdot v$$

The velocity should **accumulate gradients** by adding them.

**Current Wrong Code**:
```python
v.mul_(mu).sub_(grad)      # ❌ WRONG: subtraction
p.add_(v, alpha=-lr)
```

**Correct Code**:
```python
v.mul_(mu).add_(grad)      # ✅ CORRECT: addition
p.add_(v, alpha=-lr)
```

**Impact** (example: grad=0.1, mu=0.9, lr=0.01):

| Step | Velocity (Wrong) | Update Direction (Wrong) | Velocity (Correct) | Update Direction (Correct) |
|------|-----------------|------------------------|-------------------|---------------------------|
| v = | 0 - 0.1 = **-0.1** | (+0.001) ⬆️ **Ascending** | 0 + 0.1 = **0.1** | (-0.001) ⬇️ **Descending** |
| Next grad | accumulates negatively | keeps going up | accumulates positively | keeps going down |

**Consequence**: 
- Optimizer does **gradient ascent instead of descent**
- Loss increases instead of decreases
- Training goes in completely opposite direction
- Model parameters diverge instead of converge

This is a **critical sign reversal error** that makes the optimizer fundamentally broken.

---

### Error 13: GroupNorm — Incorrect Reshape Dimension Order

**Location**: `Models/Normalizations/groupnorm.py`, line 34

**Theory**: GroupNorm divides C channels into G groups and normalizes **within each group independently**. For input `[B, C, H, W]`:
1. Reshape to `[B, G, C//G, H, W]` to isolate groups
2. Normalize over dimensions `[C//G, H, W]` — each group independently
3. Reshape back to `[B, C, H, W]`

**Current Wrong Code**:
```python
x = x.view(B, C // self.G, self.G, *spatial)  # ❌ WRONG: [B, C//G, G, ...]
dims = tuple(range(2, x.ndim))  # Normalizing over [G, *spatial]
```

**Correct Code**:
```python
x = x.view(B, self.G, C // self.G, *spatial)  # ✅ CORRECT: [B, G, C//G, ...]
dims = tuple(range(2, x.ndim))  # Normalizing over [C//G, *spatial]
```

**Example** (B=2, C=256, G=8, H=28, W=28):

| Dimension | Wrong Shape | Correct Shape | Normalize Over |
|-----------|------------|---------------|----------------|
| Wrong | [2, 32, 8, 28, 28] | — | [8, 28, 28] = 6,272 values ❌ |
| Correct | — | [2, 8, 32, 28, 28] | [32, 28, 28] = 25,088 values ✓ |

**Impact**:
- Groups don't normalize independently (only 6,272 values per group instead of 25,088)
- Batch statistics pollution: channels from different groups mix
- Mean/variance computed over wrong channels
- Normalization statistics are fundamentally broken
- Training becomes unstable with poor gradient flow

---

### Error 14: Pointer Network — Incorrect Matmul Dimension Order

**Location**: `Models/heads.py`, lines 25–26

**Theory**: The Pointer network computes a bilinear dot product between a learned weight vector and context representations:
$$Y = w^T \cdot X$$

For batch and sequence dimensions, this becomes:
$$Y[b, l] = w^T \cdot X[b, :, l]$$

where $w \in \mathbb{R}^{2C}$ and $X \in \mathbb{R}^{B \times 2C \times L}$, producing $Y \in \mathbb{R}^{B \times L}$.

**Current Wrong Code**:
```python
Y1 = torch.matmul(self.w1, X1)  # w1: [2C], X1: [B, 2C, L]
```

**Why This Fails**:
- `torch.matmul([2C], [B, 2C, L])` tries to match dimensions `[1, 2C] @ [B, 2C, L]`
- Inner dimensions don't align: 2C ≠ B → **shape mismatch**
- PyTorch will raise a dimension error

**Correct Code** (Option 1 — Transpose):
```python
Y1 = torch.matmul(X1.transpose(1, 2), self.w1)  # [B, L, 2C] @ [2C] → [B, L]
Y2 = torch.matmul(X2.transpose(1, 2), self.w2)
```

**Correct Code** (Option 2 — Einsum, more explicit):
```python
Y1 = torch.einsum('c,bcl->bl', self.w1, X1)  # Dot product over channel c
Y2 = torch.einsum('c,bcl->bl', self.w2, X2)
```

**Dimension Walkthrough** (transpose approach):

| Step | X1 Shape | After Transpose | w1 Shape | Matmul Result |
|------|----------|-----------------|----------|---------------|
| Input | [B, 2C, L] | [B, L, 2C] | [2C] | [B, L] ✓ |
| Example | [2, 192, 400] | [2, 400, 192] | [192] | [2, 400] ✓ |

**Impact**:
- Runtime error: shape mismatch in matmul
- Forward pass crashes immediately
- No span predictions computed
- Training cannot proceed

---

## Priority Action Plan
1. **Fix Error 3** (StepLR scheduler) — enable training to work at all
2. **Fix Error 7** (EncoderBlock) — enable self-attention to work

### Phase 2: HIGH Priority (Fix Next)
3. **Fix Error 1** (Xavier init) — proper weight initialization
4. **Fix Error 2** (Kaiming init) — avoid dead ReLU
5. **Fix Error 5** (Adam weight decay) — correct regularization
6. **Fix Error 6** (Attention scaling) — stable attention gradients
7. **Fix Error 9** (Lambda scheduler) — correct learning rate multiplication
8. **Fix Error 10** (Adam bias correction) — exponential decay for bias correction
9. **Fix Error 11** (Adam second moment) — square gradients for variance
10. **Fix Error 12** (SGDMomentum velocity) — add instead of subtract gradients
11. **Fix Error 13** (GroupNorm reshape) — correct dimension order for group normalization
12. **Fix Error 14** (Pointer matmul) — correct matrix multiplication dimensions
13. **Fix Error 15** (Lambda scheduler pickling) — use named function instead of lambda
14. **Fix Error 16** (Lambda scheduler warmup) — implement QANet paper warmup scheme + use args.learning_rate in Adam
15. **Fix Error 17** (PyTorch 2.6 evaluation loading) — set `weights_only=False` for trusted local checkpoints

### Phase 3: MEDIUM Priority
7. **Fix Error 4** (Cosine scheduler) — correct learning rate schedule

### Phase 4: OPTIONAL
8. **Consider Error 8** (Early stopping) — only if you want more aggressive stopping

---

## Testing & Validation

After applying all fixes, verify:

✅ **Immediate checks**:
- Training starts without crashes
- Learning rate is positive (not zero)
- Loss computes without NaN/Inf

✅ **After first epoch**:
- Training loss decreases
- Gradients are non-zero and bounded
- Validation metrics improve over random baseline

✅ **After multiple epochs**:
- F1 and EM metrics show consistent improvement
- Learning rate follows expected schedule
- Model converges to reasonable performance

✅ **Model behavior**:
- Attention weights are well-distributed (not stuck on one token)
- Gradients flow through all components
- Convergence speed is reasonable (not too slow)

---

## Files to Modify

| # | File | Line(s) | Change |
|----|------|---------|--------|
| 1 | `Models/Initializations/xavier.py` | 33 | `* fan_out` → `+ fan_out` |
| 2 | `Models/Initializations/kaiming.py` | 31 | `1.0 / fan` → `2.0 / fan` |
| 3 | `Schedulers/step_scheduler.py` | 26 | `gamma * (t // step)` → `(gamma ** (t // step))` |
| 4 | `Schedulers/cosine_scheduler.py` | 25 | Add `0.5 *` factor |
| 5 | `Optimizers/adam.py` | 54 | `alpha=-wd` → `alpha=wd` |
| 6 | `Models/encoder.py` | 62 | Add `* self.scale` |
| 7 | `Models/encoder.py` | 99, 111 | Remove duplicate conv, change `=` to `+=` |
| 8 | `TrainTools/train.py` | 196 | (Optional) Change AND to OR |
| 9 | `Schedulers/lambda_scheduler.py` | 22 | `+` → `*` |
| 10 | `Optimizers/adam.py` | 70-71 | `beta * t` → `beta ** t` |
| 11 | `Optimizers/adam.py` | 73 | Add `** 2` for gradient squaring |
| 12 | `Optimizers/sgd_momentum.py` | 49 | `.sub_(grad)` → `.add_(grad)` |
| 13 | `Models/Normalizations/groupnorm.py` | 34 | `[B, C//G, G]` → `[B, G, C//G]` |
| 14 | `Models/heads.py` | 25-26 | Use transpose/einsum for correct matmul |
| 15 | `Schedulers/scheduler.py` | 24 | Replace lambda with named function |
| 16 | `Optimizers/optimizer.py` | 14 | `lr=1.0` → `lr=args.learning_rate` |
| 16 | `Schedulers/scheduler.py` | 25-31 | Replace `constant_lambda` with `warmup_lambda` |
| 17 | `EvaluateTools/evaluate.py` | 118 | `torch.load(..., weights_only=False)` for trusted checkpoints |


---

## Next Steps

1. **Read** `ERROR_ANALYSIS.ipynb` for detailed explanations with formulas
2. **Reference** `FIXES_QUICK_REFERENCE.md` for side-by-side code diffs
3. **Implement** fixes in priority order (Critical → High → Medium → Low)
4. **Test** training pipeline after each fix
5. **Validate** using checklist above

Good luck! 🚀

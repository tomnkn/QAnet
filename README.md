# Assignment 1 — QANet

**COMP5329 Deep Learning — University of Sydney, Semester 1 2026**

This repository contains a PyTorch implementation of [QANet](https://arxiv.org/pdf/1804.09541.pdf) for extractive question answering on SQuAD v1.1. The distributed code contains **intentional bugs** — your task is to find and fix them all. See the assignment spec for details.

The entire pipeline (download, preprocess, train, evaluate) is driven from a single notebook: **`assignment1.ipynb`**.

---

## Getting Started on Google Colab

### 1 — Clone the repo into Google Drive

Open a **new notebook** at [colab.research.google.com](https://colab.research.google.com) and run:

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
import os
REPO_URL     = "https://github.com/usyddeeplearning/Assignment1--2026.git"
PROJECT_ROOT = "/content/drive/MyDrive/Assignment1--2026"

if not os.path.exists(PROJECT_ROOT):
    !git clone {REPO_URL} {PROJECT_ROOT}
    print("Done.")
else:
    print("Already cloned — skipping.")
```

Close this temporary notebook when done.

### 2 — Open the assignment notebook

In Google Drive, navigate to `MyDrive/Assignment1--2026/` and double-click **`assignment1.ipynb`** to open it in Colab. Run cells in order from Section 0 downward — the notebook handles dependency installation, data download, preprocessing, training, and evaluation.

> Your files live on Google Drive, so they persist across Colab sessions. You only need to clone once.

### 3 — Pulling updates

If the repo is updated after you've cloned it, open a Colab cell and run:

```python
!cd /content/drive/MyDrive/Assignment1--2026 && git pull
```

Be careful — if you've edited files that were also updated upstream, you may get merge conflicts. Commit or back up your changes first.

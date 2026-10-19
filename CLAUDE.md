# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Coursework for the NYCU Data Mining course (student ID `311351009`). It is **four independent assignments**, not one application — there is no shared package, build system, lint config, test suite, or `requirements.txt`. Each assignment was developed and run in a different environment (Google Colab or a multi-GPU lab server), so paths and conventions differ between them. Treat each directory as its own self-contained project.

`README.md` documents the data-mining technique behind each assignment (in Chinese); consult it for the *why*. This file covers the *how to run* and the cross-cutting gotchas.

## The four assignments

| Dir | Task | Entry point | Stack |
|-----|------|-------------|-------|
| `HW1/` | PM2.5 time-series regression (predict hour 10 from prior 9 hours) | `HW1_311351009.ipynb` | NumPy-only linear regression (gradient descent **and** closed-form normal equation, L2/Ridge) |
| `HW2/` | Amazon review rating classification (1–5) | `HW2_311351009.ipynb` | `BertForSequenceClassification`, multi-GPU |
| `HW3/` | Outlier detection via prediction uncertainty | `HW3_311351009.py` | scikit-learn `RandomForestClassifier` |
| `final_9/` | News recommendation / click-through-rate prediction | `final_9.py` | PyTorch LSTM + RoBERTa embeddings, multi-GPU |

Naming convention: deliverables are `HW{n}_311351009.{ipynb,py,pdf,docx}` — code plus a paired report.

## Running each assignment

**HW1** — open the notebook in Colab/Jupyter and run top-to-bottom. Data paths are hardcoded to Colab (`/content/train.csv`); the CSVs (`train.csv`, `test.csv`) are in `HW1/`, so upload them or rewrite the paths to be relative. `summit.csv` is the prediction output. The `LinearRegression` class is implemented from scratch — `fit(..., method='gradient_descent' | 'closed_form')` selects the solver.

**HW2** — notebook, written for a multi-GPU server with hardcoded paths under `/home/jeremy/HW/`. It converts `train.json` → `train.csv`, combines `title + text` into one `review` field, tokenizes with `bert-base-uncased` (max_length 512), and fine-tunes with early stopping (patience 3) on Macro-F1. Edit the `/home/jeremy/HW/` paths before running elsewhere.

**HW3** — the only plain script. Run **from inside `HW3/`** because it uses relative `./data/...` paths:
```bash
cd HW3 && python HW3_311351009.py
```
Key idea: it does **not** classify outliers directly. It trains a Random Forest, takes `predict_proba` on the test set, and scores each sample by `1 / MSE(prob, uniform_distribution)` — high inverse-MSE (prob far from uniform = confident) means *not* an outlier. Writes `random_forest.csv` (`id,outliers`). `random_forest_30_9706.csv` is a kept best-run output.

**final_9** — config-driven training/inference. Run from inside `final_9/`:
```bash
cd final_9 && python final_9.py
```
Behavior is controlled entirely by `config.py` (`Config` class) — there are no CLI args:
- `Config.test = False` → **train** (writes logs, checkpoints, and copies `config.py` into `output_path`).
- `Config.test = True` → **inference**: loads `model_weight`, predicts, and fills `submission.csv` → `output.csv`.

## Cross-cutting gotchas

- **Hardcoded absolute paths everywhere.** `final_9/config.py` points at `/home/hyhsieh/DM/final_data`; HW2 at `/home/jeremy/HW/`; HW1 at `/content/`. The final-project dataset is **not in the repo** (it lives at that server path) — `final_9/` ships only code + report. Always check and fix paths first when running in a new environment.
- **`ranger21` is a required, non-obvious dependency** (the Ranger21 optimizer = Lookahead + RAdam), used in both HW2 and `final_9`. It is not a standard `pip install ranger21` package — install from the upstream Ranger21 repo. `Ranger21(...)` requires `num_epochs` and `num_batches_per_epoch` up front.
- **Multi-GPU is assumed.** HW2 and `final_9` set `os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,...,7"` and wrap the model in `nn.DataParallel` when `device_count() > 1`. When saving/loading, `DataParallel` wraps weights under `model.module` — HW2 saves `model.module.state_dict()`. Mind this prefix when loading checkpoints on a single GPU or CPU.
- **`final_9` embedding cache.** News titles/abstracts are encoded with RoBERTa once and pickled to `{train,test}_news_embeddings_dict.pkl`. On startup it loads the `.pkl` if present (in `base_root`); otherwise it regenerates from scratch (slow). To force re-encoding, delete the `.pkl`.
- **`final_9` interaction model.** The LSTM produces a user vector from the last `MAX_CLICKED_LEN = 35` clicked-news embeddings; click probability is the **dot product** of that user vector with each candidate (impression) news embedding, trained with `BCEWithLogitsLoss` and scored by AUC.

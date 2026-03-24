# Verifier-Guided Code Generation via a Learned Candidate Ranker

This repository accompanies the term paper **“Verifier-Guided Code Generation via a Learned Candidate Ranker”** (Machine Learning for Natural Language Understanding, University of Trier). It contains the implementation and experiment artifacts for a compact study of **selection under test-budget constraints** in code generation.

The project treats a common workflow as a research problem: generate multiple candidate solutions, verify them by execution when possible, and decide which candidate to keep. Here, unit tests act as an objective verifier signal, and a learned ranker is trained to prefer candidates that satisfy the task specification.

---

## What this project studies

Given a programming task with a specification (prompt), we sample **N** candidate programs from a code LLM. Each candidate is labeled by executing unit tests (pass/fail). Using this verifier signal, we train a **prompt-conditioned pairwise ranker** (CodeBERT-based) that scores candidate code *relative to the task*. At inference time, the ranker supports:

- **Top-1 selection without testing** (ranker-only selection), and  
- **Top-k shortlist + tests** (budgeted verification), where only the k highest-ranked candidates are tested.

The experiments are designed to answer:
- **RQ1:** How do sampling and selection interact as N increases?  
- **RQ2:** What is the accuracy–verification trade-off (pass@1 vs. tests executed)?  
- **RQ3:** Does a ranker trained at N=8 generalize to larger candidate sets (N=16)?

Benchmark: **HumanEval (164 tasks)**.

---

## Repository contents

### Notebook
- `main_.ipynb` — the full experiment notebook (with markdown narration), used to run the pipeline end-to-end and record intermediate outputs and results.

### Code
- `src/generate_candidates.py` — sample N candidates per HumanEval task from a generator LLM.
- `src/run_tests.py` — execute HumanEval unit tests for each (task, candidate) to obtain pass/fail labels.
- `src/evaluate_baselines.py` — compute first-sample, random, and best-of-N baselines.
- `src/build_ranker_dataset.py` — build prompt-conditioned pairwise training data from verifier labels.
- `src/train_ranker.py` — train a pairwise ranker using `microsoft/codebert-base` as the encoder.
- `src/evaluate_ranker_selection.py` — evaluate ranker top-1 and ranker top-k + tests policies.
- `src/error_analysis_shortlist.py` — shortlist generation-limited tasks and ranker misses.
- `src/utils.py` — shared utilities (I/O, dataset loading, formatting helpers).

### Configuration
- `configs/exp.yaml` — experiment configuration (paths, generator settings, decoding, sampling).

### Outputs (usually not committed)
- `outputs/candidates/` — generated candidates (JSONL).
- `outputs/tests/` — verifier results (JSONL).
- `outputs/ranker_data/` — pair datasets and ranker checkpoints.
- `outputs/results/` — summary CSVs / logs.

---

## Method at a glance

- **Generator:** `Qwen2.5-Coder-7B-Instruct` (stochastic decoding; temperature and top-p fixed in config)  
- **Verifier:** HumanEval unit tests (isolated subprocess, fixed timeout)  
- **Ranker:** prompt-conditioned cross-encoder using `microsoft/codebert-base` + scalar scoring head  
- **Training signal:** within-task pairwise supervision (pass vs. fail candidates for the same task)  
- **Selection policies:** first-sample, random, best-of-N (test-all), ranker top-1, ranker top-k + tests  
- **Metrics:** pass@1, and verification cost measured as tests executed per task

---

## Notes on reproducibility

This repo is organized so the pipeline can be rerun end-to-end by following the notebook, but large intermediate artifacts (candidate/test JSONLs, best models) can become heavy and are excluded from version control. The paper reports the exact configuration and hyperparameters used in the final runs.

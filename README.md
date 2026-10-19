# Belief state geometry in a tiny transformer, with a mouse-like extension

Status: work in progress. Milestone 0 of 6 (environment set up) is done.

This project replicates, at small scale, the central result of Shai et al. (2024): a transformer trained only to predict the next token of a hidden Markov process ends up with a linear representation of the Bayes-optimal belief over the hidden states in its residual stream. It then asks whether the same holds for a process whose hidden dynamics change over time, loosely inspired by mice switching between behavioural states within a session. All data are simulated.

## Setup

Tested with Python 3.13 on Windows 11, CPU only.

```bash
python -m pip install -r requirements.txt
python -m pytest
```

## Repository layout

* `src/` holds all the logic, with one module per step of the pipeline.
* `tests/` holds the unit tests, run with `python -m pytest`.
* `notebooks/` will only call functions from `src/` and draw figures.
* `docs/paper_notes.md` lists every detail taken from the paper, with section numbers, and every choice made where the paper is silent.

## Reference

Shai, A. S., Marzen, S. E., Teixeira, L., Gietelink Oldenziel, A. and Riechers, P. M. (2024). Transformers represent belief state geometry in their residual stream. arXiv:2405.15943.

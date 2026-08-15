# configs/

This pipeline has no separate configuration file or module. Every parameter
that affects a run — die style, seed, imaging modality, noise level, scale
ratio, rotation range, process-variation amplitude, boundary bias, image
size — is an explicit, documented command-line flag on `src/generate_dataset.py`
and `src/localize.py`.

See the "Options" table under **Quick start → 1. Generate a dataset** in the
top-level `README.md` for the full list of flags and their defaults. Every
value used to produce a committed result is also recorded per-pair in that
run's `ground_truth.csv`/`ground_truth.json` and `results.csv` under
`results/`, so a config is always recoverable from the output it produced
even without a separate config file.

This folder is kept (rather than omitted) only so the submission's directory
listing matches the challenge's recommended layout one-to-one.

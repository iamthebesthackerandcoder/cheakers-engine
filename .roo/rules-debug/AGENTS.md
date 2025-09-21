# AGENTS.md

This file provides guidance to agents when working with code in this repository.

- Shared TT dict has serialization overhead in multiprocessing; monitor memory.
- CuPy/Torch fallbacks to NumPy if not installed; check for silent slowdowns.
- Global CAPTURES_MANDATORY changes propagate unexpectedly across threads.
- Extension logs in specific channels; grep for 'neural' or 'engine'.
- Production requires optional libs (Torch/Numba); features break silently without.
- In-place weight mutations during training can cause non-deterministic evals.
- Epsilon decay affects reproducibility; fix seed for debugging selfplay.

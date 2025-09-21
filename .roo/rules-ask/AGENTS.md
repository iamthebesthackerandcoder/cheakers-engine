# AGENTS.md

This file provides guidance to agents when working with code in this repository.

- checkers/ is thin wrapper over legacy [`gameotherother.py`](gameotherother.py) (counterintuitive).
- Root holds models/data (e.g., .pkl, .npz), not checkers/.
- Two systems: engine for logic (checkers/engine.py), neural for eval (neural_eval.py).
- Scripts run from root; no cd into checkers/ needed.
- Self-play generates data for neural updates via curriculum_trainer.py.

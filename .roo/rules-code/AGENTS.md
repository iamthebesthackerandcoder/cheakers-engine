# AGENTS.md

This file provides guidance to agents when working with code in this repository.

- Sync CAPTURES_MANDATORY globally before legal_moves() calls.
- Use NeuralEvaluator singleton with hash-based caching (10k limit).
- Mutate opponent weights in-place (15% rate, 0.3 std noise) during selfplay chunks.
- Avoid raw queries; use abstractions in checkers/engine.py and checkers/search.py.
- Test files import from checkers/training/ and root-level modules correctly.
- Board uses 1-32 indices for dark squares only in [`gameotherother.py`](gameotherother.py).
- Apply 50% board flips for data augmentation in training.
- Epsilon decay from 0.1 to 0.01 over 100 games in selfplay.

# AGENTS.md

This file provides guidance to agents when working with code in this repository.

- Pre-commit runs full pytest suite on all .py files.
- Black/Ruff: line-length=100.
- Mypy: ignore missing imports for cupy.*, torch.*, numba.*; warn on unused ignores/redundant casts/return-any.
- Single test: pytest test_curriculum.py -k "test_name".
- Code style: line-length=100 (vs 88); ruff E/F/I/N/UP/B; imports combine as-imports, force-wrap long aliases.
- Board uses 1-32 indices for dark squares only [`gameotherother.py`](gameotherother.py).
- Global CAPTURES_MANDATORY affects legal_moves().
- Neural eval caches by hash(tuple(board)+player) with 10k limit.
- Opponent mutation (15% rate, 0.3 std noise) in [`selfplay_trainer.py`](selfplay_trainer.py).
- Data augmentation via 50% board flips.
- Shared Manager.dict() for TT in multiprocessing.
- Tests: root-level [`test_curriculum.py`](test_curriculum.py), [`test_improvements.py`](test_improvements.py); require torch/numba for full coverage.
- Pre-commit pytest: --maxfail=1 --disable-warnings -q on all files.

Documentation
- Consolidated architecture and roadmap: see architecture_and_todo.md

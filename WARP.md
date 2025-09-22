# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.
``

Project overview
- Python 3.11+ Checkers AI with:
  - Core engine and alpha–beta search (gameotherother.py) wrapped by the checkers/ package
  - Optional neural evaluator (neural_eval.py) for position scoring
  - Self-play training (selfplay_trainer.py) with a curriculum mode (curriculum_trainer.py)
  - Tkinter GUI (checkers/gui/*) with modular components and a simple entrypoint (run_gui.py)
- Tooling: pytest, ruff, black, mypy, pre-commit hooks declared in pyproject.toml and .pre-commit-config.yaml

Environment setup (Windows PowerShell)
- Create and activate a virtualenv (Python 3.11+):
  - py -3.11 -m venv .venv
  - .\.venv\Scripts\Activate.ps1
- Install dependencies:
  - python -m pip install -U pip
  - pip install -r requirements.txt
  - pip install pytest black ruff mypy pre-commit
- Install git hooks (recommended):
  - pre-commit install

Common commands
- Lint and format
  - Run all hooks: pre-commit run -a
  - Ruff (lint): ruff check .
  - Ruff (format): ruff format .
  - Black (format): black .
  - Black (check only): black --check .
  - Type-check: mypy --config-file pyproject.toml .
- Tests
  - Run all: pytest -q
  - Run a single file: pytest -q test_improvements.py
  - Run a single test function: pytest -q test_improvements.py::test_adam_optimizer
  - Filter by keyword: pytest -q -k "curriculum or adam"
- GUI
  - Optional defaults via env vars:
    - $env:CHECKERS_LOG_LEVEL = "INFO"   # DEBUG|INFO|WARNING|ERROR
    - $env:CHECKERS_MANDATORY = "true"   # mandatory captures default
  - Launch: python run_gui.py
- Training (CLI)
  - Self-play training: python train.py --games 200 --save-interval 25 --workers 4
  - Artifacts default to data/ (see config.py); created automatically
- Benchmarks & profiling
  - Sequential benchmark: python benchmark_seq.py
  - Parallel benchmark: python benchmark_parallel.py
  - Profiling sample: python profile_run.py

Architecture overview
- Core game logic and search (gameotherother.py)
  - Board model: 1..32 dark-square index, board is a list of length 33 (index 0 unused)
  - Legal move generation supports multi-jumps; captures are prioritized; CAPTURES_MANDATORY gate
  - apply_move handles captures and promotion; is_terminal detects side-to-move with no moves
  - Evaluation: material-based heuristic; SearchEngine performs alpha–beta with simple TT cache
  - Neural integration: SearchEngine.neural_evaluator if set will score leaf nodes; otherwise falls back to heuristic
- Package wrappers (checkers/)
  - engine.py: Thin facade over gameotherother with a synchronized CAPTURES_MANDATORY flag and re-exports (initial_board, legal_moves, apply_move, get_engine, etc.)
  - search.py: Re-exports SearchEngine / minimax
  - eval.py: Re-exports neural_eval interfaces (NeuralEvaluator, TrainingDataCollector, get_neural_evaluator)
  - training/{selfplay,curriculum}.py: Re-export training classes
- Neural evaluation (neural_eval.py)
  - Vectorized feature extraction and forward pass; optional acceleration (Numba/CuPy/Torch if installed)
  - Simple supervised training loop (Adam), position cache and augmentation (board flipping)
  - TrainingDataCollector stores (positions, scores) and saves to compressed .npz
- Self-play training (selfplay_trainer.py)
  - Generates games between a base evaluator and diversified opponents (noise/mutation)
  - Supports sequential and parallel play (ParallelGameRunner via multiprocessing)
  - Periodically trains on collected data; saves model (pickle), data (.npz), and a checkpoint (stats + optimizer state)
  - CLI entry in train.py exposes core parameters (games, save interval, workers, output paths)
- Curriculum learning (curriculum_trainer.py)
  - Adds phased training (opening→middlegame→advanced→endgame), position importance scoring, and game quality filters
  - Exposed via checkers.training.curriculum.CurriculumTrainer and demo tests in test_curriculum.py
- GUI (checkers/gui/*)
  - CheckersUI orchestrates modular components:
    - GameState: board/current player/move history and results
    - MoveManager: legal move computation, grouping, and display strings
    - BoardRenderer: drawing squares, pieces, selection, last move, hints
    - EngineIntegration: depth, (optional) neural evaluation, async search
    - TrainingIntegration: background self-play training with progress UI
  - Entry shim: checkers/gui/tk_app.py exposes CheckersUI for run_gui.py
  - There is an older single-file UI (checkers_gui_tk.py) retained for reference
- Configuration and artifacts (config.py)
  - Pydantic-based configuration with env var overrides; helpers ensure data/ and models/ exist
  - Defaults: data paths (DEFAULT_MODEL_PATH, DEFAULT_DATA_PATH → .npz, DEFAULT_CHECKPOINT_PATH), epsilon schedule, search defaults
  - Logging setup via CHECKERS_LOG_LEVEL

Key notes
- Python version: requires-python >= 3.11 (pyproject); mypy targets py313; run mypy with the provided config to align types
- Optional accelerators: install as needed for speedups (e.g., pip install numba torch). CuPy may require a CUDA-specific wheel
- Hooks: pre-commit includes black, ruff (with --fix), ruff-format, mypy, and a pytest sanity run
- Data/models: artifacts are saved under data/ by default; checkpoints contain model state and training stats

Documentation
- Consolidated architecture and roadmap: see architecture_and_todo.md

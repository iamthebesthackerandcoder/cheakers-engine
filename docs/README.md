# Project Documentation

This repository contains a Checkers AI project with the following modules:

- Core engine (legacy): gameotherother.py
- Modernized core components:
  - checkers/moves.py: BoardState, MoveGenerator, MoveValidator
  - checkers/eval.py: Evaluator (ABC), NeuralEvaluatorAdapter and compatibility exports
  - checkers/search.py: SearchStrategy (ABC), AlphaBetaSearchStrategy and compatibility exports
- Training:
  - selfplay_trainer.py and curriculum_trainer.py
  - checkers/training/: thin wrappers for trainers
- GUI:
  - checkers/gui/: Tk-based UI integration

Recent refactors
- Phase 1: Decoupled move generation from legacy engine, added unit tests.
- Phase 2: Introduced evaluation and search interfaces with adapters, added tests.
- Phase 3: Enhanced config validation for rules using Pydantic v2 field validators.

How to run tests
- pytest

How to run performance benchmarks
- python benchmark_improvements.py
- python benchmark_seq.py
- python benchmark_parallel.py

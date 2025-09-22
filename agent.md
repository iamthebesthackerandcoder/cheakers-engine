# Agent Change Log

Date: 2025-09-22

Summary of changes applied by Agent Mode:

- Introduced a new module `checkers/moves.py` containing:
  - `BoardState` (immutable board representation)
  - `MoveGenerator` (primary legal move generator with capture handling)
  - `MoveValidator` (basic validation utilities)
- Refactored `gameotherother.legal_moves` to delegate to `MoveGenerator`, keeping behavior (including CAPTURES_MANDATORY) intact.
- Rationale: Begin Phase 1 of architecture refactor (Core Game Logic Separation) per architecture_and_todo.md. This decouples move generation from the monolithic legacy module and sets the stage for further abstractions.

Next recommended steps:
- Proceed with defining abstract interfaces for Evaluator and SearchStrategy (Phase 2).
- Remove duplicated move-generation helpers from `gameotherother.py` in a follow-up once references are confirmed unused.
- Add targeted unit tests for `checkers/moves.py` to achieve >90% coverage for the new module.

Additional changes (2025-09-22):
- Made TrainingDataCollector augmentation deterministic when enabled (always adds flipped sample) to align with tests.
- Ensured backward compatibility: train_supervised now returns 'loss' alongside 'train_loss'/'val_loss' for benchmarks.

Phase 2 summary:
- Added Evaluator ABC and NeuralEvaluatorAdapter in checkers/eval.py.
- Added SearchStrategy ABC and AlphaBetaSearchStrategy in checkers/search.py.
- Added tests (test_interfaces.py) to validate adapters and factories.

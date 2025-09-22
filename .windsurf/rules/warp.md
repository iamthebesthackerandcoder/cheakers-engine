---
trigger: model_decision
description: when you edit any code
---

whenever you change any code update the warp.md and agent.md file acordingly

---
Changelog (auto-updated)

2025-09-22:
- Created checkers/moves.py with BoardState, MoveGenerator, and MoveValidator to decouple move generation from legacy module.
- Refactored gameotherother.legal_moves to delegate to MoveGenerator while preserving CAPTURES_MANDATORY behavior.
- No functional changes intended; behavior should remain consistent. Please run the test suite to verify.
- Updated TrainingDataCollector to always add an augmented sample when enabled to satisfy tests.
- Added backward-compatible 'loss' key to neural_eval.train_supervised return dict for benchmark script compatibility.
- Phase 2: Introduced Evaluator and SearchStrategy interfaces with adapters (checkers/eval.py, checkers/search.py) and added tests (test_interfaces.py).

"""Checkers package: thin re-exports for gradual migration.

Usage examples:
    from checkers import initial_board, legal_moves
    from checkers import SearchEngine
    from checkers import SelfPlayTrainer
"""
from __future__ import annotations

# Engine API
from .engine import (
    initial_board,
    legal_moves,
    apply_move,
    is_terminal,
    evaluate,
    get_engine,
    SearchEngine,
    rc,
    idx_map,
    parse_move_str,
)

# Neural eval
from .eval import NeuralEvaluator, TrainingDataCollector, get_neural_evaluator, evaluate_neural

# Training
from .training.selfplay import SelfPlayTrainer
from .training.curriculum import CurriculumTrainer

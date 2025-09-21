"""
Thin wrapper re-exporting neural evaluation components.
"""
from __future__ import annotations

from neural_eval import NeuralEvaluator, TrainingDataCollector, get_neural_evaluator, evaluate_neural

__all__ = [
    "NeuralEvaluator",
    "TrainingDataCollector",
    "get_neural_evaluator",
    "evaluate_neural",
]

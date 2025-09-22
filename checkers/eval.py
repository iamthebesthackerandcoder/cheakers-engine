"""
Evaluation interfaces and adapters, plus re-exports for backward compatibility.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Union

import numpy as np

# Backward-compatible imports
from neural_eval import (
    NeuralEvaluator,
    TrainingDataCollector,
    get_neural_evaluator,
    evaluate_neural,
)


class Evaluator(ABC):
    """Abstract evaluator interface for position scoring."""

    @abstractmethod
    def evaluate_position(self, board: List[int], player: int) -> float:  # pragma: no cover
        """Evaluate a single board position for the given player."""
        raise NotImplementedError

    def batch_predict(
        self,
        boards: Union[List[List[int]], np.ndarray],
        players: Union[List[int], np.ndarray],
    ) -> np.ndarray:
        """Optional: batch prediction for multiple boards. Default uses vectorization via single calls."""
        # Default fallback using per-position evaluation
        if isinstance(boards, list):
            boards_np = np.array(boards, dtype=np.float32)
        else:
            boards_np = boards
        if isinstance(players, list):
            players_np = np.array(players, dtype=np.float32)
        else:
            players_np = players
        out = np.zeros(len(boards_np), dtype=np.float32)
        for i in range(len(boards_np)):
            out[i] = float(self.evaluate_position(list(boards_np[i]), int(players_np[i])))
        return out


class NeuralEvaluatorAdapter(Evaluator):
    """Adapter to make neural_eval.NeuralEvaluator conform to Evaluator interface."""

    def __init__(self, impl: Optional[NeuralEvaluator] = None) -> None:
        self.impl: NeuralEvaluator = impl or get_neural_evaluator()

    def evaluate_position(self, board: List[int], player: int) -> float:
        return float(self.impl.evaluate_position(board, player))

    def batch_predict(
        self,
        boards: Union[List[List[int]], np.ndarray],
        players: Union[List[int], np.ndarray],
    ) -> np.ndarray:
        return self.impl.batch_predict(boards, players)


# Factory to get an Evaluator-conforming object

def get_evaluator() -> Evaluator:
    return NeuralEvaluatorAdapter(get_neural_evaluator())


__all__ = [
    # New interfaces
    "Evaluator",
    "NeuralEvaluatorAdapter",
    "get_evaluator",
    # Backward-compatible exports
    "NeuralEvaluator",
    "TrainingDataCollector",
    "get_neural_evaluator",
    "evaluate_neural",
]

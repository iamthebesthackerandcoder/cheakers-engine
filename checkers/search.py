"""
Search interfaces and strategy adapters, plus backward-compatible exports.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from gameotherother import SearchEngine, minimax

Board = List[int]
Move = List[int]
GameResult = Tuple[int, Optional[Move]]


class SearchStrategy(ABC):
    """Abstract interface for search strategies."""

    @abstractmethod
    def search(self, board: Board, player: int, depth: int) -> GameResult:  # pragma: no cover
        raise NotImplementedError


class AlphaBetaSearchStrategy(SearchStrategy):
    """Adapter around the existing SearchEngine implementing the interface."""

    def __init__(self) -> None:
        self._engine = SearchEngine()

    def search(self, board: Board, player: int, depth: int) -> GameResult:
        return self._engine.search(board, player, depth)


def get_search_strategy() -> SearchStrategy:
    """Factory for a default search strategy (alpha-beta)."""
    return AlphaBetaSearchStrategy()


__all__ = [
    # New interfaces
    "SearchStrategy",
    "AlphaBetaSearchStrategy",
    "get_search_strategy",
    # Backward-compatible exports
    "SearchEngine",
    "minimax",
]

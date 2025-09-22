"""
Type definitions and protocols for the Checkers AI system.

This module provides comprehensive type safety through:
- Protocol definitions for interfaces
- Dataclass implementations for game state
- Type aliases for better code readability
- Generic types for neural network operations
"""

from __future__ import annotations

import numpy as np
from typing import Protocol, List, Tuple, Dict, Any, Optional, Union, Callable, TypeVar, Generic
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Type variables for generic types
T = TypeVar('T')
BoardState = TypeVar('BoardState')
MoveSequence = TypeVar('MoveSequence')

# Basic type aliases
Board = List[int]  # Board state: index 0 is unused, 1-32 are squares
Move = List[int]   # Move sequence: [from, to, ...] for jumps
GameResult = Tuple[int, Optional[Move]]  # (score, best_move)
Player = int  # 1 for black, -1 for red
SquareIndex = int  # 1-32 for dark squares
Position = Tuple[int, int]  # (row, col) coordinates

# Neural network types
NeuralNetwork = Dict[str, np.ndarray]
TrainingData = Tuple[np.ndarray, np.ndarray]  # (features, targets)
EvaluationCache = Dict[str, float]
AdamState = Dict[str, Dict[str, np.ndarray]]  # Optimizer state

# Configuration types
ConfigDict = Dict[str, Any]
SettingsDict = Dict[str, Union[str, int, float, bool]]


@dataclass(frozen=True)
class GameState:
    """
    Immutable representation of a checkers game state.

    This dataclass provides a clean, type-safe way to represent
    the complete state of a checkers game.
    """
    board: Board
    current_player: Player
    move_count: int = 0
    is_terminal: bool = False
    winner: Optional[Player] = None

    def __post_init__(self) -> None:
        """Validate the game state after initialization."""
        if not isinstance(self.board, list) or len(self.board) != 33:
            raise ValueError("Board must be a list of 33 integers")
        if self.current_player not in [-1, 1]:
            raise ValueError("Current player must be -1 (red) or 1 (black)")
        if not (0 <= self.move_count):
            raise ValueError("Move count must be non-negative")


@dataclass
class TrainingSample:
    """A single training sample for the neural network."""
    board_state: Board
    player: Player
    target_value: float
    features: Optional[np.ndarray] = None
    augmented: bool = False


@dataclass
class TrainingStats:
    """Statistics from a training session."""
    games_played: int = 0
    black_wins: int = 0
    red_wins: int = 0
    draws: int = 0
    avg_game_length: float = 0.0
    positions_collected: int = 0
    random_moves_total: int = 0
    random_moves_per_game: List[int] = None  # type: ignore
    current_epsilon: float = 0.0

    def __post_init__(self) -> None:
        """Initialize mutable fields."""
        if self.random_moves_per_game is None:
            self.random_moves_per_game = []

    def __getitem__(self, key: str) -> Any:
        """Support dictionary-style access for backward compatibility."""
        if key == 'games_played':
            return self.games_played
        elif key == 'black_wins':
            return self.black_wins
        elif key == 'red_wins':
            return self.red_wins
        elif key == 'draws':
            return self.draws
        elif key == 'avg_game_length':
            return self.avg_game_length
        elif key == 'positions_collected':
            return self.positions_collected
        elif key == 'random_moves_total':
            return self.random_moves_total
        elif key == 'random_moves_per_game':
            return self.random_moves_per_game
        elif key == 'current_epsilon':
            return self.current_epsilon
        else:
            raise KeyError(f"'{key}' not found in TrainingStats")

    def __setitem__(self, key: str, value: Any) -> None:
        """Support dictionary-style assignment for backward compatibility."""
        if key == 'games_played':
            self.games_played = int(value)
        elif key == 'black_wins':
            self.black_wins = int(value)
        elif key == 'red_wins':
            self.red_wins = int(value)
        elif key == 'draws':
            self.draws = int(value)
        elif key == 'avg_game_length':
            self.avg_game_length = float(value)
        elif key == 'positions_collected':
            self.positions_collected = int(value)
        elif key == 'random_moves_total':
            self.random_moves_total = int(value)
        elif key == 'random_moves_per_game':
            if isinstance(value, list):
                self.random_moves_per_game = value
            else:
                raise ValueError("random_moves_per_game must be a list")
        elif key == 'current_epsilon':
            self.current_epsilon = float(value)
        else:
            raise KeyError(f"'{key}' not found in TrainingStats")


@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    hit_rate: float = 0.0
    cache_size: int = 0


class GameEngineProtocol(Protocol):
    """Protocol for game engine implementations."""

    def search(self, board: Board, player: Player, depth: int) -> GameResult:
        """Search for the best move given a board state."""
        ...

    def evaluate(self, board: Board, player: Player) -> int:
        """Evaluate a board position."""
        ...


class NeuralNetworkProtocol(Protocol):
    """Protocol for neural network implementations."""

    def evaluate_position(self, board: Board, player: Player) -> float:
        """Evaluate a single board position."""
        ...

    def batch_predict(self, boards: Union[List[Board], np.ndarray],
                     players: Union[List[Player], np.ndarray]) -> np.ndarray:
        """Evaluate multiple board positions in batch."""
        ...

    def train_supervised(self, positions: np.ndarray, targets: np.ndarray,
                        epochs: int = 1, batch_size: int = 1024,
                        lr: float = 5e-4, l2: float = 1e-6,
                        shuffle: bool = True, verbose: bool = True) -> Dict[str, Union[float, int]]:
        """Train the network on supervised data."""
        ...


class TrainingDataCollectorProtocol(Protocol):
    """Protocol for training data collection."""

    def add_position(self, board: Board, player: Player, game_result: float) -> None:
        """Add a training position."""
        ...

    def save_training_data(self, filepath: str) -> None:
        """Save collected training data."""
        ...

    def load_training_data(self, filepath: str) -> None:
        """Load training data from file."""
        ...

    def clear(self) -> None:
        """Clear all collected data."""
        ...


class SearchEngineProtocol(Protocol):
    """Protocol for search engine implementations."""

    def __init__(self, seed: Optional[int] = None,
                 shared_tt: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the search engine."""
        ...

    def search(self, board: Board, player: Player, depth: int) -> GameResult:
        """Perform search to find the best move."""
        ...


class PositionEvaluatorProtocol(Protocol):
    """Protocol for position evaluation functions."""

    def __call__(self, board: Board, player: Player) -> float:
        """Evaluate a position and return a score."""
        ...


class MoveGeneratorProtocol(Protocol):
    """Protocol for move generation functions."""

    def __call__(self, board: Board, player: Player) -> List[Move]:
        """Generate all legal moves for a player."""
        ...


class BoardUtilitiesProtocol(Protocol):
    """Protocol for board utility functions."""

    def initial_board(self) -> Board:
        """Create the initial board state."""
        ...

    def count_pieces(self, board: Board) -> Tuple[int, int, int, int]:
        """Count pieces of each type on the board."""
        ...

    def apply_move(self, board: Board, move: Move) -> Board:
        """Apply a move to a board state."""
        ...

    def is_terminal(self, board: Board, player: Player) -> bool:
        """Check if the game is in a terminal state."""
        ...


# Factory functions for creating typed objects
def create_game_state(board: Board, current_player: Player,
                     move_count: int = 0) -> GameState:
    """Create a new game state with validation."""
    return GameState(
        board=board,
        current_player=current_player,
        move_count=move_count
    )


def create_training_sample(board_state: Board, player: Player,
                          target_value: float, augmented: bool = False) -> TrainingSample:
    """Create a training sample with optional feature pre-computation."""
    return TrainingSample(
        board_state=board_state,
        player=player,
        target_value=target_value,
        augmented=augmented
    )


def create_training_stats() -> TrainingStats:
    """Create empty training statistics."""
    return TrainingStats()


def create_cache_stats() -> CacheStats:
    """Create empty cache statistics."""
    return CacheStats()


# Utility functions for type checking
def is_valid_board(board: Any) -> bool:
    """Check if an object is a valid board representation."""
    return (isinstance(board, list) and len(board) == 33 and
            all(isinstance(x, int) for x in board))


def is_valid_move(move: Any) -> bool:
    """Check if an object is a valid move representation."""
    return (isinstance(move, list) and all(isinstance(x, int) and 1 <= x <= 32
                                          for x in move))


def is_valid_player(player: Any) -> bool:
    """Check if a value is a valid player identifier."""
    return player in [-1, 1]


# Constants for type safety
BOARD_SIZE = 33
SQUARES_COUNT = 32
MIN_SQUARE_INDEX = 1
MAX_SQUARE_INDEX = 32
VALID_PLAYERS = [-1, 1]  # Red and Black
PIECE_VALUES = [-2, -1, 0, 1, 2]  # King red, man red, empty, man black, king black

"""
Move management functionality for the Checkers GUI.
"""
from __future__ import annotations

from typing import List, Dict, Optional, Tuple, Any

from checkers.engine import legal_moves, rc, seq_to_str


def group_moves_by_start(moves: List[List[int]]) -> Dict[int, List[List[int]]]:
    """Group moves by their starting square."""
    result = {}
    for move in moves:
        result.setdefault(move[0], []).append(move)
    return result


def group_moves_by_dest(moves: List[List[int]]) -> Dict[int, List[List[int]]]:
    """Group moves by their ending square."""
    result = {}
    for move in moves:
        result.setdefault(move[-1], []).append(move)
    return result


def manhattan_distance(a: int, b: int) -> int:
    """Calculate Manhattan distance between two squares."""
    ra, ca = rc(a)
    rb, cb = rc(b)
    return abs(ra - rb) + abs(ca - cb)


class MoveManager:
    """Manages move validation, grouping, and selection."""

    def __init__(self, board: Any, current_player: int):
        self.board = board
        self.current_player = current_player
        self.legal_moves_list: List[List[int]] = []
        self.moves_by_start: Dict[int, List[List[int]]] = {}
        self._refresh_moves()

    def set_game_state(self, board: Any, current_player: int) -> None:
        """Update the game state for move calculation."""
        self.board = board
        self.current_player = current_player
        self._refresh_moves()

    def _refresh_moves(self) -> None:
        """Recalculate legal moves and groupings."""
        self.legal_moves_list = legal_moves(self.board, self.current_player)
        self.moves_by_start = group_moves_by_start(self.legal_moves_list)

    def get_legal_moves_for_square(self, square: int) -> List[List[int]]:
        """Get all legal moves starting from a specific square."""
        return self.moves_by_start.get(square, [])

    def get_destinations_for_square(self, square: int) -> Dict[int, List[List[int]]]:
        """Get move sequences grouped by destination for a square."""
        moves = self.get_legal_moves_for_square(square)
        return group_moves_by_dest(moves)

    def is_valid_move(self, move_sequence: List[int]) -> bool:
        """Check if a move sequence is valid."""
        return move_sequence in self.legal_moves_list

    def get_best_capture_sequence(self, destination_moves: List[List[int]]) -> List[int]:
        """Get the best capture sequence from multiple options for a destination."""
        if not destination_moves:
            return []

        # Prefer the longest capture; if tie, prefer the first
        return max(destination_moves, key=lambda s: (len(s), s))

    def get_ordered_moves_for_display(self) -> List[List[int]]:
        """Get moves ordered for display (captures first, then regular moves)."""
        jumps = []
        regular_moves = []

        for move in self.legal_moves_list:
            start, end = move[0], move[1]
            ra, ca = rc(start)
            rb, cb = rc(end)
            if abs(ra - rb) == 2:
                jumps.append(move)
            else:
                regular_moves.append(move)

        return jumps + regular_moves

    def get_move_display_strings(self) -> List[str]:
        """Get string representations of moves for display."""
        ordered_moves = self.get_ordered_moves_for_display()
        return [seq_to_str(move) for move in ordered_moves]

    def find_move_by_start_square(self, start_square: int, ordered_moves: List[List[int]]) -> Optional[int]:
        """Find the index of the first move starting from a specific square."""
        for i, move in enumerate(ordered_moves):
            if move[0] == start_square:
                return i
        return None

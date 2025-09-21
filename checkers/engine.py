"""
Thin wrapper re-exporting engine API from legacy module `gameotherother`.
This allows importing via `from checkers.engine import legal_moves`, etc.
"""
from __future__ import annotations

from gameotherother import (
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
    seq_to_str,
    count_pieces,
)

__all__ = [
    "initial_board",
    "legal_moves",
    "apply_move",
    "is_terminal",
    "evaluate",
    "get_engine",
    "SearchEngine",
    "rc",
    "idx_map",
    "parse_move_str",
    "seq_to_str",
    "count_pieces",
]

"""
Thin wrapper re-exporting engine API from legacy module `gameotherother`.
This allows importing via `from checkers.engine import legal_moves`, etc.
Adds a small synchronization layer for the `CAPTURES_MANDATORY` rule so the GUI
can toggle it via this module and have the underlying engine respect it.
"""
from __future__ import annotations

import gameotherother as _g

from gameotherother import (
    initial_board,
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
    "CAPTURES_MANDATORY",
    "set_captures_mandatory",
]

# Expose and keep a local copy of the captures mandatory flag.
CAPTURES_MANDATORY = getattr(_g, "CAPTURES_MANDATORY", False)


def set_captures_mandatory(value: bool) -> None:
    """Set captures mandatory rule and propagate to the underlying engine module."""
    global CAPTURES_MANDATORY
    CAPTURES_MANDATORY = bool(value)
    _g.CAPTURES_MANDATORY = CAPTURES_MANDATORY


def legal_moves(board, player):
    """
    Wrapper that ensures the underlying engine uses the current rule flag managed
    via this module, then delegates to the actual implementation.
    """
    _g.CAPTURES_MANDATORY = CAPTURES_MANDATORY
    return _g.legal_moves(board, player)

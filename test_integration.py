from __future__ import annotations

from checkers.eval import get_evaluator
from checkers.search import get_search_strategy
from gameotherother import initial_board, apply_move


def test_end_to_end_move_and_eval():
    evaluator = get_evaluator()
    strategy = get_search_strategy()

    board = initial_board()
    # Evaluate initial position
    score = evaluator.evaluate_position(board, 1)
    assert isinstance(score, float)

    # Search a move and apply it
    s, best = strategy.search(board, 1, depth=2)
    if best is not None:
        new_board = apply_move(board, best)
        assert new_board != board

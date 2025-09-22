import numpy as np

from checkers.eval import NeuralEvaluatorAdapter, get_neural_evaluator, get_evaluator
from checkers.search import AlphaBetaSearchStrategy, get_search_strategy
from gameotherother import initial_board


def test_evaluator_adapter_matches_impl():
    impl = get_neural_evaluator()
    adapter = NeuralEvaluatorAdapter(impl)

    board = initial_board()
    score_impl = float(impl.evaluate_position(board, 1))
    score_adapter = float(adapter.evaluate_position(board, 1))

    assert abs(score_impl - score_adapter) < 1e-6


def test_search_strategy_returns_move():
    strat = AlphaBetaSearchStrategy()
    board = initial_board()
    score, move = strat.search(board, 1, depth=2)
    assert isinstance(score, int)
    assert move is None or (isinstance(move, list) and len(move) >= 2)


def test_factories_work():
    evaluator = get_evaluator()
    board = initial_board()
    s = evaluator.evaluate_position(board, 1)
    assert isinstance(s, float)

    strat = get_search_strategy()
    sc, mv = strat.search(board, 1, 2)
    assert isinstance(sc, int)

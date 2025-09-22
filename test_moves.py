import pytest

from checkers.moves import MoveGenerator, MoveValidator, BoardState, rc, idx_map
from gameotherother import initial_board

# Helpers

def make_empty_board():
    return [0] * 33


def place_piece(board, idx, value):
    board[idx] = value


def find_simple_move_for_black(board):
    # Find any quiet move for a black man from the initial position
    gen = MoveGenerator(captures_mandatory=False)
    moves = gen.legal_moves(board, 1)
    # Return the first simple (length==2) move found
    for m in moves:
        if len(m) == 2:
            return m
    return None


def test_simple_moves_initial_position():
    board = initial_board()
    gen = MoveGenerator(captures_mandatory=True)
    moves = gen.legal_moves(board, 1)
    assert isinstance(moves, list)
    assert all(isinstance(m, list) and len(m) >= 2 for m in moves)
    # There should be quiet moves available from the initial position
    assert any(len(m) == 2 for m in moves)


def test_captures_mandatory_true_enforces_capture():
    # Build a small capture scenario for black (player=1)
    board = make_empty_board()

    # Choose a square with room to capture upward: find i with (r-1,c+/-1) and (r-2,c+/-2) valid
    src = None
    mid = None
    end = None
    for i in range(1, 33):
        r, c = rc(i)
        for dr, dc in [(-1, -1), (-1, 1)]:  # black man moves up
            mr, mc = r + dr, c + dc
            er, ec = r + 2 * dr, c + 2 * dc
            if 0 <= mr < 8 and 0 <= mc < 8 and (mr + mc) % 2 == 1 and (mr, mc) in idx_map:
                if 0 <= er < 8 and 0 <= ec < 8 and (er + ec) % 2 == 1 and (er, ec) in idx_map:
                    src = i
                    mid = idx_map[(mr, mc)]
                    end = idx_map[(er, ec)]
                    break
        if src:
            break

    assert src and mid and end

    # Place black man at src, red man at mid, landing empty
    place_piece(board, src, 1)
    place_piece(board, mid, -1)

    gen_cap = MoveGenerator(captures_mandatory=True)
    moves_cap = gen_cap.legal_moves(board, 1)
    # All legal moves should be captures and include [src, end]
    from checkers.moves import MoveValidator as _MV
    assert all(_MV.is_capture(board, m) for m in moves_cap)
    assert any(m[0] == src and m[-1] == end for m in moves_cap)

    # With captures_mandatory=False, quiet moves could appear if any exist, but the capture must still be present
    gen_no_cap = MoveGenerator(captures_mandatory=False)
    moves_no_cap = gen_no_cap.legal_moves(board, 1)
    assert any(m[0] == src and m[-1] == end for m in moves_no_cap)


def test_move_validator_with_generated_move():
    board = initial_board()
    mv = find_simple_move_for_black(board)
    assert mv is not None
    assert MoveValidator.validate(board, 1, mv, captures_mandatory=True)


def test_king_moves_backward():
    # Create a king that can move backward
    board = make_empty_board()
    # Pick a mid-board square with room both ways; search for a square that has a backward neighbor for red and forward neighbor for black
    src = None
    back_nb = None
    for i in range(1, 33):
        r, c = rc(i)
        # Try one backward direction for black (which is down in rows for a man, but king should move both)
        for dr, dc in [(1, -1), (1, 1), (-1, -1), (-1, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 8 and 0 <= nc < 8 and (nr + nc) % 2 == 1 and (nr, nc) in idx_map:
                nb = idx_map[(nr, nc)]
                src = i
                back_nb = nb
                break
        if src:
            break
    assert src and back_nb

    # Place a black king at src with empty neighbor
    place_piece(board, src, 2)  # 2 == black king

    gen = MoveGenerator(captures_mandatory=True)
    moves = gen.legal_moves(board, 1)
    # Expect at least one quiet move to back_nb
    assert any(m == [src, back_nb] for m in moves)

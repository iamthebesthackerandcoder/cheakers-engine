# Global type aliases
from typing import Optional, List, Tuple, Dict, Any

Board = List[int]  # Board state: index 0 is unused, 1-32 are squares
Move = List[int]   # Move sequence: [from, to, ...] for jumps
GameResult = Tuple[int, Optional[Move]]  # (score, best_move)

# Import configuration management
from config import get_game_rules

# ============================
# Board indexing and constants
# ============================
SQUARES: int = 32

# Mapping between 1..32 indices and (row, col) on 8x8 board for dark squares
_rc_of: List[Optional[Tuple[int, int]]] = [None] * (SQUARES + 1)
idx_map: Dict[Tuple[int, int], int] = {}


def _build_mappings() -> None:
    i = 1
    for r in range(8):
        for c in range(8):
            if (r + c) % 2 == 1:
                _rc_of[i] = (r, c)
                idx_map[(r, c)] = i
                i += 1


_build_mappings()


def rc(i: int) -> Tuple[int, int]:
    return _rc_of[i]  # type: ignore[index]


# Rules toggle (used by GUI)
CAPTURES_MANDATORY: bool = bool(get_game_rules().captures_mandatory)


# ============================
# Board setup and utilities
# ============================
def initial_board() -> Board:
    """Initial position: Red (-1) on rows 0..2, Black (1) on rows 5..7."""
    b: Board = [0] * (SQUARES + 1)
    for i in range(1, SQUARES + 1):
        r, c = rc(i)
        if r <= 2:
            b[i] = -1  # Red men
        elif r >= 5:
            b[i] = 1   # Black men
    return b


def count_pieces(board: Board) -> Tuple[int, int, int, int]:
    blacks = sum(1 for v in board[1:] if v > 0)
    reds = sum(1 for v in board[1:] if v < 0)
    bk = sum(1 for v in board[1:] if v == 2)
    rk = sum(1 for v in board[1:] if v == -2)
    return blacks, reds, bk, rk


def seq_to_str(seq: Move) -> str:
    if not seq or len(seq) < 2:
        return ""
    # Use 'x' if this is a jump sequence
    use_x = False
    for a, b in zip(seq, seq[1:]):
        ra, ca = rc(a)
        rb, cb = rc(b)
        if abs(ra - rb) == 2:
            use_x = True
            break
    sep = 'x' if use_x else '-'
    return sep.join(str(x) for x in seq)


def parse_move_str(s: str) -> Optional[Move]:
    s = s.strip().lower().replace('x', '-').replace(' ', '')
    if not s:
        return None
    parts = [p for p in s.split('-') if p]
    try:
        seq = [int(p) for p in parts]
    except ValueError:
        return None
    if not all(1 <= x <= 32 for x in seq):
        return None
    return seq


# ============================
# Move generation
# ============================
_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def _neighbor(idx: int, dr: int, dc: int) -> Optional[int]:
    r, c = rc(idx)
    nr, nc = r + dr, c + dc
    if 0 <= nr < 8 and 0 <= nc < 8 and (nr + nc) % 2 == 1:
        return idx_map.get((nr, nc))
    return None


def _gen_simple_moves(board: Board, idx: int, player: int) -> List[Move]:
    v = board[idx]
    is_king = abs(v) == 2
    moves: List[Move] = []
    if is_king:
        dirs = _DIRS
    else:
        dirs = [(-1, -1), (-1, 1)] if player == 1 else [(1, -1), (1, 1)]
    for dr, dc in dirs:
        nb = _neighbor(idx, dr, dc)
        if nb and board[nb] == 0:
            moves.append([idx, nb])
    return moves


def _gen_captures_from(board: Board, idx: int, player: int, visited: Optional[set] = None) -> List[Move]:
    v = board[idx]
    is_king = abs(v) == 2
    if visited is None:
        visited = set()
    sequences: List[Move] = []
    dirs = _DIRS if is_king else ([(-1, -1), (-1, 1)] if player == 1 else [(1, -1), (1, 1)])
    found_any = False
    for dr, dc in dirs:
        mid = _neighbor(idx, dr, dc)
        end = _neighbor(idx, 2 * dr, 2 * dc)
        if mid and end and board[mid] * player < 0 and board[end] == 0 and mid not in visited:
            found_any = True
            # simulate jump
            new_board = board.copy()
            new_board[end] = new_board[idx]
            new_board[idx] = 0
            new_board[mid] = 0
            new_visited = set(visited)
            new_visited.add(mid)
            tails = _gen_captures_from(new_board, end, player, new_visited)
            if tails:
                for t in tails:
                    sequences.append([idx] + t)
            else:
                sequences.append([idx, end])
    if found_any:
        return sequences
    else:
        return []


def legal_moves(board: Board, player: int) -> List[Move]:
    captures: List[Move] = []
    quiets: List[Move] = []
    for i in range(1, SQUARES + 1):
        v = board[i]
        if v * player <= 0:
            continue
        caps = _gen_captures_from(board, i, player)
        if caps:
            captures.extend(caps)
        else:
            quiets.extend(_gen_simple_moves(board, i, player))
    if captures:
        return captures if CAPTURES_MANDATORY else captures + quiets
    return quiets


# ============================
# Applying moves
# ============================
def _mid_square(a: int, b: int) -> Optional[int]:
    ra, ca = rc(a)
    rb, cb = rc(b)
    if abs(ra - rb) == 2 and abs(ca - cb) == 2:
        mr, mc = (ra + rb) // 2, (ca + cb) // 2
        return idx_map.get((mr, mc))
    return None


def apply_move(board: Board, seq: Move) -> Board:
    nb = board.copy()
    cur = seq[0]
    piece = nb[cur]
    nb[cur] = 0
    for nxt in seq[1:]:
        mid = _mid_square(cur, nxt)
        if mid is not None:
            nb[mid] = 0
        cur = nxt
    nb[cur] = piece
    # Promotion
    r, c = rc(cur)
    if nb[cur] == 1 and r == 0:
        nb[cur] = 2
    elif nb[cur] == -1 and r == 7:
        nb[cur] = -2
    return nb


def is_terminal(board: Board, player: int) -> bool:
    return len(legal_moves(board, player)) == 0


# ============================
# Evaluation and search engine
# ============================
def evaluate(board: Board, player: int) -> int:
    men_black = sum(1 for v in board[1:] if v == 1)
    men_red = sum(1 for v in board[1:] if v == -1)
    kings_black = sum(1 for v in board[1:] if v == 2)
    kings_red = sum(1 for v in board[1:] if v == -2)
    score = (men_black - men_red) * 100 + (kings_black - kings_red) * 160
    return score if player == 1 else -score


class SearchEngine:
    def __init__(self, seed: Optional[int] = None, shared_tt: Optional[Dict[str, Any]] = None) -> None:
        self.neural_evaluator: Optional[Any] = None
        self.shared_tt = shared_tt

    def _eval(self, board: Board, player: int) -> int:
        if self.neural_evaluator is not None:
            try:
                val = float(self.neural_evaluator.evaluate_position(board, player))
                return int(val)
            except Exception:
                pass
        return evaluate(board, player)

    def search(self, board: Board, player: int, depth: int) -> GameResult:
        best_move: Optional[Move] = None
        alpha, beta = -10**9, 10**9

        def ab(pos: Board, side: int, d: int, a: int, b: int) -> int:
            lm = legal_moves(pos, side)
            if d == 0 or not lm:
                if not lm:
                    # No legal moves: side to move loses
                    return -10000
                return self._eval(pos, side)
            if side == player:
                val = -10**9
                for m in lm:
                    child = apply_move(pos, m)
                    sc = ab(child, -side, d - 1, a, b)
                    if sc > val:
                        val = sc
                    if val > a:
                        a = val
                    if a >= b:
                        break
                return val
            else:
                val = 10**9
                for m in lm:
                    child = apply_move(pos, m)
                    sc = ab(child, -side, d - 1, a, b)
                    if sc < val:
                        val = sc
                    if val < b:
                        b = val
                    if a >= b:
                        break
                return val

        # Root: pick best move
        moves = legal_moves(board, player)
        if not moves:
            return (-10000, None)
        best_score = -10**9
        for m in moves:
            child = apply_move(board, m)
            sc = ab(child, -player, depth - 1, alpha, beta)
            if sc > best_score:
                best_score = sc
                best_move = m
            if best_score > alpha:
                alpha = best_score
        return (int(best_score), best_move)


def get_engine() -> SearchEngine:
    return SearchEngine()


# Backward-compat export (used in some scripts)
def minimax(board: Board, player: int, depth: int) -> GameResult:
    eng = get_engine()
    return eng.search(board, player, depth)
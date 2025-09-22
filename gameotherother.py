# Import basic typing
from typing import Optional, List, Tuple, Dict, Any, Set, Union

# Type aliases for better readability (avoiding circular imports)
Board = List[int]  # Board state: index 0 is unused, 1-32 are squares
Move = List[int]   # Move sequence: [from, to, ...] for jumps
GameResult = Tuple[int, Optional[Move]]  # (score, best_move)
Player = int  # 1 for black, -1 for red
SquareIndex = int  # 1-32 for dark squares
Position = Tuple[int, int]  # (row, col) coordinates

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
    """Build internal mappings between square indices and row/column coordinates."""
    i: int = 1
    for r in range(8):
        for c in range(8):
            if (r + c) % 2 == 1:
                _rc_of[i] = (r, c)
                idx_map[(r, c)] = i
                i += 1


_build_mappings()


def rc(i: SquareIndex) -> Position:
    """Convert square index to row/column coordinates."""
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
    """Count pieces of each type on the board.

    Returns:
        Tuple of (black_men, red_men, black_kings, red_kings)
    """
    blacks: int = sum(1 for v in board[1:] if v > 0)
    reds: int = sum(1 for v in board[1:] if v < 0)
    bk: int = sum(1 for v in board[1:] if v == 2)
    rk: int = sum(1 for v in board[1:] if v == -2)
    return blacks, reds, bk, rk


def seq_to_str(seq: Move) -> str:
    """Convert a move sequence to string notation."""
    if not seq or len(seq) < 2:
        return ""
    # Use 'x' if this is a jump sequence
    use_x: bool = False
    for a, b in zip(seq, seq[1:]):
        ra, ca = rc(a)
        rb, cb = rc(b)
        if abs(ra - rb) == 2:
            use_x = True
            break
    sep: str = 'x' if use_x else '-'
    return sep.join(str(x) for x in seq)


def parse_move_str(s: str) -> Optional[Move]:
    """Parse a move string into a move sequence."""
    s = s.strip().lower().replace('x', '-').replace(' ', '')
    if not s:
        return None
    parts: List[str] = [p for p in s.split('-') if p]
    try:
        seq: List[int] = [int(p) for p in parts]
    except ValueError:
        return None
    if not all(1 <= x <= 32 for x in seq):
        return None
    return seq


# ============================
# Move generation
# ============================
_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def _neighbor(idx: SquareIndex, dr: int, dc: int) -> Optional[SquareIndex]:
    """Get the neighboring square in a given direction."""
    r, c = rc(idx)
    nr, nc = r + dr, c + dc
    if 0 <= nr < 8 and 0 <= nc < 8 and (nr + nc) % 2 == 1:
        return idx_map.get((nr, nc))
    return None


def _gen_simple_moves(board: Board, idx: SquareIndex, player: Player) -> List[Move]:
    """Generate simple (non-capture) moves for a piece."""
    v: int = board[idx]
    is_king: bool = abs(v) == 2
    moves: List[Move] = []
    if is_king:
        dirs: List[Tuple[int, int]] = _DIRS
    else:
        dirs = [(-1, -1), (-1, 1)] if player == 1 else [(1, -1), (1, 1)]
    for dr, dc in dirs:
        nb: Optional[SquareIndex] = _neighbor(idx, dr, dc)
        if nb and board[nb] == 0:
            moves.append([idx, nb])
    return moves


def _gen_captures_from(board: Board, idx: SquareIndex, player: Player,
                      visited: Optional[Set[SquareIndex]] = None) -> List[Move]:
    """Generate capture sequences starting from a given square."""
    v: int = board[idx]
    is_king: bool = abs(v) == 2
    if visited is None:
        visited = set()
    sequences: List[Move] = []
    dirs: List[Tuple[int, int]] = _DIRS if is_king else ([(-1, -1), (-1, 1)] if player == 1 else [(1, -1), (1, 1)])
    found_any: bool = False

    # Pre-allocate board copy to avoid repeated allocations
    for dr, dc in dirs:
        mid = _neighbor(idx, dr, dc)
        end = _neighbor(idx, 2 * dr, 2 * dc)
        if mid and end and board[mid] * player < 0 and board[end] == 0 and mid not in visited:
            found_any = True
            # Use iterative approach instead of recursion for better performance
            current_sequences = [[idx, end]]
            current_visited = set(visited)
            current_visited.add(mid)

            # Continue jumping from the landing square
            current_idx = end
            while True:
                has_more_jumps = False
                for dr2, dc2 in dirs:
                    mid2 = _neighbor(current_idx, dr2, dc2)
                    end2 = _neighbor(current_idx, 2 * dr2, 2 * dc2)
                    if mid2 and end2 and board[mid2] * player < 0 and board[end2] == 0 and mid2 not in current_visited:
                        has_more_jumps = True
                        # Create new sequences with the additional jump
                        new_sequences = []
                        for seq in current_sequences:
                            new_sequences.append(seq + [end2])
                        current_sequences = new_sequences
                        current_visited.add(mid2)
                        current_idx = end2
                        break
                if not has_more_jumps:
                    break

            sequences.extend(current_sequences)

    if found_any:
        return sequences
    else:
        return []


def legal_moves(board: Board, player: Player) -> List[Move]:
    """Generate all legal moves for a player via the new MoveGenerator abstraction."""
    # Local import to avoid circular import during module initialization
    from checkers.moves import MoveGenerator

    generator = MoveGenerator(captures_mandatory=CAPTURES_MANDATORY)
    return generator.legal_moves(board, player)


# ============================
# Applying moves
# ============================
def _mid_square(a: SquareIndex, b: SquareIndex) -> Optional[SquareIndex]:
    """Get the square between two adjacent squares (for captures)."""
    ra, ca = rc(a)
    rb, cb = rc(b)
    if abs(ra - rb) == 2 and abs(ca - cb) == 2:
        mr, mc = (ra + rb) // 2, (ca + cb) // 2
        return idx_map.get((mr, mc))
    return None


def apply_move(board: Board, seq: Move) -> Board:
    """Apply a move sequence to a board state."""
    nb: Board = board.copy()
    cur: SquareIndex = seq[0]
    piece: int = nb[cur]
    nb[cur] = 0
    for nxt in seq[1:]:
        mid: Optional[SquareIndex] = _mid_square(cur, nxt)
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


def is_terminal(board: Board, player: Player) -> bool:
    """Check if the game is in a terminal state for the given player."""
    return len(legal_moves(board, player)) == 0


# ============================
# Evaluation and search engine
# ============================
def evaluate(board: Board, player: Player) -> int:
    """Evaluate a board position using piece count heuristic."""
    men_black: int = sum(1 for v in board[1:] if v == 1)
    men_red: int = sum(1 for v in board[1:] if v == -1)
    kings_black: int = sum(1 for v in board[1:] if v == 2)
    kings_red: int = sum(1 for v in board[1:] if v == -2)
    score: int = (men_black - men_red) * 100 + (kings_black - kings_red) * 160
    return score if player == 1 else -score


class SearchEngine:
    """Alpha-beta search engine with transposition table and neural evaluation."""

    def __init__(self, seed: Optional[int] = None,
                 shared_tt: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the search engine."""
        self.neural_evaluator: Optional[Any] = None
        self.tt: Dict[str, Any] = shared_tt if shared_tt is not None else {}
        # Pre-allocate TT with better initial size
        if not shared_tt:
            self.tt = {}  # Will grow as needed

    def _eval(self, board: Board, player: Player) -> int:
        """Evaluate a position with caching and neural evaluation."""
        # More efficient key generation using tuple directly
        board_tuple: Tuple[int, ...] = tuple(board)
        key: str = str(board_tuple) + f"_{player}"
        if key in self.tt:
            return int(self.tt[key])

        # Limit TT size to prevent memory issues
        if len(self.tt) > 100000:
            # Simple LRU eviction: clear 10% of entries
            items_to_remove: int = len(self.tt) // 10
            keys_to_remove: List[str] = list(self.tt.keys())[:items_to_remove]
            for k in keys_to_remove:
                del self.tt[k]

        if self.neural_evaluator is not None:
            try:
                val: float = float(self.neural_evaluator.evaluate_position(board, player))
                self.tt[key] = val
                return int(val)
            except Exception:
                pass
        return evaluate(board, player)

    def search(self, board: Board, player: Player, depth: int) -> GameResult:
        """Perform alpha-beta search to find the best move."""
        best_move: Optional[Move] = None
        alpha: int = -10**9
        beta: int = 10**9

        def ab(pos: Board, side: Player, d: int, a: int, b: int) -> int:
            """Alpha-beta search implementation."""
            lm: List[Move] = legal_moves(pos, side)
            if d == 0 or not lm:
                if not lm:
                    # No legal moves: side to move loses
                    return -10000
                return self._eval(pos, side)
            if side == player:
                val: int = -10**9
                for m in lm:
                    child: Board = apply_move(pos, m)
                    sc: int = ab(child, -side, d - 1, a, b)
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
        moves: List[Move] = legal_moves(board, player)
        if not moves:
            return (-10000, None)
        best_score: int = -10**9
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
    """Get a new search engine instance."""
    return SearchEngine()


# Backward-compat export (used in some scripts)
def minimax(board: Board, player: Player, depth: int) -> GameResult:
    """Backward compatibility wrapper for minimax search."""
    eng: SearchEngine = get_engine()
    return eng.search(board, player, depth)

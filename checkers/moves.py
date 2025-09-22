from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Set

# Basic type aliases (kept local to avoid circular imports)
Board = List[int]  # index 0 unused; squares 1..32
Move = List[int]   # move sequence: [from, to, ...]
Player = int       # 1 (black) or -1 (red)
SquareIndex = int  # 1..32
Position = Tuple[int, int]

SQUARES: int = 32

# -----------------------------
# Board indexing and utilities
# -----------------------------
_rc_of: List[Optional[Tuple[int, int]]] = [None] * (SQUARES + 1)
idx_map: Dict[Tuple[int, int], int] = {}


def _build_mappings() -> None:
    i: int = 1
    for r in range(8):
        for c in range(8):
            if (r + c) % 2 == 1:
                _rc_of[i] = (r, c)
                idx_map[(r, c)] = i
                i += 1


_build_mappings()


def rc(i: SquareIndex) -> Position:
    return _rc_of[i]  # type: ignore[index]


_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def _neighbor(idx: SquareIndex, dr: int, dc: int) -> Optional[SquareIndex]:
    r, c = rc(idx)
    nr, nc = r + dr, c + dc
    if 0 <= nr < 8 and 0 <= nc < 8 and (nr + nc) % 2 == 1:
        return idx_map.get((nr, nc))
    return None


@dataclass(frozen=True)
class BoardState:
    """Immutable representation of a board state (1..32 indexing)."""

    board: Board

    def __post_init__(self) -> None:
        if not isinstance(self.board, list) or len(self.board) != SQUARES + 1:
            raise ValueError("Board must be a list of length 33 (index 0 unused)")

    def piece_at(self, idx: SquareIndex) -> int:
        return self.board[idx]


class MoveGenerator:
    """Generates legal moves for a given board and player.

    This class encapsulates move generation and capture rules. It is deliberately
    standalone (no imports from legacy modules) to avoid circular imports.
    """

    def __init__(self, captures_mandatory: bool = True) -> None:
        self.captures_mandatory = bool(captures_mandatory)

    def _gen_simple_moves(self, board: Board, idx: SquareIndex, player: Player) -> List[Move]:
        v: int = board[idx]
        if v == 0:
            return []
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

    def _gen_captures_from(self, board: Board, idx: SquareIndex, player: Player,
                            visited: Optional[Set[SquareIndex]] = None) -> List[Move]:
        v: int = board[idx]
        if v == 0:
            return []
        is_king: bool = abs(v) == 2
        if visited is None:
            visited = set()
        sequences: List[Move] = []
        dirs: List[Tuple[int, int]] = _DIRS if is_king else ([(-1, -1), (-1, 1)] if player == 1 else [(1, -1), (1, 1)])
        found_any: bool = False

        for dr, dc in dirs:
            mid = _neighbor(idx, dr, dc)
            end = _neighbor(idx, 2 * dr, 2 * dc)
            if mid and end and board[mid] * player < 0 and board[end] == 0 and mid not in visited:
                found_any = True
                current_sequences = [[idx, end]]
                current_visited = set(visited)
                current_visited.add(mid)

                current_idx = end
                while True:
                    has_more_jumps = False
                    for dr2, dc2 in dirs:
                        mid2 = _neighbor(current_idx, dr2, dc2)
                        end2 = _neighbor(current_idx, 2 * dr2, 2 * dc2)
                        if mid2 and end2 and board[mid2] * player < 0 and board[end2] == 0 and mid2 not in current_visited:
                            has_more_jumps = True
                            new_sequences: List[Move] = []
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
        return []

    def legal_moves(self, board: Board, player: Player) -> List[Move]:
        captures: List[Move] = []
        quiets: List[Move] = []
        for i in range(1, SQUARES + 1):
            v: int = board[i]
            if v * player <= 0:
                continue
            caps = self._gen_captures_from(board, i, player)
            if caps:
                captures.extend(caps)
            else:
                quiets.extend(self._gen_simple_moves(board, i, player))
        if captures:
            return captures if self.captures_mandatory else captures + quiets
        return quiets


class MoveValidator:
    """Validates moves against generated legal moves and basic rules."""

    @staticmethod
    def is_capture(board: Board, seq: Move) -> bool:
        if len(seq) < 2:
            return False
        for a, b in zip(seq, seq[1:]):
            ra, ca = rc(a)
            rb, cb = rc(b)
            if abs(ra - rb) == 2 and abs(ca - cb) == 2:
                return True
        return False

    @staticmethod
    def validate(board: Board, player: Player, seq: Move, captures_mandatory: bool = True) -> bool:
        gen = MoveGenerator(captures_mandatory=captures_mandatory)
        legal = gen.legal_moves(board, player)
        return seq in legal


# Convenience functional API

def legal_moves(board: Board, player: Player, captures_mandatory: bool = True) -> List[Move]:
    return MoveGenerator(captures_mandatory=captures_mandatory).legal_moves(board, player)

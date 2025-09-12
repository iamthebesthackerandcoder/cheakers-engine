#!/usr/bin/env python3
# Enhanced Checkers Game with Premium CLI UI
# Improved visual design, better UX, enhanced feedback, and streamlined interaction
# Engine upgraded: fast move-gen, transposition table, PVS, quiescence, better eval
# Now supports optional neural evaluation via SearchEngine.neural_evaluator

from copy import deepcopy
import collections
import math
import sys
import re
import time
import os
import random
from typing import Optional

import numpy as np
try:
    from neural_eval import NeuralEvaluator
except ImportError:
    # Fallback if neural_eval is not available
    class _NeuralEvaluator:
        """Dummy NeuralEvaluator for type checking"""
        def evaluate_position(self, board, player):
            return 0
    NeuralEvaluator = _NeuralEvaluator  # Alias for compatibility

# ------------- Configuration -------------

CAPTURES_MANDATORY = False  # runtime-togglable

# Enhanced UI preferences with better defaults
UI = {
    "use_color": True,
    "use_unicode": True,
    "show_indices": True,
    "compact": False,
    "animations": True,
    "show_borders": True,
    "highlight_moves": True,
}

# Smart terminal detection with fallbacks
if not sys.stdout.isatty():
    UI["use_color"] = False
    UI["animations"] = False

# ------------- Engine core + fast move gen -------------

SQUARES = 32

rc_of: list[tuple[int, int]] = [(-1, -1)] * (SQUARES + 1)  # Initialize with dummy values
idx_map = {}

def build_mappings():
    i = 1
    for r in range(8):
        for c in range(8):
            if (r + c) % 2 == 1:
                rc_of[i] = (r, c)
                idx_map[(r, c)] = i
                i += 1
    # Set index 0 to a dummy value (not a valid square)
    rc_of[0] = (-1, -1)

build_mappings()

def rc(i) -> tuple[int, int]:
    return rc_of[i]

# Direction indices: 0: NW(-1,-1), 1: NE(-1, 1), 2: SW(1,-1), 3: SE(1,1)
DIAGS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

def is_valid_rc(r, c):
    return 0 <= r < 8 and 0 <= c < 8 and ((r + c) % 2 == 1)

# Precomputed neighbors and jump maps for speed
NEI = [[0]*4 for _ in range(SQUARES + 1)]      # immediate diagonal neighbors
JUMP_MID = [[0]*4 for _ in range(SQUARES + 1)] # jumped-over index
JUMP_DST = [[0]*4 for _ in range(SQUARES + 1)] # landing square after jump

def build_neighbors():
    for i in range(1, SQUARES + 1):
        r, c = rc(i)
        for d, (dr, dc) in enumerate(DIAGS):
            nr, nc = r + dr, c + dc
            if is_valid_rc(nr, nc):
                NEI[i][d] = idx_map[(nr, nc)]
            midr, midc = r + dr, c + dc
            tor, toc = r + 2*dr, c + 2*dc
            if is_valid_rc(midr, midc) and is_valid_rc(tor, toc):
                JUMP_MID[i][d] = idx_map[(midr, midc)]
                JUMP_DST[i][d] = idx_map[(tor, toc)]

build_neighbors()

def initial_board():
    b = [0] * (SQUARES + 1)
    for i in range(1, SQUARES + 1):
        r, c = rc(i)
        if r <= 2:
            b[i] = -1
        elif r >= 5:
            b[i] = 1
    return b

def allowed_dirs(piece):
    if abs(piece) == 2:
        return DIAGS
    elif piece == 1:
        return [(-1, -1), (-1, 1)]
    else:
        return [(1, -1), (1, 1)]

def dir_indices_for(piece):
    # Return indices into DIAGS/NEI/JUMP_* arrays
    if abs(piece) == 2:
        return (0, 1, 2, 3)
    elif piece == 1:
        return (0, 1)
    else:
        return (2, 3)

def simple_moves_from(b, pos):
    # Fast reimplementation using precomputed NEI
    moves = []
    piece = b[pos]
    if piece == 0: return moves
    for d in dir_indices_for(piece):
        to = NEI[pos][d]
        if to and b[to] == 0:
            moves.append([pos, to])
    return moves

def jumps_from(b, pos):
    # Iterative in-place DFS for multi-jumps without recursion
    results = []
    piece = b[pos]
    if piece == 0:
        return results

    directions = dir_indices_for(piece)
    num_dirs = len(directions)

    # Stack: (cur_pos, seq_start_idx, changed_start_idx, d_idx)
    # seq and changed are global lists, append/pop for backtrack
    seq = [pos]
    changed = []
    stack = [(pos, 0, 0, 0)]  # start with d_idx=0

    while stack:
        cur_pos, seq_start, changed_start, d_idx = stack[-1]
        p = piece  # fixed

        if d_idx == num_dirs:
            # All directions tried, check if leaf
            stack.pop()
            if len(seq) > 1:
                results.append(seq[:])  # full sequence copy
            # Backtrack changes from changed_start
            for i in range(len(changed) - 1, changed_start - 1, -1):
                idx, old_val = changed[i]
                b[idx] = old_val
            changed = changed[:changed_start]
            seq = seq[:seq_start]
            continue

        # Try current direction
        d = directions[d_idx]
        mid = JUMP_MID[cur_pos][d]
        to = JUMP_DST[cur_pos][d]
        can_jump = mid and to and b[mid] != 0 and (b[mid] * p < 0) and b[to] == 0

        if can_jump:
            # Save state
            old_cur = b[cur_pos]
            old_mid = b[mid]
            old_to = b[to]

            # Do jump
            b[cur_pos] = 0
            b[mid] = 0
            b[to] = p

            # Append to seq and changed
            seq.append(to)
            changed.append((cur_pos, old_cur))
            changed.append((mid, old_mid))
            changed.append((to, old_to))

            # Increment d_idx for current, push new branch
            stack[-1] = (cur_pos, seq_start, changed_start, d_idx + 1)
            # New state for branch
            stack.append((to, len(seq) - 1, len(changed) - 3, 0))
        else:
            # No jump, next direction
            stack[-1] = (cur_pos, seq_start, changed_start, d_idx + 1)

    return results

def legal_moves(b, player):
    # Optimized: generate jumps first; if none or not mandatory, generate simples
    all_jumps = []
    for pos in range(1, SQUARES + 1):
        if b[pos] != 0 and (b[pos] * player > 0):
            js = jumps_from(b, pos)
            if js:
                all_jumps.extend(js)
    all_jumps = [m for m in all_jumps if len(m) >= 2 and b[m[0]] * player > 0]
    if CAPTURES_MANDATORY:
        if all_jumps:
            return all_jumps
        # else fall back to simple moves only
        all_simples = []
        for pos in range(1, SQUARES + 1):
            if b[pos] != 0 and (b[pos] * player > 0):
                sm = simple_moves_from(b, pos)
                if sm:
                    all_simples.extend(sm)
        return all_simples
    else:
        # optional captures: return simple + jumps
        all_simples = []
        for pos in range(1, SQUARES + 1):
            if b[pos] != 0 and (b[pos] * player > 0):
                sm = simple_moves_from(b, pos)
                if sm:
                    all_simples.extend(sm)
        return all_simples + all_jumps

def apply_move(b, seq):
    # UI-friendly copying apply (kept for compatibility)
    newb = b.copy()
    frm = seq[0]
    to = seq[-1]
    piece = newb[frm]
    newb[frm] = 0
    if len(seq) >= 2:
        for a, c in zip(seq, seq[1:]):
            ra, ca = rc(a)
            rb, cb = rc(c)
            if abs(ra - rb) == 2:
                midr = (ra + rb) // 2
                midc = (ca + cb) // 2
                midi = idx_map[(midr, midc)]
                newb[midi] = 0
    newb[to] = piece
    tr, tc = rc(to)
    if piece == 1 and tr == 0:
        newb[to] = 2
    elif piece == -1 and tr == 7:
        newb[to] = -2
    return newb

# ------------- Improved evaluation -------------

# Piece values
MAN_VALUE = 100
KING_VALUE = 180

# Precompute piece-square tables (center control + advancement)
PSQ_MAN_BLACK = [0] * (SQUARES + 1)
PSQ_MAN_RED = [0] * (SQUARES + 1)
PSQ_KING = [0] * (SQUARES + 1)

for i in range(1, SQUARES + 1):
    r, c = rc(i)
    # Center bonus: Manhattan distance from center ~3.5
    center_bonus = int(3 - (abs(r - 3.5) + abs(c - 3.5)))
    if center_bonus < 0:
        center_bonus = 0
    adv_black = (7 - r)  # higher when closer to crown row (0)
    adv_red = r          # higher when closer to crown row (7)
    PSQ_MAN_BLACK[i] = 2 * adv_black + 3 * center_bonus
    PSQ_MAN_RED[i] = 2 * adv_red + 3 * center_bonus
    PSQ_KING[i] = 4 * center_bonus

def fast_pseudo_mobility(b):
    # Very cheap directional mobility (ignores captures)
    black = 0
    red = 0
    for pos in range(1, SQUARES + 1):
        p = b[pos]
        if p == 0:
            continue
        if p > 0:
            for d in dir_indices_for(p):
                to = NEI[pos][d]
                if to and b[to] == 0:
                    black += 1
        else:
            for d in dir_indices_for(p):
                to = NEI[pos][d]
                if to and b[to] == 0:
                    red += 1
    return black - red

def evaluate(b, player):
    # Material + piece-square + mobility + phase scaling
    score = 0
    black_cnt = red_cnt = bk_cnt = rk_cnt = 0
    for i in range(1, SQUARES + 1):
        v = b[i]
        if v == 1:
            score += MAN_VALUE + PSQ_MAN_BLACK[i]
            black_cnt += 1
        elif v == 2:
            score += KING_VALUE + PSQ_KING[i]
            bk_cnt += 1
            black_cnt += 1
        elif v == -1:
            score -= MAN_VALUE + PSQ_MAN_RED[i]
            red_cnt += 1
        elif v == -2:
            score -= KING_VALUE + PSQ_KING[i]
            rk_cnt += 1
            red_cnt += 1

    total_pieces = black_cnt + red_cnt
    # Endgame scaling (kings matter more, mobility matters less)
    if total_pieces <= 8:
        # Boost kings in endgame, small boost to men
        score += (bk_cnt - rk_cnt) * 40 + ((black_cnt - bk_cnt) - (red_cnt - rk_cnt)) * 10
        mobility_weight = 2
    else:
        mobility_weight = 4

    # Light mobility term (very cheap)
    score += mobility_weight * fast_pseudo_mobility(b)

    return player * score

def is_terminal(b, player):
    moves = legal_moves(b, player)
    if not moves:
        return True
    has_my = any(x != 0 and (x * player > 0) for x in b[1:])
    if not has_my:
        return True
    return False

# ------------- Strong search engine (TT + PVS + QSearch) -------------

INF = 1_000_000
MATE_VALUE = 500_000

def is_capture_move(seq):
    if len(seq) < 2:
        return False
    ra, ca = rc(seq[0]); rb, cb = rc(seq[1])
    return abs(ra - rb) == 2

def capture_count(seq):
    cnt = 0
    for a, c in zip(seq, seq[1:]):
        ra, ca = rc(a); rb, cb = rc(c)
        if abs(ra - rb) == 2:
            cnt += 1
    return cnt

def will_promote(b, seq):
    frm = seq[0]
    piece = b[frm]
    to = seq[-1]
    tr, _ = rc(to)
    if piece == 1 and tr == 0:
        return True
    if piece == -1 and tr == 7:
        return True
    return False

def piece_index(v):
    # Map piece to zobrist index
    # 1:0, 2:1, -1:2, -2:3
    if v == 1: return 0
    if v == 2: return 1
    if v == -1: return 2
    if v == -2: return 3
    raise ValueError("Invalid piece for hashing")

class SearchEngine:
    def __init__(self, seed=2025, shared_tt=None, batch_size=32):
        self.rand = random.Random(seed)
        self.batch_size = batch_size
        # Zobrist keys
        self.z_piece = [[self.rand.getrandbits(64) for _ in range(4)] for _ in range(SQUARES + 1)]
        self.z_side = self.rand.getrandbits(64)
        if shared_tt is not None:
            self.tt = shared_tt  # Use shared dict, no ordering
            self.is_shared_tt = True
            self.tt_max_size = 100000
        else:
            self.tt = collections.OrderedDict()
            self.is_shared_tt = False
            self.tt_max_size = 100000
        self.max_ply = 0
        self.nodes = 0
        self.killers = []  # [ [m1, m2], ... ]
        # history[(side, frm, to)] = score
        self.history = {}
        self.board: list[int] = []  # Initialize as empty list, will be set in reset_search
        self.hash = 0
        self.side = 1
        self.root_best_move = None
        self.total_pieces = 0
        self.move_cache = collections.OrderedDict()
        self.tt_hits = 0
        self.tt_misses = 0
        self.leaf_buffer = []

        # Optional neural evaluator
        self.neural_evaluator: Optional[NeuralEvaluator] = None

    def compute_hash(self, b, side):
        h = 0
        for sq in range(1, SQUARES + 1):
            v = b[sq]
            if v != 0:
                h ^= self.z_piece[sq][piece_index(v)]
        if side == 1:
            h ^= self.z_side
        return h

    def _get_tt_key(self, board, side):
        return hash(tuple(board) + (side,))

    def reset_search(self, b, side, depth):
        self.board = b.copy()
        self.side = side
        self.hash = self.compute_hash(self.board, side)
        self.nodes = 0
        self.max_ply = depth + 64
        self.killers = [[None, None] for _ in range(self.max_ply)]
        self.total_pieces = sum(1 for x in self.board[1:] if x != 0)
        self.tt_hits = 0
        self.tt_misses = 0
        self.leaf_buffer = []
        # Keep TT and history across moves
        # If shared TT, no clear needed as shared
        if not self.is_shared_tt:
            # For local, could clear if desired, but keep for now
            pass

    def _eval(self, b, side):
        """Internal evaluation using batch_predict if available for consistency."""
        if self.neural_evaluator is not None:
            try:
                return int(self.neural_evaluator.batch_predict(np.array([b]), np.array([side]))[0])
            except Exception:
                pass
        return evaluate(b, side)
    
    def batch_eval(self, boards, sides):
        """Batch evaluation for multiple positions, if neural evaluator supports it."""
        if self.neural_evaluator is not None and hasattr(self.neural_evaluator, 'batch_predict'):
            try:
                scores = self.neural_evaluator.batch_predict(boards, sides)
                return [int(s) for s in scores]
            except Exception:
                pass
        # Fallback to individual evals
        return [self._eval(b, s) for b, s in zip(boards, sides)]

    def do_move(self, seq):
        # Apply seq in-place, return undo info
        frm = seq[0]
        to = seq[-1]
        piece = self.board[frm]
        # Zobrist: remove moving piece from frm
        self.hash ^= self.z_piece[frm][piece_index(piece)]
        self.board[frm] = 0
        captured = []
        cur = frm
        # perform hops/steps
        for nxt in seq[1:]:
            ra, ca = rc(cur); rb, cb = rc(nxt)
            if abs(ra - rb) == 2:
                midr = (ra + rb) // 2
                midc = (ca + cb) // 2
                midi = idx_map[(midr, midc)]
                cap_piece = self.board[midi]
                if cap_piece != 0:
                    # Zobrist: remove captured
                    self.hash ^= self.z_piece[midi][piece_index(cap_piece)]
                self.board[midi] = 0
                captured.append((midi, cap_piece))
            cur = nxt
        # place piece at destination
        self.board[to] = piece
        self.hash ^= self.z_piece[to][piece_index(piece)]
        promoted = 0
        tr, _ = rc(to)
        if piece == 1 and tr == 0:
            # promote to king
            self.hash ^= self.z_piece[to][piece_index(piece)]
            piece = 2
            self.board[to] = piece
            self.hash ^= self.z_piece[to][piece_index(piece)]
            promoted = 1
        elif piece == -1 and tr == 7:
            self.hash ^= self.z_piece[to][piece_index(piece)]
            piece = -2
            self.board[to] = piece
            self.hash ^= self.z_piece[to][piece_index(piece)]
            promoted = -1
        # switch side
        self.side = -self.side
        self.hash ^= self.z_side
        return (seq, frm, to, piece, promoted, captured)

    def undo_move(self, undo):
        seq, frm, to, piece_after, promoted, captured = undo
        # switch side back
        self.hash ^= self.z_side
        self.side = -self.side
        # undo promotion if any
        if promoted != 0:
            # remove king at 'to'
            self.hash ^= self.z_piece[to][piece_index(piece_after)]
            # restore man
            orig_piece = 1 if promoted == 1 else -1
            self.board[to] = orig_piece
            self.hash ^= self.z_piece[to][piece_index(orig_piece)]
            piece = orig_piece
        else:
            piece = piece_after
        # remove piece from 'to'
        self.hash ^= self.z_piece[to][piece_index(piece)]
        self.board[to] = 0
        # restore captured pieces
        for midi, cap_piece in captured:
            if cap_piece != 0:
                self.board[midi] = cap_piece
                self.hash ^= self.z_piece[midi][piece_index(cap_piece)]
        # restore moving piece at 'frm'
        self.board[frm] = piece
        self.hash ^= self.z_piece[frm][piece_index(piece)]

    def gen_moves(self, side):
        key = (self.hash, side)
        if key in self.move_cache:
            self.move_cache.move_to_end(key)
            return self.move_cache[key]
        moves = legal_moves(self.board, side)
        self.move_cache[key] = moves
        if len(self.move_cache) > 5000:
            self.move_cache.popitem(last=False)
        return moves

    def gen_captures_only(self, side):
        # Generate only capture sequences
        res = []
        for pos in range(1, SQUARES + 1):
            v = self.board[pos]
            if v != 0 and v * side > 0:
                js = jumps_from(self.board, pos)
                if js:
                    res.extend(js)
        return [m for m in res if len(m) >= 2 and self.board[m[0]] * side > 0]

    def quiescence(self, alpha, beta, ply):
        self.nodes += 1
        stand_pat = self._eval(self.board, self.side)
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat

        # Only consider capture sequences for quiescence
        moves = self.gen_captures_only(self.side)
        moves = [m for m in moves if len(m) >= 2 and self.board[m[0]] * self.side > 0]
        if not moves:
            return stand_pat

        # Simple ordering: longer captures first, promotion bonus
        def mscore(m):
            sc = 100 * capture_count(m)
            if will_promote(self.board, m):
                sc += 50
            return -sc

        moves.sort(key=mscore)
        for m in moves:
            undo = self.do_move(m)
            score = -self.quiescence(-beta, -alpha, ply + 1)
            self.undo_move(undo)
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha

    def score_move(self, m, tt_move, ply, side):
        # Move ordering heuristic
        # Highest priority: PV/TT move
        if tt_move is not None and m == tt_move:
            return 1_000_000
        sc = 0
        cap = is_capture_move(m)
        if cap:
            sc += 100_000 + 1_000 * capture_count(m)
        else:
            k1, k2 = self.killers[ply]
            if m == k1:
                sc += 90_000
            elif m == k2:
                sc += 80_000
            # history heuristic
            frm = m[0]; to = m[-1]
            sc += self.history.get((side, frm, to), 0)
        if will_promote(self.board, m):
            sc += 2_000
        return sc

    def negamax(self, depth, ply, alpha, beta, allow_pvs):
        self.nodes += 1

        # TT probe
        tt_key = self._get_tt_key(self.board, self.side) if self.is_shared_tt else self.hash
        entry = self.tt.get(tt_key)
        if entry is not None:
            self.tt_hits += 1
        else:
            self.tt_misses += 1
        tt_move = None
        if entry is not None:
            if self.is_shared_tt:
                tt_depth, tt_value, tt_flag = entry
                tt_best = None
            else:
                tt_depth, tt_value, tt_flag, tt_best = entry
                if tt_best is not None:
                    tt_move = tt_best
            if tt_depth >= depth:
                if tt_flag == "EXACT":
                    return tt_value
                elif tt_flag == "LOWER" and tt_value > alpha:
                    alpha = tt_value
                elif tt_flag == "UPPER" and tt_value < beta:
                    beta = tt_value
                if alpha >= beta:
                    return tt_value
            # LRU only if not shared
            if not self.is_shared_tt:
                if hasattr(self.tt, 'move_to_end'):
                    self.tt.move_to_end(self.hash)
        else:
            self.tt_misses += 1

        # Null-move pruning
        if depth >= 3 and self.total_pieces > 10:
            self.side = -self.side
            self.hash ^= self.z_side
            null_score = -self.negamax(depth - 3, ply + 1, -beta, -beta + 1, False)
            self.side = -self.side
            self.hash ^= self.z_side
            if null_score >= beta:
                # Store cutoff in TT
                tt_key = self._get_tt_key(self.board, self.side) if self.is_shared_tt else self.hash
                if self.is_shared_tt:
                    self.tt[tt_key] = (depth, beta, "LOWER")
                else:
                    self.tt[tt_key] = (depth, beta, "LOWER", None)
                self._limit_tt_size()
                return beta

        if depth == 0:
            self.leaf_buffer.append((self.board.copy(), self.side))
            if len(self.leaf_buffer) < 2:
                print("Fallback to single eval")
                return self.quiescence(alpha, beta, ply)
            else:
                batch_size_actual = min(self.batch_size, len(self.leaf_buffer))
                batch_boards = [p[0] for p in self.leaf_buffer[-batch_size_actual:]]
                batch_sides = [p[1] for p in self.leaf_buffer[-batch_size_actual:]]
                batch_values = self.batch_eval(np.array(batch_boards), np.array(batch_sides))
                score = batch_values[-1]  # for current leaf
                del self.leaf_buffer[-batch_size_actual:]
                print(f"Used batch size {batch_size_actual}")
                return score

        moves = self.gen_moves(self.side)
        if not moves:
            # No legal moves: loss
            return -MATE_VALUE + ply

        # Move ordering
        scores = []
        for m in moves:
            scores.append((self.score_move(m, tt_move, ply, self.side), m))
        scores.sort(reverse=True, key=lambda x: x[0])
        best_val = -INF
        best_move = None

        first = True
        orig_alpha = alpha

        for _, m in scores:
            undo = self.do_move(m)
            if first:
                score = -self.negamax(depth - 1, ply + 1, -beta, -alpha, allow_pvs)
                first = False
            else:
                # PVS search
                if allow_pvs:
                    score = -self.negamax(depth - 1, ply + 1, -(alpha + 1), -alpha, False)
                    if alpha < score < beta:
                        score = -self.negamax(depth - 1, ply + 1, -beta, -alpha, True)
                else:
                    score = -self.negamax(depth - 1, ply + 1, -beta, -alpha, False)
            self.undo_move(undo)

            if score > best_val:
                best_val = score
                best_move = m
            if best_val > alpha:
                alpha = best_val
                if ply == 0:
                    self.root_best_move = best_move
            if alpha >= beta:
                # store killers/history for non-captures
                if not is_capture_move(m):
                    k1, k2 = self.killers[ply]
                    if m != k1:
                        self.killers[ply][1] = k1
                        self.killers[ply][0] = m
                    frm = m[0]; to = m[-1]
                    self.history[(self.side, frm, to)] = self.history.get((self.side, frm, to), 0) + depth * depth
                # TT store as lower bound (beta cutoff)
                tt_key = self._get_tt_key(self.board, self.side) if self.is_shared_tt else self.hash
                if self.is_shared_tt:
                    self.tt[tt_key] = (depth, best_val, "LOWER")
                else:
                    self.tt[tt_key] = (depth, best_val, "LOWER", best_move)
                self._limit_tt_size()
                return best_val

        # Store in TT
        tt_key = self._get_tt_key(self.board, self.side) if self.is_shared_tt else self.hash
        flag = "EXACT" if best_val > orig_alpha and best_val < beta else ("UPPER" if best_val <= orig_alpha else "LOWER")
        if self.is_shared_tt:
            self.tt[tt_key] = (depth, best_val, flag)
        else:
            self.tt[tt_key] = (depth, best_val, flag, best_move)
        self._limit_tt_size()
        return best_val

    def _limit_tt_size(self):
        """Limit TT size, handle shared vs local."""
        if len(self.tt) > self.tt_max_size:
            if self.is_shared_tt:
                # Simple: remove oldest by key (assume keys are hashes, remove first few)
                keys = list(self.tt.keys())
                for _ in range(len(keys) // 4):  # Remove 25%
                    if keys:
                        del self.tt[keys.pop(0)]
            else:
                self.tt.popitem(last=False)

    def search(self, b, player, depth):
        self.reset_search(b, player, depth)
        best_val = -INF
        self.root_best_move = None
        # Iterative deepening with aspiration windows
        prev_val = -INF
        aspiration_margin = 100
        for d in range(1, depth + 1):
            alpha = max(-INF, prev_val - aspiration_margin)
            beta = min(INF, prev_val + aspiration_margin)
            val = self.negamax(d, 0, alpha, beta, True)
            # If outside window, re-search with full window
            if val <= alpha or val >= beta:
                val = self.negamax(d, 0, -INF, INF, True)
            prev_val = val
            best_val = val
            # Keep principal variation move cached in TT; root_best_move updated inside negamax
            if self.root_best_move is None:
                # Fallback: pick any legal move if needed
                ms = self.gen_moves(self.side)
                self.root_best_move = ms[0] if ms else None
        hit_rate = self.tt_hits / max(1, self.tt_hits + self.tt_misses) * 100 if self.tt_hits + self.tt_misses > 0 else 0
        print(f"Search completed: TT hits {self.tt_hits}/{self.tt_hits + self.tt_misses} ({hit_rate:.1f}%), TT size {len(self.tt)}")
        return int(best_val), self.root_best_move

# Backwards-compatible wrapper
_ENGINE_SINGLETON = None
def get_engine(shared_tt=None):
    global _ENGINE_SINGLETON
    if _ENGINE_SINGLETON is None:
        _ENGINE_SINGLETON = SearchEngine(shared_tt=shared_tt)
    else:
        # If already created, can't change shared_tt easily
        pass
    return _ENGINE_SINGLETON

def minimax(b, depth, alpha, beta, player):
    # Delegate to stronger engine; alpha/beta ignored (kept for API compatibility)
    engine = get_engine()
    return engine.search(b, player, depth)

_move_re = re.compile(r"[^\d]+")

def parse_move_str(s):
    parts = re.split(_move_re, s.strip())
    parts = [p for p in parts if p]
    try:
        seq = [int(p) for p in parts]
        return seq
    except ValueError:
        return None

# ------------- Enhanced UI System -------------

class Colors:
    """Enhanced color palette with semantic meaning"""
    RESET = "\033[0m" if UI["use_color"] else ""
    
    # Base colors
    BLACK = "\033[30m" if UI["use_color"] else ""
    RED = "\033[31m" if UI["use_color"] else ""
    GREEN = "\033[32m" if UI["use_color"] else ""
    YELLOW = "\033[33m" if UI["use_color"] else ""
    BLUE = "\033[34m" if UI["use_color"] else ""
    MAGENTA = "\033[35m" if UI["use_color"] else ""
    CYAN = "\033[36m" if UI["use_color"] else ""
    WHITE = "\033[37m" if UI["use_color"] else ""
    
    # Bright colors
    BRIGHT_BLACK = "\033[90m" if UI["use_color"] else ""
    BRIGHT_RED = "\033[91m" if UI["use_color"] else ""
    BRIGHT_GREEN = "\033[92m" if UI["use_color"] else ""
    BRIGHT_YELLOW = "\033[93m" if UI["use_color"] else ""
    BRIGHT_BLUE = "\033[94m" if UI["use_color"] else ""
    BRIGHT_MAGENTA = "\033[95m" if UI["use_color"] else ""
    BRIGHT_CYAN = "\033[96m" if UI["use_color"] else ""
    BRIGHT_WHITE = "\033[97m" if UI["use_color"] else ""
    
    # Background colors
    BG_YELLOW = "\033[43m" if UI["use_color"] else ""
    BG_GREEN = "\033[42m" if UI["use_color"] else ""
    BG_RED = "\033[41m" if UI["use_color"] else ""
    BG_BLUE = "\033[44m" if UI["use_color"] else ""
    
    # Semantic colors
    SUCCESS = BRIGHT_GREEN
    WARNING = BRIGHT_YELLOW
    ERROR = BRIGHT_RED
    INFO = BRIGHT_CYAN
    MUTED = BRIGHT_BLACK
    HIGHLIGHT = BG_YELLOW + BLACK
    PLAYER = BRIGHT_WHITE
    ENGINE = BRIGHT_RED
    ACCENT = BRIGHT_MAGENTA

def clear_screen():
    """Clear terminal screen"""
    if UI["use_color"]:
        os.system('cls' if os.name == 'nt' else 'clear')

def print_banner():
    """Display enhanced game banner"""
    if not UI["use_color"]:
        print("=" * 60)
        print("CHECKERS - Premium Edition")
        print("=" * 60)
        return
    
    banner = f"""
{Colors.BRIGHT_CYAN}╔══════════════════════════════════════════════════════════╗
║                    {Colors.BRIGHT_WHITE}⚫ CHECKERS ⚫{Colors.BRIGHT_CYAN}                    ║
║                  {Colors.MUTED}Premium CLI Edition{Colors.BRIGHT_CYAN}                   ║
╚══════════════════════════════════════════════════════════╝{Colors.RESET}
    """
    print(banner)

def print_separator(char="─", width=60, color=Colors.MUTED):
    """Print a visual separator"""
    print(f"{color}{char * width}{Colors.RESET}")

def print_status_box(title, content, color=Colors.INFO):
    """Print content in a styled box"""
    if not UI["use_color"]:
        print(f"[{title}] {content}")
        return
    
    lines = content.split('\n') if isinstance(content, str) else [str(content)]
    max_len = max(len(line) for line in [title] + lines)
    width = min(max_len + 4, 60)
    
    print(f"{color}┌{'─' * (width-2)}┐{Colors.RESET}")
    print(f"{color}│ {title.ljust(width-4)} │{Colors.RESET}")
    print(f"{color}├{'─' * (width-2)}┤{Colors.RESET}")
    for line in lines:
        print(f"{color}│ {line.ljust(width-4)} │{Colors.RESET}")
    print(f"{color}└{'─' * (width-2)}┘{Colors.RESET}")

def style_text(text, color=Colors.RESET, bold=False):
    """Apply styling to text"""
    if not UI["use_color"]:
        return text
    style = "\033[1m" if bold else ""
    return f"{style}{color}{text}{Colors.RESET}"

def piece_display(v):
    """Enhanced piece display with better Unicode characters"""
    if UI["use_unicode"]:
        if v == 1:   return "●"    # Black man
        if v == 2:   return "♛"    # Black king
        if v == -1:  return "○"    # Red man  
        if v == -2:  return "♕"    # Red king
    else:
        if v == 1:   return "b"
        if v == 2:   return "B"
        if v == -1:  return "r"
        if v == -2:  return "R"
    return " "

def render_board_enhanced(b, last_move=None, legal_moves_list=None):
    """Enhanced board rendering with better visual design"""
    path = set(last_move or [])
    legal_targets = set()
    if legal_moves_list and UI["highlight_moves"]:
        for move in legal_moves_list:
            legal_targets.update(move)
    
    # Header with column labels
    col_header = "    " + "".join(f" {chr(ord('a') + i)} " for i in range(8))
    print(f"{Colors.MUTED}{col_header}{Colors.RESET}")
    
    if UI["show_borders"]:
        print(f"{Colors.MUTED}  ┌{'───' * 8}┐{Colors.RESET}")
    
    for r in range(8):
        row_display = []
        
        # Row number
        row_num = f"{8-r}"
        if UI["show_borders"]:
            row_display.append(f"{Colors.MUTED}{row_num} │{Colors.RESET}")
        else:
            row_display.append(f"{Colors.MUTED}{row_num} {Colors.RESET}")
        
        for c in range(8):
            on_dark = ((r + c) % 2 == 1)
            cell_content = ""
            
            if on_dark:
                idx = idx_map[(r, c)]
                v = b[idx]
                is_highlighted = idx in path
                is_legal_target = idx in legal_targets
                
                if v == 0:
                    # Empty square
                    if UI["show_indices"]:
                        num_text = f"{idx:2d}"
                        if is_legal_target:
                            cell_content = f"{Colors.BG_GREEN}{Colors.BLACK} {num_text[0]} {Colors.RESET}"
                        else:
                            cell_content = f"{Colors.MUTED} {num_text[0]} {Colors.RESET}"
                    else:
                        if is_legal_target:
                            cell_content = f"{Colors.BG_GREEN}   {Colors.RESET}"
                        else:
                            cell_content = "   "
                else:
                    # Piece on square
                    piece_char = piece_display(v)
                    
                    if v > 0:  # Black pieces
                        if abs(v) == 2:  # King
                            piece_text = f"{Colors.PLAYER} {piece_char} {Colors.RESET}"
                        else:  # Regular piece
                            piece_text = f"{Colors.BRIGHT_WHITE} {piece_char} {Colors.RESET}"
                    else:  # Red pieces  
                        if abs(v) == 2:  # King
                            piece_text = f"{Colors.ENGINE} {piece_char} {Colors.RESET}"
                        else:  # Regular piece
                            piece_text = f"{Colors.BRIGHT_RED} {piece_char} {Colors.RESET}"
                    
                    if is_highlighted:
                        cell_content = f"{Colors.HIGHLIGHT}{piece_text.strip()}{Colors.RESET}"
                    else:
                        cell_content = piece_text
            else:
                # Light square
                filler = "·" if UI["use_unicode"] else "."
                cell_content = f"{Colors.MUTED} {filler} {Colors.RESET}"
            
            row_display.append(cell_content)
        
        # Row number on right side
        if UI["show_borders"]:
            row_display.append(f"{Colors.MUTED}│ {row_num}{Colors.RESET}")
        else:
            row_display.append(f"{Colors.MUTED} {row_num}{Colors.RESET}")
        
        print("".join(row_display))
    
    if UI["show_borders"]:
        print(f"{Colors.MUTED}  └{'───' * 8}┘{Colors.RESET}")
    
    # Footer with column labels
    print(f"{Colors.MUTED}{col_header}{Colors.RESET}")

def print_game_status(move_num, player, b, captures_mandatory):
    """Print enhanced game status information"""
    blacks, reds, bk, rk = count_pieces(b)
    
    player_name = style_text("BLACK", Colors.PLAYER, bold=True) if player == 1 else style_text("RED", Colors.ENGINE, bold=True)
    
    status_lines = [
        f"Move: {style_text(str(move_num), Colors.ACCENT, bold=True)}",
        f"Turn: {player_name}",
        f"Pieces → Black: {blacks} (♛{bk})  Red: {reds} (♕{rk})"
    ]
    
    if captures_mandatory:
        status_lines.append(style_text("⚠ Captures are MANDATORY", Colors.WARNING, bold=True))
    
    print_status_box("GAME STATUS", "\n".join(status_lines))

def print_moves_enhanced(moves):
    """Display legal moves in an enhanced format"""
    if not moves:
        print(style_text("No legal moves available", Colors.ERROR))
        return
    
    print(style_text("LEGAL MOVES:", Colors.SUCCESS, bold=True))
    
    # Separate jumps and regular moves
    jumps = [m for m in moves if len(m) > 2 or (len(m) == 2 and abs(rc(m[0])[0] - rc(m[1])[0]) == 2)]
    regulars = [m for m in moves if m not in jumps]
    
    if jumps:
        print(style_text("  Captures:", Colors.WARNING, bold=True))
        for i, m in enumerate(jumps, 1):
            move_str = seq_to_str(m)
            print(f"    {style_text(f'{i:2d})', Colors.ACCENT)} {style_text(move_str, Colors.WARNING)}")
    
    if regulars:
        offset = len(jumps)
        if jumps:
            print(style_text("  Regular moves:", Colors.INFO, bold=True))
        for i, m in enumerate(regulars, offset + 1):
            move_str = seq_to_str(m)
            print(f"    {style_text(f'{i:2d})', Colors.ACCENT)} {style_text(move_str, Colors.INFO)}")

def seq_to_str(seq):
    if not seq: return ""
    parts = []
    for a, b_ in zip(seq, seq[1:]):
        ra, ca = rc(a); rb, cb = rc(b_)
        sep = "×" if UI["use_unicode"] and abs(ra - rb) == 2 else ("x" if abs(ra - rb) == 2 else "→" if UI["use_unicode"] else "-")
        parts.append(str(a)); parts.append(sep)
    parts.append(str(seq[-1]))
    return "".join(parts)

def count_pieces(b):
    blacks = sum(1 for v in b[1:] if v > 0)
    reds = sum(1 for v in b[1:] if v < 0)
    bk = sum(1 for v in b[1:] if v == 2)
    rk = sum(1 for v in b[1:] if v == -2)
    return blacks, reds, bk, rk

def parse_command(line):
    s = line.strip().lower()
    if s in ("q", "quit", "exit"): return ("quit", None)
    if s in ("r", "resign"): return ("resign", None)
    if s in ("h", "help", "?"): return ("help", None)
    if s in ("l", "list"): return ("list", None)
    if s in ("u", "undo"): return ("undo", None)
    if s.startswith("depth"):
        parts = s.split()
        if len(parts) == 2 and parts[1].isdigit():
            return ("depth", int(parts[1]))
        return ("depth", None)
    if s.startswith("mandatory"):
        parts = s.split()
        if len(parts) == 2 and parts[1] in ("on","off"):
            return ("mandatory", parts[1])
        return ("mandatory", None)
    if s.startswith("color"):
        parts = s.split()
        if len(parts) == 2 and parts[1] in ("on","off"):
            return ("color", parts[1])
        return ("color", None)
    if s.startswith("unicode"):
        parts = s.split()
        if len(parts) == 2 and parts[1] in ("on","off"):
            return ("unicode", parts[1])
        return ("unicode", None)
    if s in ("hint", "suggest"): return ("hint", None)
    if s.isdigit(): return ("pick", int(s))
    seq = parse_move_str(s)
    if seq:
        return ("move", seq)
    return (None, None)

ENHANCED_HELP = f"""
{Colors.BRIGHT_CYAN}COMMAND REFERENCE{Colors.RESET}
{Colors.MUTED}{'─' * 50}{Colors.RESET}

{Colors.SUCCESS}MAKING MOVES:{Colors.RESET}
  {Colors.ACCENT}12→16{Colors.RESET}     Enter move sequence (various formats accepted)
  {Colors.ACCENT}12×19×26{Colors.RESET}   Chain captures 
  {Colors.ACCENT}3{Colors.RESET}         Pick move by number from the list

{Colors.SUCCESS}GAME COMMANDS:{Colors.RESET}
  {Colors.ACCENT}help{Colors.RESET} (h)   Show this help screen
  {Colors.ACCENT}list{Colors.RESET} (l)   Show legal moves again  
  {Colors.ACCENT}hint{Colors.RESET}       Get engine suggestion for your move
  {Colors.ACCENT}undo{Colors.RESET} (u)   Take back the last move

{Colors.SUCCESS}GAME CONTROL:{Colors.RESET}
  {Colors.ACCENT}resign{Colors.RESET}     Concede the game
  {Colors.ACCENT}quit{Colors.RESET} (q)   Exit the application

{Colors.SUCCESS}SETTINGS:{Colors.RESET}
  {Colors.ACCENT}depth N{Colors.RESET}         Set engine difficulty (1-10)
  {Colors.ACCENT}mandatory on/off{Colors.RESET} Toggle forced captures
  {Colors.ACCENT}color on/off{Colors.RESET}     Toggle color display
  {Colors.ACCENT}unicode on/off{Colors.RESET}   Toggle Unicode pieces

{Colors.MUTED}{'─' * 50}{Colors.RESET}
{Colors.INFO}TIP: Legal move destinations are highlighted in green{Colors.RESET}
"""

def prompt_enhanced(msg):
    """Enhanced input prompt with styling"""
    try:
        prompt_text = f"{Colors.BRIGHT_WHITE}{msg}{Colors.RESET}"
        return input(prompt_text)
    except (EOFError, KeyboardInterrupt):
        print(f"\n{Colors.INFO}Goodbye!{Colors.RESET}")
        return "quit"

# ------------- Enhanced Game Loop -------------

def play_vs_engine(depth=6):
    global CAPTURES_MANDATORY
    b = initial_board()
    player = 1
    move_num = 1
    history = []
    last_move = None

    # Enhanced welcome sequence
    clear_screen()
    print_banner()
    
    welcome_msg = f"""
{Colors.SUCCESS}Welcome to Premium Checkers!{Colors.RESET}
You play as {style_text('BLACK', Colors.PLAYER, bold=True)} pieces (●/♛)
Engine plays as {style_text('RED', Colors.ENGINE, bold=True)} pieces (○/♕)

Captures are currently {style_text('OPTIONAL' if not CAPTURES_MANDATORY else 'MANDATORY', Colors.WARNING if CAPTURES_MANDATORY else Colors.SUCCESS, bold=True)}
Engine difficulty: {style_text(f'Level {depth}', Colors.ACCENT, bold=True)}

Type '{style_text('help', Colors.ACCENT)}' for commands or start playing!
    """
    
    print(welcome_msg)
    print_separator()

    engine = get_engine()

    while True:
        # Clear and redraw for better visual flow
        if UI["animations"] and move_num > 1:
            print("\n" * 2)
        
        # Enhanced board display
        moves = legal_moves(b, player)
        render_board_enhanced(b, last_move=last_move, legal_moves_list=moves if player == 1 else None)
        
        print()
        print_game_status(move_num, player, b, CAPTURES_MANDATORY)
        print()

        # Check terminal conditions
        if is_terminal(b, player):
            winner = style_text("RED", Colors.ENGINE, bold=True) if player == 1 else style_text("BLACK", Colors.PLAYER, bold=True)
            print_status_box("GAME OVER", f"{winner} WINS!", Colors.SUCCESS)
            break

        if player == 1:
            # Human turn with enhanced interface
            print_moves_enhanced(moves)
            print()
            
            while True:
                line = prompt_enhanced("Your move ➤ ")
                cmd, arg = parse_command(line)
                
                if cmd is None:
                    print(style_text("❌ Invalid input. Type 'help' for commands.", Colors.ERROR))
                    continue
                    
                if cmd == "help":
                    print(ENHANCED_HELP)
                    continue
                    
                if cmd == "list":
                    print_moves_enhanced(moves)
                    continue
                    
                if cmd == "depth":
                    if isinstance(arg, int) and 1 <= arg <= 10:
                        depth = arg
                        print(style_text(f"✓ Engine difficulty set to Level {depth}", Colors.SUCCESS))
                    else:
                        print(style_text("❌ Usage: depth N (where N is 1-10)", Colors.ERROR))
                    continue
                    
                if cmd == "mandatory":
                    if arg in ("on","off"):
                        CAPTURES_MANDATORY = (arg == "on")
                        status = "MANDATORY" if CAPTURES_MANDATORY else "OPTIONAL"
                        color = Colors.WARNING if CAPTURES_MANDATORY else Colors.SUCCESS
                        print(style_text(f"✓ Captures are now {status}", color))
                        moves = legal_moves(b, player)
                        print_moves_enhanced(moves)
                    else:
                        print(style_text("❌ Usage: mandatory on/off", Colors.ERROR))
                    continue
                    
                if cmd == "color":
                    if arg in ("on","off"):
                        UI["use_color"] = (arg == "on")
                        # Reinitialize color constants
                        for attr in dir(Colors):
                            if not attr.startswith('_'):
                                setattr(Colors, attr, getattr(Colors, attr) if UI["use_color"] else "")
                        print(style_text(f"✓ Colors {'enabled' if UI['use_color'] else 'disabled'}", Colors.SUCCESS))
                    else:
                        print(style_text("❌ Usage: color on/off", Colors.ERROR))
                    continue
                    
                if cmd == "unicode":
                    if arg in ("on","off"):
                        UI["use_unicode"] = (arg == "on")
                        print(style_text(f"✓ Unicode {'enabled' if UI['use_unicode'] else 'disabled'}", Colors.SUCCESS))
                    else:
                        print(style_text("❌ Usage: unicode on/off", Colors.ERROR))
                    continue
                    
                if cmd == "hint":
                    print(style_text("🤔 Analyzing position...", Colors.INFO))
                    t0 = time.time()
                    val, m = engine.search(b, player, max(2, min(10, depth)))
                    dt = time.time() - t0
                    if m:
                        hint_text = f"💡 Suggested move: {style_text(seq_to_str(m), Colors.ACCENT, bold=True)}\n"
                        hint_text += f"   Evaluation: {val:+d} ({dt:.2f}s)"
                        print_status_box("ENGINE HINT", hint_text, Colors.INFO)
                    else:
                        print(style_text("❌ No hint available", Colors.ERROR))
                    continue
                    
                if cmd == "undo":
                    if not history:
                        print(style_text("❌ Nothing to undo", Colors.ERROR))
                    else:
                        b, player, move_num, last_move = history.pop()
                        print(style_text("✓ Move undone", Colors.SUCCESS))
                        break
                    continue
                    
                if cmd == "resign":
                    print_status_box("GAME OVER", f"{style_text('RED', Colors.ENGINE, bold=True)} WINS BY RESIGNATION", Colors.ERROR)
                    return
                    
                if cmd == "quit":
                    print(style_text("👋 Thanks for playing!", Colors.INFO))
                    return
                    
                if cmd == "pick":
                    k = arg
                    if isinstance(k, int) and 1 <= k <= len(moves):
                        mv = moves[k - 1]
                        history.append((b.copy(), player, move_num, last_move))
                        b = apply_move(b, mv)
                        last_move = mv
                        print(style_text(f"✓ Played: {seq_to_str(mv)}", Colors.SUCCESS))
                        player = -player
                        if player == 1:
                            move_num += 1
                        break
                    else:
                        print(style_text(f"❌ Please enter a number between 1 and {len(moves)}", Colors.ERROR))
                    continue
                    
                if cmd == "move":
                    seq = arg
                    matched = None
                    for m in moves:
                        if m == seq:
                            matched = m
                            break
                    if matched is None:
                        print(style_text("❌ Illegal move. Use move number or exact sequence.", Colors.ERROR))
                        continue
                    history.append((b.copy(), player, move_num, last_move))
                    b = apply_move(b, matched)
                    last_move = matched
                    print(style_text(f"✓ Played: {seq_to_str(matched)}", Colors.SUCCESS))
                    player = -player
                    break
        else:
            # Enhanced engine turn
            print(style_text("🤖 Engine thinking...", Colors.ENGINE))
            
            # Show thinking animation if enabled
            if UI["animations"]:
                for i in range(3):
                    print(f"{Colors.ENGINE}{'.' * (i + 1)}{Colors.RESET}", end='\r')
                    time.sleep(0.3)
                print(" " * 10, end='\r')  # Clear dots
            
            t0 = time.time()
            val, m = engine.search(b, player, depth)
            dt = time.time() - t0
            
            if m is None:
                print_status_box("GAME OVER", f"{style_text('BLACK', Colors.PLAYER, bold=True)} WINS!", Colors.SUCCESS)
                break
            
            # Enhanced engine move display
            engine_move_text = f"Engine plays: {style_text(seq_to_str(m), Colors.ENGINE, bold=True)}\n"
            engine_move_text += f"Evaluation: {val:+d} | Time: {dt:.2f}s | Depth: {depth}"
            print_status_box("ENGINE MOVE", engine_move_text, Colors.ENGINE)
            
            history.append((b.copy(), player, move_num, last_move))
            b = apply_move(b, m)
            last_move = m
            player = -player
            move_num += 1
            
            # Brief pause for dramatic effect
            if UI["animations"]:
                time.sleep(0.5)

def main():
    global CAPTURES_MANDATORY
    depth = 6
    
    # Enhanced argument parsing
    args = sys.argv[1:]
    for a in args:
        if a.isdigit() and 1 <= int(a) <= 10:
            depth = int(a)
        elif a in ("--mandatory", "-m"):
            CAPTURES_MANDATORY = True
        elif a in ("--no-color", "--nocolor"):
            UI["use_color"] = False
        elif a in ("--ascii", "--no-unicode", "--nounicode"):
            UI["use_unicode"] = False
        elif a in ("--compact", "-c"):
            UI["compact"] = True
        elif a in ("--no-animations", "--static"):
            UI["animations"] = False
        elif a in ("-h", "--help"):
            help_text = f"""
{Colors.BRIGHT_CYAN}Enhanced Checkers - Command Line Arguments{Colors.RESET}

{Colors.SUCCESS}Usage:{Colors.RESET} python {sys.argv[0]} [options]

{Colors.SUCCESS}Options:{Colors.RESET}
  {Colors.ACCENT}N{Colors.RESET}                Engine difficulty level (1-10, default: 6)
  {Colors.ACCENT}--mandatory, -m{Colors.RESET}   Start with mandatory captures
  {Colors.ACCENT}--no-color{Colors.RESET}       Disable colored output  
  {Colors.ACCENT}--ascii{Colors.RESET}          Use ASCII pieces instead of Unicode
  {Colors.ACCENT}--compact, -c{Colors.RESET}     Use compact board layout
  {Colors.ACCENT}--no-animations{Colors.RESET}  Disable animations and delays
  {Colors.ACCENT}--help, -h{Colors.RESET}       Show this help message

{Colors.SUCCESS}Examples:{Colors.RESET}
  python {sys.argv[0]} 8 --mandatory    # Hard engine with forced captures
  python {sys.argv[0]} 4 --ascii         # Easy engine with ASCII pieces
  python {sys.argv[0]} --no-color        # Disable colors for terminal compatibility

{Colors.INFO}Features:{Colors.RESET}
  • Enhanced visual board with move highlighting
  • Smart move suggestions and hints  
  • Comprehensive undo system
  • Configurable difficulty and rules
  • Rich terminal UI with animations
  • Cross-platform compatibility
            """
            print(help_text)
            return
    
    try:
        play_vs_engine(depth)
    except KeyboardInterrupt:
        print(f"\n{Colors.INFO}Game interrupted. Goodbye!{Colors.RESET}")
    except Exception as e:
        print(f"{Colors.ERROR}Unexpected error: {e}{Colors.RESET}")
        print(f"{Colors.MUTED}Please report this issue.{Colors.RESET}")

if __name__ == "__main__":
    main()
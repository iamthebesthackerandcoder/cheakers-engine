# checkers_gui_tk.py
# A friendly Tkinter GUI for gameotherother.py
# Place this file next to gameotherother.py and run:
#   python checkers_gui_tk.py
#
# Updated to support:
# - Optional neural evaluation (toggle)
# - "Train AI" button running self-play training in background

import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox

# Import your engine and core logic
import importlib
engine_mod = importlib.import_module("gameotherother")

# Convenience aliases from your engine
rc = engine_mod.rc
idx_map = engine_mod.idx_map
initial_board = engine_mod.initial_board
legal_moves = engine_mod.legal_moves
apply_move = engine_mod.apply_move
is_terminal = engine_mod.is_terminal
seq_to_str = engine_mod.seq_to_str
count_pieces = engine_mod.count_pieces
get_engine = engine_mod.get_engine

# Neural evaluator and trainer
from neural_eval import get_neural_evaluator
from selfplay_trainer import SelfPlayTrainer

# -------------------- GUI Config --------------------

BOARD_BG_LIGHT = "#EEEED2"   # light square
BOARD_BG_DARK  = "#769656"   # dark/playable square
BOARD_HL_SQ    = "#F6F669"   # selected square highlight
BOARD_HL_DEST  = "#9AE66E"   # destination highlight
BOARD_LASTMOVE = "#F1C40F"   # last move highlight path

PIECE_BLACK_FILL = "#222222"
PIECE_BLACK_OUTLINE = "#FFFFFF"
PIECE_RED_FILL   = "#C0392B"
PIECE_RED_OUTLINE = "#FFFFFF"
KING_TEXT_COLOR = "#FFD700"  # gold-ish

SQUARE_SIZE = 72   # pixels per square (window will be ~8*SQUARE_SIZE)

# -------------------- Helpers --------------------

def group_moves_by_start(moves):
    d = {}
    for m in moves:
        d.setdefault(m[0], []).append(m)
    return d

def group_moves_by_dest(moves):
    d = {}
    for m in moves:
        d.setdefault(m[-1], []).append(m)
    return d

def manhattan(a, b):
    ra, ca = rc(a)
    rb, cb = rc(b)
    return abs(ra - rb) + abs(ca - cb)

# -------------------- Main UI --------------------

class CheckersUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Checkers - GUI")
        self.resizable(False, False)

        # Game state
        self.board = initial_board()
        self.player = 1          # 1 = Black (human default), -1 = Red (engine default)
        self.human_side = tk.StringVar(value="Black")
        self.move_num = 1
        self.last_move = None
        self.history = []        # stack of (board_copy, player, move_num, last_move)
        self.legal = []
        self.moves_by_start = {}
        self.selected = None     # currently selected square index
        self.dest_map = {}       # dest -> [seqs] for selected piece
        self.engine = get_engine()
        self.engine_depth = tk.IntVar(value=6)
        self.mandatory_var = tk.BooleanVar(value=engine_mod.CAPTURES_MANDATORY)
        self.thinking = False
        self.hint_seq = None
        self.flipped = False

        # Neural eval toggle
        self.use_neural_var = tk.BooleanVar(value=False)
        self.training_active = False

        # UI layout
        self._build_ui()
        self._new_game(reset_side=False)  # keep default human side

    # -------- UI Construction --------

    def _build_ui(self):
        container = ttk.Frame(self, padding=8)
        container.grid(row=0, column=0, sticky="nsew")

        # Left: Board
        self.canvas = tk.Canvas(container,
                                width=8 * SQUARE_SIZE,
                                height=8 * SQUARE_SIZE,
                                highlightthickness=0,
                                bg=BOARD_BG_LIGHT)
        self.canvas.grid(row=0, column=0, rowspan=3, sticky="n")
        self.canvas.bind("<Button-1>", self.on_click)

        # Right: Controls
        controls = ttk.Frame(container, padding=(10, 0))
        controls.grid(row=0, column=1, sticky="nw")

        title = ttk.Label(controls, text="Checkers", font=("Segoe UI", 18, "bold"))
        title.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))

        # Side selector
        ttk.Label(controls, text="You play as:").grid(row=1, column=0, sticky="w")
        side_combo = ttk.Combobox(controls, values=["Black", "Red"],
                                  state="readonly", textvariable=self.human_side, width=10)
        side_combo.grid(row=1, column=1, sticky="w")
        side_combo.bind("<<ComboboxSelected>>", lambda e: self._new_game(reset_side=True))

        # Difficulty
        ttk.Label(controls, text="Difficulty:").grid(row=2, column=0, sticky="w", pady=(8,0))
        depth_scale = ttk.Scale(controls, from_=1, to=10,
                                orient="horizontal",
                                variable=self.engine_depth,
                                command=self._on_depth_slide)
        depth_scale.grid(row=2, column=1, sticky="we", padx=(0,4))
        self.depth_label = ttk.Label(controls, text=f"Level {self.engine_depth.get()}")
        self.depth_label.grid(row=2, column=1, sticky="e")

        # Mandatory captures
        mandatory_chk = ttk.Checkbutton(controls, text="Mandatory captures",
                                        variable=self.mandatory_var,
                                        command=self._toggle_mandatory)
        mandatory_chk.grid(row=3, column=0, columnspan=2, sticky="w", pady=(8,0))

        # Neural eval toggle
        nn_chk = ttk.Checkbutton(controls, text="Use Neural Eval",
                                 variable=self.use_neural_var,
                                 command=self._toggle_use_neural)
        nn_chk.grid(row=4, column=0, columnspan=2, sticky="w", pady=(8,0))

        # Buttons
        btns = ttk.Frame(controls)
        btns.grid(row=5, column=0, columnspan=2, sticky="we", pady=(10, 4))
        self.btn_new = ttk.Button(btns, text="New Game", command=lambda: self._new_game(reset_side=False))
        self.btn_new.grid(row=0, column=0, padx=(0,5))
        self.btn_undo = ttk.Button(btns, text="Undo", command=self._undo)
        self.btn_undo.grid(row=0, column=1, padx=5)
        self.btn_hint = ttk.Button(btns, text="Hint", command=self._hint)
        self.btn_hint.grid(row=0, column=2, padx=5)
        self.btn_flip = ttk.Button(btns, text="Flip Board", command=self._flip_board)
        self.btn_flip.grid(row=0, column=3, padx=5)
        self.btn_train = ttk.Button(btns, text="Train AI", command=self._start_training)
        self.btn_train.grid(row=0, column=4, padx=5)

        # Move list (click to play)
        ttk.Label(controls, text="Your legal moves:").grid(row=6, column=0, columnspan=2, sticky="w", pady=(8, 2))
        self.moves_list = tk.Listbox(controls, height=10, width=28, activestyle="dotbox")
        self.moves_list.grid(row=7, column=0, columnspan=2, sticky="w")
        self.moves_list.bind("<Double-Button-1>", self._play_selected_move)

        # Status panel
        status = ttk.Frame(container, padding=(0, 8))
        status.grid(row=2, column=1, sticky="nw")
        self.lbl_turn = ttk.Label(status, text="", font=("Segoe UI", 11, "bold"))
        self.lbl_turn.grid(row=0, column=0, sticky="w")
        self.lbl_counts = ttk.Label(status, text="")
        self.lbl_counts.grid(row=1, column=0, sticky="w", pady=(4,0))
        self.lbl_eval = ttk.Label(status, text="", foreground="#888")
        self.lbl_eval.grid(row=2, column=0, sticky="w", pady=(4,0))

        # Pre-draw board grid (squares only)
        self._draw_squares()

    # -------- Game Flow --------

    def _new_game(self, reset_side: bool):
        self.board = initial_board()
        self.move_num = 1
        self.last_move = None
        self.history.clear()
        self.selected = None
        self.hint_seq = None

        # Set player based on chosen side
        side = self.human_side.get()
        self.player = 1 if side == "Black" else -1

        # Recompute moves
        self._refresh_moves()
        self._redraw()

        # If engine plays first (human is Red), let engine move
        if side == "Red":
            self.after(300, self._engine_move_async)

    def _toggle_mandatory(self):
        engine_mod.CAPTURES_MANDATORY = self.mandatory_var.get()
        self._refresh_moves()
        self._redraw()

    def _toggle_use_neural(self):
        if self.use_neural_var.get():
            evaluator = get_neural_evaluator()
            self.engine.neural_evaluator = evaluator
            self.lbl_eval.config(text="Neural evaluation enabled")
        else:
            self.engine.neural_evaluator = None
            self.lbl_eval.config(text="Neural evaluation disabled")

    def _on_depth_slide(self, val):
        v = max(1, min(10, int(float(val))))
        self.engine_depth.set(v)
        self.depth_label.config(text=f"Level {v}")

    def _refresh_moves(self):
        self.legal = legal_moves(self.board, self.player)
        self.moves_by_start = group_moves_by_start(self.legal)
        self._refresh_moves_list()
        self._update_status()

    def _refresh_moves_list(self):
        self.moves_list.delete(0, tk.END)
        if not self.legal:
            self.moves_list.insert(tk.END, "(no legal moves)")
            return
        # Show captures first
        jumps = []
        regs = []
        for m in self.legal:
            a, b = m[0], m[1]
            ra, ca = rc(a); rb, cb = rc(b)
            if abs(ra - rb) == 2:
                jumps.append(m)
            else:
                regs.append(m)
        order = jumps + regs
        for m in order:
            self.moves_list.insert(tk.END, seq_to_str(m))

    def _update_status(self):
        side_name = "BLACK" if self.player == 1 else "RED"
        self.lbl_turn.config(text=f"Move {self.move_num} — Turn: {side_name}")
        blacks, reds, bk, rk = count_pieces(self.board)
        self.lbl_counts.config(text=f"Pieces: Black {blacks} (K:{bk}) | Red {reds} (K:{rk})")

    def _flip_board(self):
        self.flipped = not self.flipped
        self._redraw()

    # -------- Rendering --------

    def _draw_squares(self):
        self.canvas.delete("square")
        for r in range(8):
            for c in range(8):
                x1 = c * SQUARE_SIZE
                y1 = r * SQUARE_SIZE
                x2 = x1 + SQUARE_SIZE
                y2 = y1 + SQUARE_SIZE
                color = BOARD_BG_DARK if (r + c) % 2 == 1 else BOARD_BG_LIGHT
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline=color, tags=("square",))

    def _redraw(self):
        # Clear old overlays
        self.canvas.delete("piece")
        self.canvas.delete("sel")
        self.canvas.delete("dest")
        self.canvas.delete("last")
        self.canvas.delete("hint")

        # Last move path highlight
        if self.last_move:
            path = self.last_move
            for a, b in zip(path, path[1:]):
                ra, ca = rc(a)
                rb, cb = rc(b)
                if self.flipped:
                    ra = 7 - ra
                    ca = 7 - ca
                    rb = 7 - rb
                    cb = 7 - cb
                x1 = ca * SQUARE_SIZE
                y1 = ra * SQUARE_SIZE
                x2 = cb * SQUARE_SIZE
                y2 = rb * SQUARE_SIZE
                self.canvas.create_rectangle(x1, y1, x2, y2, outline=BOARD_LASTMOVE,
                                             width=4, tags=("last",))

        # Draw pieces
        for idx in range(1, 33):
            v = self.board[idx]
            if v == 0:
                continue
            r, c = rc(idx)
            if self.flipped:
                r = 7 - r
                c = 7 - c
            cx = c * SQUARE_SIZE + SQUARE_SIZE // 2
            cy = r * SQUARE_SIZE + SQUARE_SIZE // 2
            rad = int(SQUARE_SIZE * 0.36)
            fill = PIECE_BLACK_FILL if v > 0 else PIECE_RED_FILL
            outline = PIECE_BLACK_OUTLINE if v > 0 else PIECE_RED_OUTLINE
            self.canvas.create_oval(cx - rad, cy - rad, cx + rad, cy + rad,
                                    fill=fill, outline=outline, width=2,
                                    tags=("piece",))
            # King marker
            if abs(v) == 2:
                self.canvas.create_text(cx, cy, text="K", fill=KING_TEXT_COLOR,
                                        font=("Segoe UI", int(SQUARE_SIZE*0.33), "bold"),
                                        tags=("piece",))

        # Selection highlight
        if self.selected:
            r, c = rc(self.selected)
            if self.flipped:
                r = 7 - r
                c = 7 - c
            x1 = c * SQUARE_SIZE
            y1 = r * SQUARE_SIZE
            x2 = x1 + SQUARE_SIZE
            y2 = y1 + SQUARE_SIZE
            self.canvas.create_rectangle(x1+3, y1+3, x2-3, y2-3,
                                         outline=BOARD_HL_SQ, width=3, tags=("sel",))

        # Destination highlights
        for dst in getattr(self, "highlight_dests", []):
            r, c = rc(dst)
            if self.flipped:
                r = 7 - r
                c = 7 - c
            cx = c * SQUARE_SIZE + SQUARE_SIZE // 2
            cy = r * SQUARE_SIZE + SQUARE_SIZE // 2
            rad = int(SQUARE_SIZE * 0.16)
            self.canvas.create_oval(cx - rad, cy - rad, cx + rad, cy + rad,
                                    fill=BOARD_HL_DEST, outline="", tags=("dest",))

        # Hint overlay (path)
        if self.hint_seq:
            for a, b in zip(self.hint_seq, self.hint_seq[1:]):
                ra, ca = rc(a)
                rb, cb = rc(b)
                if self.flipped:
                    ra = 7 - ra
                    ca = 7 - ca
                    rb = 7 - rb
                    cb = 7 - cb
                x1 = ca * SQUARE_SIZE + SQUARE_SIZE//2
                y1 = ra * SQUARE_SIZE + SQUARE_SIZE//2
                x2 = cb * SQUARE_SIZE + SQUARE_SIZE//2
                y2 = rb * SQUARE_SIZE + SQUARE_SIZE//2
                self.canvas.create_line(x1, y1, x2, y2, fill="#AA66FF",
                                        width=5, arrow="last", tags=("hint",))

    # -------- Interaction --------

    def on_click(self, event):
        if self.thinking:
            return
        display_c = event.x // SQUARE_SIZE
        display_r = event.y // SQUARE_SIZE
        if not (0 <= display_r < 8 and 0 <= display_c < 8):
            return
        original_r = display_r if not self.flipped else 7 - display_r
        original_c = display_c if not self.flipped else 7 - display_c
        if (original_r + original_c) % 2 == 0:
            return  # non-playable square
        idx = idx_map.get((original_r, original_c))
        if not idx:
            return
        self._handle_square_click(idx)

    def _handle_square_click(self, idx):
        # If selecting a piece
        if self.selected is None:
            v = self.board[idx]
            if v == 0 or v * self.player <= 0:
                return
            self.selected = idx
            # Collect all moves for this start
            start_moves = self.moves_by_start.get(idx, [])
            self.dest_map = group_moves_by_dest(start_moves)
            self.highlight_dests = list(self.dest_map.keys())
            self.hint_seq = None
            self._redraw()
            self._focus_move_in_list_from_start(idx)
            return

        # Clicking the same square deselects
        if idx == self.selected:
            self.selected = None
            self.dest_map = {}
            self.highlight_dests = []
            self._redraw()
            return

        # If click on a destination for selected piece, play that move
        if idx in self.dest_map:
            seqs = self.dest_map[idx]
            # If multiple sequences, prefer the longest capture; if tie, first
            best = max(seqs, key=lambda s: (len(s), s))
            self._play_move(best)
            return

        # Otherwise, try selecting another piece of current side
        v = self.board[idx]
        if v != 0 and v * self.player > 0:
            self.selected = idx
            start_moves = self.moves_by_start.get(idx, [])
            self.dest_map = group_moves_by_dest(start_moves)
            self.highlight_dests = list(self.dest_map.keys())
            self.hint_seq = None
            self._redraw()
            self._focus_move_in_list_from_start(idx)
            return

        # Invalid click; ignore
        return

    def _focus_move_in_list_from_start(self, start_idx):
        # Try to select the first move in list that matches this start
        for i, m in enumerate(self._ordered_moves_for_list()):
            if m[0] == start_idx:
                self.moves_list.selection_clear(0, tk.END)
                self.moves_list.selection_set(i)
                self.moves_list.see(i)
                break

    def _ordered_moves_for_list(self):
        # Same ordering used in list: captures first
        jumps = []
        regs = []
        for m in self.legal:
            a, b = m[0], m[1]
            ra, ca = rc(a); rb, cb = rc(b)
            if abs(ra - rb) == 2:
                jumps.append(m)
            else:
                regs.append(m)
        return jumps + regs

    def _play_selected_move(self, _event=None):
        if self.thinking:
            return
        idx = self.moves_list.curselection()
        if not idx:
            return
        i = idx[0]
        ordered = self._ordered_moves_for_list()
        if not ordered:
            return
        seq = ordered[i]
        self._play_move(seq)

    def _play_move(self, seq):
        if not seq in self.legal:
            messagebox.showerror("Illegal move", "That move is not legal.")
            return
        # Record history
        self.history.append((self.board.copy(), self.player, self.move_num, self.last_move))
        # Apply
        self.board = apply_move(self.board, seq)
        self.last_move = seq
        # Switch side
        self.player = -self.player
        if self.player == 1:
            self.move_num += 1
        # Reset selection/hint
        self.selected = None
        self.dest_map = {}
        self.highlight_dests = []
        self.hint_seq = None

        self._redraw()

        # Check terminal
        if is_terminal(self.board, self.player):
            winner = "RED" if self.player == 1 else "BLACK"
            self._refresh_moves()
            self._redraw()
            messagebox.showinfo("Game Over", f"{winner} wins!")
            return

        self._refresh_moves()
        self._redraw()

        # Engine turn if next to move is not the human
        human_is_black = (self.human_side.get() == "Black")
        human_player = 1 if human_is_black else -1
        if self.player != human_player:
            self.after(100, self._engine_move_async)

    def _undo(self):
        if self.thinking:
            return
        if not self.history:
            return
        self.board, self.player, self.move_num, self.last_move = self.history.pop()
        self.selected = None
        self.dest_map = {}
        self.highlight_dests = []
        self.hint_seq = None
        self._refresh_moves()
        self._redraw()

    # -------- Engine / Hint --------

    def _engine_move_async(self):
        if self.thinking:
            return
        self.thinking = True
        self._set_controls_state(False)
        self.lbl_eval.config(text="Engine thinking...")

        depth = max(1, min(10, int(self.engine_depth.get())))
        board_copy = self.board.copy()
        player_to_move = self.player

        def worker():
            t0 = time.time()
            val, move = self.engine.search(board_copy, player_to_move, depth)
            dt = time.time() - t0
            self.after(0, lambda: self._engine_move_apply(val, move, dt))

        threading.Thread(target=worker, daemon=True).start()

    def _engine_move_apply(self, val, move, dt):
        self.thinking = False
        self._set_controls_state(True)

        if move is None:
            # No engine move; game over
            winner = "BLACK" if self.player == -1 else "RED"
            self._refresh_moves()
            self._redraw()
            messagebox.showinfo("Game Over", f"{winner} wins!")
            return

        # Record history
        self.history.append((self.board.copy(), self.player, self.move_num, self.last_move))

        # Apply move
        self.board = apply_move(self.board, move)
        self.last_move = move
        self.player = -self.player
        self.move_num += 1

        self._refresh_moves()
        self._redraw()
        self.lbl_eval.config(text=f"Engine depth {self.engine_depth.get()} | Eval {val:+d} | {dt:.2f}s")

        # Check terminal after engine move
        if is_terminal(self.board, self.player):
            winner = "RED" if self.player == 1 else "BLACK"
            messagebox.showinfo("Game Over", f"{winner} wins!")

    def _hint(self):
        if self.thinking:
            return
        if not self.legal:
            return
        self.thinking = True
        self._set_controls_state(False)
        self.lbl_eval.config(text="Analyzing hint...")

        depth = max(2, min(8, int(self.engine_depth.get())))
        board_copy = self.board.copy()
        player_to_move = self.player

        def worker():
            t0 = time.time()
            val, move = self.engine.search(board_copy, player_to_move, depth)
            dt = time.time() - t0
            self.after(0, lambda: self._hint_apply(val, move, dt))

        threading.Thread(target=worker, daemon=True).start()

    def _hint_apply(self, val, move, dt):
        self.thinking = False
        self._set_controls_state(True)
        if move:
            self.hint_seq = move
            self._redraw()
            self.lbl_eval.config(text=f"Hint: {seq_to_str(move)} | {val:+d} | {dt:.2f}s")
        else:
            self.hint_seq = None
            self.lbl_eval.config(text=f"No hint available")

    def _set_controls_state(self, enabled: bool):
        state = ("!disabled" if enabled else "disabled")
        for w in (self.btn_new, self.btn_undo, self.btn_hint, self.moves_list, self.btn_flip, self.btn_train):
            try:
                w.state([state])
            except Exception:
                try:
                    w.configure(state=("normal" if enabled else "disabled"))
                except Exception:
                    pass

    # -------- Training Integration --------

    def _start_training(self):
        if self.training_active:
            return
        self.training_active = True
        self._set_controls_state(False)
        self.btn_train.configure(text="Training...")

        # Ask user for number of games with increased default
        try:
            import tkinter.simpledialog as simpledialog
            num_games = simpledialog.askinteger("Training Setup", 
                                                "Number of self-play games:",
                                                initialvalue=200, minvalue=10, maxvalue=5000)
            if not num_games:
                self.training_active = False
                self._set_controls_state(True)
                self.btn_train.configure(text="Train AI")
                return
        except Exception:
            num_games = 200
        
        # Create progress window
        self._create_progress_window(num_games)

        def worker():
            try:
                trainer = SelfPlayTrainer()
                # Define progress callback
                def progress_cb(games, total):
                    self.after(0, lambda: self._update_progress(games, total))
                # Monitor progress and update UI
                save_interval = max(25, num_games//8)
                trainer.run_training_session(num_games=num_games, save_interval=save_interval, progress_callback=progress_cb)
            except Exception as e:
                error_str = str(e)
                self.after(0, lambda: self._training_error(error_str))
                return
            self.after(0, self._training_completed)

        threading.Thread(target=worker, daemon=True).start()

    def _training_completed(self):
        self.training_active = False
        self._set_controls_state(True)
        self.btn_train.configure(text="Train AI")
        
        # Close progress window
        if hasattr(self, 'progress_window'):
            self.progress_window.destroy()
        
        # If neural eval is enabled, keep using the (now updated) evaluator
        if self.use_neural_var.get():
            self.engine.neural_evaluator = get_neural_evaluator()
        messagebox.showinfo("Training Complete",
                            "Self-play training completed!\n"
                            "The neural network has been updated and saved.")

    def _training_error(self, error_msg):
        self.training_active = False
        self._set_controls_state(True)
        self.btn_train.configure(text="Train AI")
        
        # Close progress window
        if hasattr(self, 'progress_window'):
            self.progress_window.destroy()
        
        messagebox.showerror("Training Error", f"Training failed: {error_msg}")

# -------------------- Run --------------------

    def _create_progress_window(self, num_games):
        self.progress_window = tk.Toplevel(self)
        self.progress_window.title("Training Progress")
        self.progress_window.geometry("350x120")
        self.progress_window.transient(self)
        self.progress_window.grab_set()
        self.progress_window.resizable(False, False)
        
        ttk.Label(self.progress_window,
                  text="Self-Play Training in Progress...",
                  font=("Segoe UI", 10, "bold")).pack(pady=10)
        
        self.progress_label = ttk.Label(self.progress_window,
                                        text=f"Completed 0/{num_games} games")
        self.progress_label.pack(pady=5)
        
        self.progress_bar = ttk.Progressbar(self.progress_window,
                                            mode='determinate',
                                            maximum=num_games,
                                            length=300)
        self.progress_bar.pack(pady=10)
        
        ttk.Button(self.progress_window,
                   text="Cancel",
                   command=self._cancel_training).pack(pady=5)
    
    def _update_progress(self, games, total):
        self.progress_label.config(text=f"Completed {games}/{total} games")
        self.progress_bar['value'] = games
    
    def _cancel_training(self):
        """Cancel training by setting flag and closing window."""
        self.training_active = False
        if hasattr(self, 'progress_window'):
            self.progress_window.destroy()
        self._set_controls_state(True)
        self.btn_train.configure(text="Train AI")
        # Note: Daemon thread will continue but window is closed

if __name__ == "__main__":
    app = CheckersUI()
    app.mainloop()
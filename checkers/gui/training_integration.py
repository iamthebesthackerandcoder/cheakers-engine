"""
Training integration for the Checkers GUI.
"""
from __future__ import annotations

from typing import Optional, Callable, Any
import threading
import tkinter as tk
from tkinter import messagebox, ttk

from checkers.training.selfplay import SelfPlayTrainer


class TrainingIntegration:
    """Handles training workflow integration."""

    def __init__(self, parent: tk.Tk):
        self.parent = parent
        self.is_training = False
        self.progress_window: Optional[tk.Toplevel] = None
        self.progress_bar: Optional[ttk.Progressbar] = None
        self.progress_label: Optional[ttk.Label] = None

    def start_training_async(self, num_games: int = 200,
                           on_complete: Optional[Callable[[], None]] = None,
                           on_error: Optional[Callable[[str], None]] = None) -> None:
        """Start training asynchronously."""
        if self.is_training:
            return

        self.is_training = True
        self._create_progress_window(num_games)

        def worker():
            try:
                trainer = SelfPlayTrainer()

                def progress_callback(games_completed: int, total_games: int):
                    if self.parent:
                        self.parent.after(0, lambda: self._update_progress(games_completed, total_games))

                # Monitor progress and update UI
                save_interval = max(25, num_games // 8)
                trainer.run_training_session(
                    num_games=num_games,
                    save_interval=save_interval,
                    progress_callback=progress_callback
                )

                # Training completed successfully
                if self.parent:
                    self.parent.after(0, self._training_completed)

                if on_complete:
                    if self.parent:
                        self.parent.after(0, on_complete)

            except Exception as e:
                error_msg = str(e)
                if self.parent:
                    self.parent.after(0, lambda: self._training_error(error_msg))
                if on_error:
                    if self.parent:
                        self.parent.after(0, lambda: on_error(error_msg))

        threading.Thread(target=worker, daemon=True).start()

    def cancel_training(self) -> None:
        """Cancel the current training session."""
        self.is_training = False
        if self.progress_window:
            self.progress_window.destroy()

    def _create_progress_window(self, num_games: int) -> None:
        """Create and show the training progress window."""
        self.progress_window = tk.Toplevel(self.parent)
        self.progress_window.title("Training Progress")
        self.progress_window.geometry("350x120")
        self.progress_window.transient(self.parent)
        self.progress_window.grab_set()
        self.progress_window.resizable(False, False)

        # Title
        ttk.Label(
            self.progress_window,
            text="Self-Play Training in Progress...",
            font=("Segoe UI", 10, "bold")
        ).pack(pady=10)

        # Progress label
        self.progress_label = ttk.Label(
            self.progress_window,
            text=f"Completed 0/{num_games} games"
        )
        self.progress_label.pack(pady=5)

        # Progress bar
        self.progress_bar = ttk.Progressbar(
            self.progress_window,
            mode='determinate',
            maximum=num_games,
            length=300
        )
        self.progress_bar.pack(pady=10)

        # Cancel button
        ttk.Button(
            self.progress_window,
            text="Cancel",
            command=self.cancel_training
        ).pack(pady=5)

    def _update_progress(self, games_completed: int, total_games: int) -> None:
        """Update the progress display."""
        if self.progress_label:
            self.progress_label.config(text=f"Completed {games_completed}/{total_games} games")
        if self.progress_bar:
            self.progress_bar['value'] = games_completed

    def _training_completed(self) -> None:
        """Handle training completion."""
        self.is_training = False
        if self.progress_window:
            self.progress_window.destroy()

    def _training_error(self, error_msg: str) -> None:
        """Handle training error."""
        self.is_training = False
        if self.progress_window:
            self.progress_window.destroy()
        messagebox.showerror("Training Error", f"Training failed: {error_msg}")

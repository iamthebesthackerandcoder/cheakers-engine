# Self-Play Training System for Neural Checkers Engine
# Now actually updates the neural network via simple supervised learning.

import numpy as np
import pickle
import time
import threading
import random
from copy import deepcopy
from collections import deque
import os
import multiprocessing as mp
from multiprocessing import Pool, Manager

from typing import Optional, Dict, Any, List, Tuple, Callable, Union

# Type aliases for better readability (avoiding circular imports)
Board = List[int]  # Board state: index 0 is unused, 1-32 are squares
Player = int  # 1 for black, -1 for red
GameResult = Tuple[int, Optional[List[int]]]  # (score, best_move)
TrainingStats = Dict[str, Any]  # Statistics from a training session
CacheStats = Dict[str, Any]
NeuralNetworkProtocol = Any  # Protocol for neural network implementations
TrainingDataCollectorProtocol = Any  # Protocol for training data collection

# Import the neural evaluator from the previous implementation
from neural_eval import NeuralEvaluator, TrainingDataCollector, get_neural_evaluator
from config import (
    DEFAULT_SEARCH_DEPTHS,
    EPSILON_START,
    EPSILON_END,
    EPSILON_DECAY_GAMES,
)

class SelfPlayTrainer:
    """
    Self-play training system that generates training data by playing games
    between different versions of the neural network, and performs simple
    supervised updates to the base network.
    """

    def __init__(self, base_evaluator: Optional[NeuralEvaluator] = None,
                 resume_from_checkpoint: Optional[str] = None) -> None:
        """Initialize the self-play trainer."""
        self.base_evaluator: NeuralEvaluator = base_evaluator or get_neural_evaluator()
        self.training_data: TrainingDataCollector = TrainingDataCollector(use_augmentation=True)
        self.games_played: int = 0
        self.total_positions: int = 0
        self.training_stats: TrainingStats = dict()
        # Epsilon schedule parameters (from config)
        self.epsilon_start: float = EPSILON_START
        self.epsilon_end: float = EPSILON_END
        self.epsilon_decay_games: int = EPSILON_DECAY_GAMES
        self.current_epsilon: float = self.epsilon_start  # Current epsilon value

        # Load previous training progress if specified
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)

    def create_opponent_evaluator(self, noise_level=0.3, mutation_rate=0.15):
        """
        Create a slightly different version of the evaluator for opponent play.
        This introduces diversity in self-play games.
        """
        opponent = NeuralEvaluator()
        opponent.weights = {k: v.copy() if isinstance(v, np.ndarray) else deepcopy(v) for k, v in self.base_evaluator.weights.items()}

        # Use random seed based on time to ensure different opponents
        np.random.seed(None)  # Use current time as seed

        # Add noise to create variation
        for layer in opponent.weights:
            # Keys are strings like 'W1', 'b1', ...
            if layer.startswith('W'):  # Weight matrices
                noise = np.random.randn(*opponent.weights[layer].shape) * noise_level
                mask = np.random.random(opponent.weights[layer].shape) < mutation_rate
                opponent.weights[layer] += noise * mask
            elif layer.startswith('b'):  # Bias vectors
                noise = np.random.randn(*opponent.weights[layer].shape) * noise_level * 0.5
                mask = np.random.random(opponent.weights[layer].shape) < mutation_rate
                opponent.weights[layer] += noise * mask

        return opponent

    def _play_single_game(self, game_num, epsilon, base_evaluator, opponent_evaluator, shared_tt):
        return _play_single_game_core(game_num, epsilon, base_evaluator, opponent_evaluator, shared_tt, deepcopy_evaluators=False)


    def train_on_collected_data(self, epochs=2, batch_size=1024, lr=5e-4, l2=1e-6):
        """Perform a simple supervised training pass on the collected data."""
        if len(self.training_data.positions) == 0:
            print("No training data collected yet.")
            return None

        X = np.array(self.training_data.positions, dtype=np.float32)
        y = np.array(self.training_data.scores, dtype=np.float32)
        print(f"Training on {len(X)} positions...")
        stats = self.base_evaluator.train_supervised(
            X, y, epochs=epochs, batch_size=batch_size, lr=lr, l2=l2, shuffle=True, verbose=True
        )
        return stats

    def run_training_session(self, num_games=100, save_interval=10, num_workers=4,
                             model_save_path="data/neural_model.pkl",
                             data_save_path="data/training_data.pkl",
                             checkpoint_path="data/training_checkpoint.pkl",
                             progress_callback=None):
        """
        Run a complete self-play training session.
        """
        random.seed(int(time.time()))
        print(f"Starting self-play training session: {num_games} games")
        print(f"Model will be saved to: {model_save_path}")
        print(f"Training data will be saved to: {data_save_path}")
        print("-" * 60)

        start_time = time.time()

        if num_games < num_workers * 2:
            shared_tt = None
            for game_num in range(num_games):
                # Update epsilon based on current game number (linear decay)
                progress = min(1.0, (game_num) / max(1, self.epsilon_decay_games))
                self.current_epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * progress

                opponent = self.create_opponent_evaluator()
                result, positions, moves, random_moves, game_positions_list, game_time = self._play_single_game(game_num, self.current_epsilon, self.base_evaluator, opponent, shared_tt)

                # Update exploration tracking stats
                self.training_stats.random_moves_total += random_moves
                if not hasattr(self.training_stats, 'random_moves_per_game'):
                    self.training_stats.random_moves_per_game = []
                self.training_stats.random_moves_per_game.append(random_moves)

                if result == 1:
                    self.training_stats.black_wins += 1
                elif result == -1:
                    self.training_stats.red_wins += 1
                else:
                    self.training_stats.draws += 1

                self.games_played += 1
                self.total_positions += positions
                self.training_stats.positions_collected = self.total_positions

                total_games: int = self.training_stats.black_wins + self.training_stats.red_wins + self.training_stats.draws
                if total_games > 0:
                    self.training_stats.avg_game_length = ((self.training_stats.avg_game_length * (total_games - 1)) + moves) / total_games

                for board_state, pos_player, target_score in game_positions_list:
                    self.training_data.add_position(board_state, pos_player, target_score)

                # Print progress with exploration metrics
                if (game_num + 1) % 5 == 0 or game_num == 0:
                    result_str = "Black wins" if result == 1 else "Red wins" if result == -1 else "Draw"
                    avg_random_moves = sum(self.training_stats.random_moves_per_game[-5:]) / min(5, len(self.training_stats.random_moves_per_game))
                    print(f"Game {game_num + 1:3d}: {result_str:10s} | "
                          f"Moves: {moves:3d} | Positions: {positions:3d} | "
                          f"Random: {random_moves:2d} (ε={self.current_epsilon:.3f}) | "
                          f"Time: {game_time:.2f}s")

                # Report progress
                if progress_callback:
                    progress_callback(game_num + 1, num_games)

                # Save progress and train periodically
                if (game_num + 1) % save_interval == 0:
                    # Train on the data collected so far
                    stats = self.train_on_collected_data(epochs=2, batch_size=1024, lr=5e-4, l2=1e-6)
                    if stats and stats.get("loss") is not None:
                        print(f"Post-chunk training: MSE {stats['loss']:.4f} on {stats['count']} samples")
                    # Save progress
                    self.save_training_progress(model_save_path, data_save_path, checkpoint_path)
                    self.print_training_stats()
                    print("-" * 60)
                    # Optionally clear data to avoid overfitting on same samples repeatedly
                    self.training_data.clear()
        else:
            runner = ParallelGameRunner(self, num_workers)
            runner.run_games(num_games, save_interval, model_save_path, data_save_path, checkpoint_path, progress_callback)

        # Final report
        if progress_callback:
            progress_callback(num_games, num_games)

        # Final train and save
        if len(self.training_data.positions) > 0:
            stats = self.train_on_collected_data(epochs=3, batch_size=1024, lr=5e-4, l2=1e-6)
            if stats and stats.get("loss") is not None:
                print(f"Final training: MSE {stats['loss']:.4f} on {stats['count']} samples")

        self.save_training_progress(model_save_path, data_save_path, checkpoint_path)

        elapsed_time = time.time() - start_time
        print(f"\nTraining session completed!")
        print(f"Total time: {elapsed_time:.2f} seconds")
        print(f"Games per second: {num_games / max(1e-9, elapsed_time):.2f}")
        self.print_training_stats()

    def save_checkpoint(self, checkpoint_path):
        """
        Save complete training checkpoint including model, stats, and collected data.
        """
        # Ensure directory exists
        chk_dir = os.path.dirname(checkpoint_path)
        if chk_dir and not os.path.exists(chk_dir):
            os.makedirs(chk_dir, exist_ok=True)
        checkpoint_data = {
            'games_played': self.games_played,
            'total_positions': self.total_positions,
            'training_stats': self.training_stats,
            'model_weights': self.base_evaluator.weights,
            'adam_state': {
                'adam_m': self.base_evaluator.adam_m,
                'adam_v': self.base_evaluator.adam_v,
                'adam_t': self.base_evaluator.adam_t
            }
        }

        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        print(f"Training checkpoint saved to: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        """
        Load training checkpoint and resume from where we left off.
        """
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint file not found: {checkpoint_path}")
            return False

        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)

            # Restore training progress
            self.games_played = checkpoint_data.get('games_played', 0)
            self.total_positions = checkpoint_data.get('total_positions', 0)
            self.training_stats = checkpoint_data.get('training_stats', {
                'black_wins': 0, 'red_wins': 0, 'draws': 0,
                'avg_game_length': 0, 'positions_collected': 0
            })

            # Restore collected training data (prefer .npz, try common locations near checkpoint)
            chk_dir = os.path.dirname(checkpoint_path) or "."
            candidates = [
                os.path.join(chk_dir, "training_data.npz"),
                os.path.join(chk_dir, "training_data.pkl"),
                "training_data.npz",
                "training_data.pkl",
            ]
            loaded = False
            for cand in candidates:
                # Only load .npz; if .pkl path is given, look for the .npz with same stem
                load_path = cand
                if cand.endswith('.pkl'):
                    npz_path = cand.rsplit('.', 1)[0] + '.npz'
                    if os.path.exists(npz_path):
                        load_path = npz_path
                    else:
                        continue
                if os.path.exists(load_path) and load_path.endswith('.npz'):
                    try:
                        self.training_data.load_training_data(load_path)
                        loaded = True
                        break
                    except Exception:
                        pass
            if not loaded:
                print("Warning: Training data not found, starting with empty buffer.")

            # Restore model state
            if 'model_weights' in checkpoint_data:
                self.base_evaluator.weights = checkpoint_data['model_weights']

            # Restore Adam optimizer state
            adam_state = checkpoint_data.get('adam_state', {})
            if adam_state:
                self.base_evaluator.adam_m = adam_state.get('adam_m', {})
                self.base_evaluator.adam_v = adam_state.get('adam_v', {})
                self.base_evaluator.adam_t = adam_state.get('adam_t', 0)

            print(f"Training checkpoint loaded from: {checkpoint_path}")
            print(f"Resuming from game {self.games_played + 1} with {self.total_positions} collected positions")
            return True

        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return False

    def save_training_progress(self, model_save_path="data/neural_model.pkl", data_save_path="data/training_data.pkl", checkpoint_path="data/training_checkpoint.pkl"):
        """Save model, training data, and checkpoint after training chunk."""
        # Save updated model (full evaluator)
        model_dir = os.path.dirname(model_save_path)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
        with open(model_save_path, 'wb') as f:
            pickle.dump(self.base_evaluator, f)
        print(f"Model saved to: {model_save_path}")

        # Save training data (positions and scores)
        data_dir = os.path.dirname(data_save_path)
        if data_dir and not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
        data_save_path = data_save_path.rsplit('.', 1)[0] + '.npz'
        np.savez_compressed(data_save_path, positions=np.array(self.training_data.positions), scores=np.array(self.training_data.scores), allow_pickle=False)
        print(f"Training data saved to: {data_save_path}")

        # Save checkpoint (stats, partial data, model state)
        self.save_checkpoint(checkpoint_path)

    def print_training_stats(self) -> None:
        """Print current training statistics."""
        total_games: int = self.training_stats.black_wins + self.training_stats.red_wins + self.training_stats.draws

        if total_games == 0:
            print("No games played yet.")
            return

        print("TRAINING STATISTICS:")
        print(f"  Total games played: {total_games}")
        print(f"  Black wins: {self.training_stats.black_wins} ({100*self.training_stats.black_wins/total_games:.1f}%)")
        print(f"  Red wins: {self.training_stats.red_wins} ({100*self.training_stats.red_wins/total_games:.1f}%)")
        print(f"  Draws: {self.training_stats.draws} ({100*self.training_stats.draws/total_games:.1f}%)")
        print(f"  Average game length: {self.training_stats.avg_game_length:.1f} moves")
        print(f"  Total positions collected: {self.training_stats.positions_collected}")
        if total_games > 0:
            avg_random_per_game: float = self.training_stats.random_moves_total / total_games
            print(f"  Total random moves: {self.training_stats.random_moves_total}")
            print(f"  Average random moves per game: {avg_random_per_game:.1f}")
            print(f"  Current epsilon: {self.current_epsilon:.3f}")


class TrainingUI:
    """Simple command-line interface for training management."""

    def __init__(self):
        self.trainer = None
        self.is_training = False
        self.training_thread = None

    def show_menu(self):
        """Display the main training menu."""
        print("\n" + "="*60)
        print("           NEURAL CHECKERS SELF-PLAY TRAINER")
        print("="*60)
        print("1. Start New Training Session")
        print("2. Continue Training (load existing model)")
        print("3. View Training Statistics")
        print("4. Test Current Model vs Original")
        print("5. Export Training Data")
        print("6. Settings")
        print("0. Exit")
        print("-"*60)

    def start_training_session(self):
        """Start a new training session with user parameters."""
        try:
            num_games = int(input("Number of games to play (default 100): ") or "100")
            save_interval = int(input("Save progress every N games (default 25): ") or "25")

            model_path = input("Model save path (default 'neural_model.pkl'): ") or "neural_model.pkl"
            data_path = input("Data save path (default 'training_data.pkl'): ") or "training_data.pkl"

            print(f"\nStarting training session...")
            print(f"Games: {num_games}, Save interval: {save_interval}")

            # Initialize trainer
            self.trainer = SelfPlayTrainer()

            # Run in separate thread to allow interruption
            self.is_training = True
            self.training_thread = threading.Thread(
                target=self.trainer.run_training_session,
                args=(num_games, save_interval),
                kwargs={
                    "model_save_path": model_path,
                    "data_save_path": data_path,
                },
                daemon=True
            )
            self.training_thread.start()

            # Monitor training
            while self.is_training and self.training_thread.is_alive():
                try:
                    time.sleep(1)
                except KeyboardInterrupt:
                    print("\nTraining interrupted by user.")
                    self.is_training = False
                    break

            if not self.training_thread.is_alive():
                self.is_training = False
                print("Training completed successfully!")

        except ValueError:
            print("Invalid input. Please enter numeric values.")
        except Exception as e:
            print(f"Error during training: {e}")
            self.is_training = False

    def continue_training(self):
        """Continue training with an existing model."""
        model_path = input("Path to existing model (default 'neural_model.pkl'): ") or "neural_model.pkl"

        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return

        try:
            # Load existing model
            evaluator = NeuralEvaluator(model_path)
            self.trainer = SelfPlayTrainer(evaluator)

            print(f"Loaded model from {model_path}")
            self.start_training_session()

        except Exception as e:
            print(f"Error loading model: {e}")

    def view_stats(self):
        """View training statistics."""
        stats_path = input("Path to stats file (default 'training_data_stats.pkl'): ") or "training_data_stats.pkl"

        if not os.path.exists(stats_path):
            print(f"Stats file not found: {stats_path}")
            return

        try:
            with open(stats_path, 'rb') as f:
                stats = pickle.load(f)

            total_games = stats['black_wins'] + stats['red_wins'] + stats['draws']
            print("\nTRAINING STATISTICS:")
            print(f"Total games: {total_games}")
            print(f"Black wins: {stats['black_wins']} ({100*stats['black_wins']/total_games:.1f}%)")
            print(f"Red wins: {stats['red_wins']} ({100*stats['red_wins']/total_games:.1f}%)")
            print(f"Draws: {stats['draws']} ({100*stats['draws']/total_games:.1f}%)")
            print(f"Avg game length: {stats['avg_game_length']:.1f} moves")
            print(f"Positions collected: {stats['positions_collected']}")

        except Exception as e:
            print(f"Error loading stats: {e}")

    def run(self):
        """Main training UI loop."""
        while True:
            self.show_menu()

            try:
                choice = input("Select option: ").strip()

                if choice == '1':
                    self.start_training_session()
                elif choice == '2':
                    self.continue_training()
                elif choice == '3':
                    self.view_stats()
                elif choice == '4':
                    print("Model testing not implemented yet.")
                elif choice == '5':
                    print("Data export not implemented yet.")
                elif choice == '6':
                    print("Settings not implemented yet.")
                elif choice == '0':
                    if self.is_training:
                        print("Training in progress. Stop training first.")
                    else:
                        print("Goodbye!")
                        break
                else:
                    print("Invalid option. Please try again.")

            except KeyboardInterrupt:
                print("\nExiting...")
                if self.is_training:
                    self.is_training = False
                break
            except Exception as e:
                print(f"Error: {e}")


def integrate_training_button_to_gui():
    """
    Code to integrate training functionality into the existing GUI.
    Add this to your checkers_gui_tk.py file.
    (Note: The new checkers_gui_tk.py already integrates these features.)
    """
    training_button_code = '''
# Add this to your CheckersUI class __init__ method, in the buttons section:

self.btn_train = ttk.Button(btns, text="Train AI", command=self._start_training)
self.btn_train.grid(row=0, column=4, padx=5)

# Add these methods to your CheckersUI class:

def _start_training(self):
    """Start self-play training in a separate window."""
    import tkinter.simpledialog as simpledialog
    import tkinter.messagebox as messagebox

    # Get training parameters
    num_games = simpledialog.askinteger("Training Setup",
                                       "Number of games to play:",
                                       initialvalue=50,
                                       minvalue=1,
                                       maxvalue=1000)
    if not num_games:
        return

    # Start training in background
    self.training_active = True
    self.btn_train.configure(text="Training...", state="disabled")

    def run_training():
        try:
            trainer = SelfPlayTrainer()
            trainer.run_training_session(num_games, save_interval=10)

            # Update UI when done
            self.after(0, self._training_completed)

        except Exception as e:
            self.after(0, lambda: self._training_error(str(e)))

    import threading
    threading.Thread(target=run_training, daemon=True).start()

def _training_completed(self):
    """Called when training completes successfully."""
    self.training_active = False
    self.btn_train.configure(text="Train AI", state="normal")
    messagebox.showinfo("Training Complete",
                       "Self-play training completed successfully!\\n"
                       "Neural network has been updated with new data.")

def _training_error(self, error_msg):
    """Called when training encounters an error."""
    self.training_active = False
    self.btn_train.configure(text="Train AI", state="normal")
    messagebox.showerror("Training Error", f"Training failed: {error_msg}")
'''

    return training_button_code


def _play_single_game_worker(game_num, epsilon, base_evaluator, opponent_evaluator, shared_tt):
    return _play_single_game_core(game_num, epsilon, base_evaluator, opponent_evaluator, shared_tt, deepcopy_evaluators=True)

def _play_single_game_core(game_num, epsilon, base_evaluator, opponent_evaluator, shared_tt, deepcopy_evaluators=False):
    """Shared implementation used by both sequential and parallel runners."""
    from gameotherother import initial_board, legal_moves, apply_move, is_terminal, SearchEngine
    evaluator1 = deepcopy(base_evaluator) if deepcopy_evaluators else base_evaluator
    evaluator2 = deepcopy(opponent_evaluator) if deepcopy_evaluators else opponent_evaluator
    engine1 = SearchEngine(seed=random.randint(0, 10000), shared_tt=shared_tt)
    engine2 = SearchEngine(seed=random.randint(0, 10000), shared_tt=shared_tt)
    engine1.neural_evaluator = evaluator1
    engine2.neural_evaluator = evaluator2
    board = initial_board()
    player = 1  # Black starts
    move_count = 0
    random_moves_count = 0  # Track random moves in this game
    game_positions = []  # Store positions for this game
    game_start = time.time()
    while move_count < 200:
        if is_terminal(board, player):
            break
        # Store position before move
        game_positions.append((board[:], player))
        # Choose engine based on player
        current_engine = engine1 if player == 1 else engine2
        # Exploration: use current epsilon for random move probability
        depth = random.choice(DEFAULT_SEARCH_DEPTHS)  # Vary search depth
        try:
            if random.random() < epsilon:
                moves = legal_moves(board, player)
                best_move = random.choice(moves) if moves else None
                random_moves_count += 1  # Increment random moves counter
                _ = 0
            else:
                _, best_move = current_engine.search(board, player, depth)
        except Exception:
            # Fallback to random legal move if search fails
            moves = legal_moves(board, player)
            best_move = random.choice(moves) if moves else None
            random_moves_count += 1  # Count fallback as random move
        if best_move is None:
            break
        # Apply move
        board = apply_move(board, best_move)
        player = -player
        move_count += 1
    # Determine game result
    if is_terminal(board, player):
        # Current player to move has no legal moves - they lose
        game_result = -player  # Winner is the opposite player
    else:
        # Draw (max moves reached)
        game_result = 0
    # Convert game result to training targets
    game_positions_list = []
    for board_state, pos_player in game_positions:
        if game_result == 0:
            target_score = 0.0  # Draw
        else:
            target_score = 1.0 if pos_player == game_result else -1.0
        game_positions_list.append((board_state, pos_player, target_score))
    game_time = time.time() - game_start
    return game_result, len(game_positions), move_count, random_moves_count, game_positions_list, game_time


class ParallelGameRunner:
    def __init__(self, trainer, num_workers=4):
        self.trainer = trainer
        self.num_workers = num_workers
        self.shared_tt = Manager().dict()

        # Use spawn method for better compatibility and performance
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            # Already set, continue
            pass

        # Create pool with better initialization
        self.pool = Pool(self.num_workers)

    def run_games(self, num_games, save_interval, model_save_path, data_save_path, checkpoint_path, progress_callback=None):
        batch_size = self.num_workers * 5

        # Pre-calculate epsilon values for the entire batch to avoid repeated calculations
        epsilon_values = []
        for game_num in range(num_games):
            progress = min(1.0, game_num / max(1, self.trainer.epsilon_decay_games))
            epsilon = self.trainer.epsilon_start - (self.trainer.epsilon_start - self.trainer.epsilon_end) * progress
            epsilon_values.append(epsilon)

        for start in range(0, num_games, batch_size):
            end = min(start + batch_size, num_games)
            batch_params = []
            for game_num in range(start, end):
                opponent = self.trainer.create_opponent_evaluator()
                batch_params.append((game_num, epsilon_values[game_num], self.trainer.base_evaluator, opponent, self.shared_tt))
            batch_results = self.pool.starmap(_play_single_game_worker, batch_params)
            for i, res in enumerate(batch_results):
                game_num = start + i
                result, positions, moves, random_moves, game_positions_list, game_time = res
                self.trainer.training_stats['random_moves_total'] += random_moves
                self.trainer.training_stats['random_moves_per_game'].append(random_moves)
                if result == 1:
                    self.trainer.training_stats['black_wins'] += 1
                elif result == -1:
                    self.trainer.training_stats['red_wins'] += 1
                else:
                    self.trainer.training_stats['draws'] += 1
                self.trainer.games_played += 1
                self.trainer.total_positions += positions
                self.trainer.training_stats['positions_collected'] = self.trainer.total_positions
                total_games = self.trainer.training_stats['black_wins'] + self.trainer.training_stats['red_wins'] + self.trainer.training_stats['draws']
                if total_games > 0:
                    self.trainer.training_stats['avg_game_length'] = ((self.trainer.training_stats['avg_game_length'] * (total_games - 1)) + moves) / total_games
                for board_state, pos_player, target_score in game_positions_list:
                    self.trainer.training_data.add_position(board_state, pos_player, target_score)
                progress = min(1.0, game_num / max(1, self.trainer.epsilon_decay_games))
                epsilon_print = self.trainer.epsilon_start - (self.trainer.epsilon_start - self.trainer.epsilon_end) * progress
                if (game_num + 1) % 5 == 0 or game_num == 0:
                    result_str = "Black wins" if result == 1 else "Red wins" if result == -1 else "Draw"
                    avg_random_moves = sum(self.trainer.training_stats['random_moves_per_game'][-5:]) / min(5, len(self.trainer.training_stats['random_moves_per_game']))
                    print(f"Game {game_num + 1:3d}: {result_str:10s} | "
                          f"Moves: {moves:3d} | Positions: {positions:3d} | "
                          f"Random: {random_moves:2d} (ε={epsilon_print:.3f}) | "
                          f"Time: {game_time:.2f}s")
                if progress_callback:
                    progress_callback(game_num + 1, num_games)
                if (game_num + 1) % save_interval == 0:
                    stats = self.trainer.train_on_collected_data(epochs=2, batch_size=1024, lr=5e-4, l2=1e-6)
                    if stats and stats.get("loss") is not None:
                        print(f"Post-chunk training: MSE {stats['loss']:.4f} on {stats['count']} samples")
                    self.trainer.save_training_progress(model_save_path, data_save_path, checkpoint_path)
                    self.trainer.print_training_stats()
                    print("-" * 60)
                    self.trainer.training_data.clear()
        self.pool.close()
        self.pool.join()


if __name__ == "__main__":
    # Run the training UI
    ui = TrainingUI()
    ui.run()

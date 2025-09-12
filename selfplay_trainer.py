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

# Import the neural evaluator from the previous implementation
from neural_eval import NeuralEvaluator, TrainingDataCollector, get_neural_evaluator

class SelfPlayTrainer:
    """
    Self-play training system that generates training data by playing games
    between different versions of the neural network, and performs simple
    supervised updates to the base network.
    """
    
    def __init__(self, base_evaluator=None, resume_from_checkpoint=None):
        self.base_evaluator = base_evaluator or get_neural_evaluator()
        self.training_data = TrainingDataCollector(use_augmentation=True)
        self.games_played = 0
        self.total_positions = 0
        self.training_stats = {
            'black_wins': 0,
            'red_wins': 0,
            'draws': 0,
            'avg_game_length': 0,
            'positions_collected': 0
        }
        
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
    
    def play_self_play_game(self, max_moves=200, noise_temp=0.1):
        """
        Play a complete self-play game and collect training positions.
        
        Returns:
            tuple: (game_result, positions_collected, move_count)
        """
        from gameotherother import initial_board, legal_moves, apply_move, is_terminal, SearchEngine
        
        # Create two evaluators: base vs slightly mutated opponent
        evaluator1 = self.base_evaluator
        evaluator2 = self.create_opponent_evaluator()
        
        # Create search engines for both players and attach evaluators
        engine1 = SearchEngine(seed=random.randint(0, 10000))
        engine2 = SearchEngine(seed=random.randint(0, 10000))
        engine1.neural_evaluator = evaluator1
        engine2.neural_evaluator = evaluator2
        
        board = initial_board()
        player = 1  # Black starts
        move_count = 0
        game_positions = []  # Store positions for this game
        
        while move_count < max_moves:
            if is_terminal(board, player):
                break
                
            # Store position before move
            game_positions.append((board[:], player))
            
            # Choose engine based on player
            current_engine = engine1 if player == 1 else engine2
            
            # Exploration: 5% random move, otherwise search
            depth = random.choice([4, 5, 6])  # Vary search depth
            try:
                if random.random() < 0.05:
                    moves = legal_moves(board, player)
                    best_move = random.choice(moves) if moves else None
                    _ = 0
                else:
                    _, best_move = current_engine.search(board, player, depth)
            except Exception:
                # Fallback to random legal move if search fails
                moves = legal_moves(board, player)
                best_move = random.choice(moves) if moves else None
            
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
        for board_state, pos_player in game_positions:
            if game_result == 0:
                target_score = 0.0  # Draw
            else:
                target_score = 1.0 if pos_player == game_result else -1.0
            
            self.training_data.add_position(board_state, pos_player, target_score)
        
        # Update stats
        if game_result == 1:
            self.training_stats['black_wins'] += 1
        elif game_result == -1:
            self.training_stats['red_wins'] += 1
        else:
            self.training_stats['draws'] += 1
            
        self.games_played += 1
        positions_in_game = len(game_positions)
        self.total_positions += positions_in_game
        self.training_stats['positions_collected'] = self.total_positions
        
        # Update average game length
        total_games = self.training_stats['black_wins'] + self.training_stats['red_wins'] + self.training_stats['draws']
        if total_games > 0:
            self.training_stats['avg_game_length'] = ((self.training_stats['avg_game_length'] * (total_games - 1)) + move_count) / total_games
        
        return game_result, positions_in_game, move_count
    
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
    
    def run_training_session(self, num_games=100, save_interval=10,
                           model_save_path="neural_model.pkl",
                           data_save_path="training_data.pkl",
                           checkpoint_path="training_checkpoint.pkl",
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
        
        for game_num in range(num_games):
            game_start = time.time()
            result, positions, moves = self.play_self_play_game()
            game_time = time.time() - game_start
            
            # Print progress
            if (game_num + 1) % 5 == 0 or game_num == 0:
                result_str = "Black wins" if result == 1 else "Red wins" if result == -1 else "Draw"
                print(f"Game {game_num + 1:3d}: {result_str:10s} | "
                      f"Moves: {moves:3d} | Positions: {positions:3d} | "
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
        checkpoint_data = {
            'games_played': self.games_played,
            'total_positions': self.total_positions,
            'training_stats': self.training_stats,
            'collected_positions': np.array(self.training_data.positions) if self.training_data.positions else np.array([]),
            'collected_scores': np.array(self.training_data.scores) if self.training_data.scores else np.array([]),
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
            
            # Restore collected training data
            positions = checkpoint_data.get('collected_positions', np.array([]))
            scores = checkpoint_data.get('collected_scores', np.array([]))
            if len(positions) > 0:
                self.training_data.positions = positions.tolist()
                self.training_data.scores = scores.tolist()
            
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
    
    def save_training_progress(self, model_save_path="neural_model.pkl", data_save_path="training_data.pkl", checkpoint_path="training_checkpoint.pkl"):
        """Save model, training data, and checkpoint after training chunk."""
        # Save updated model (full evaluator)
        with open(model_save_path, 'wb') as f:
            pickle.dump(self.base_evaluator, f)
        print(f"Model saved to: {model_save_path}")
        
        # Save training data (positions and scores)
        data_to_save = {
            'positions': self.training_data.positions,
            'scores': self.training_data.scores
        }
        with open(data_save_path, 'wb') as f:
            pickle.dump(data_to_save, f)
        print(f"Training data saved to: {data_save_path}")
        
        # Save checkpoint (stats, partial data, model state)
        self.save_checkpoint(checkpoint_path)
    
    def print_training_stats(self):
        """Print current training statistics."""
        total_games = self.training_stats['black_wins'] + self.training_stats['red_wins'] + self.training_stats['draws']
        
        if total_games == 0:
            print("No games played yet.")
            return
        
        print("TRAINING STATISTICS:")
        print(f"  Total games played: {total_games}")
        print(f"  Black wins: {self.training_stats['black_wins']} ({100*self.training_stats['black_wins']/total_games:.1f}%)")
        print(f"  Red wins: {self.training_stats['red_wins']} ({100*self.training_stats['red_wins']/total_games:.1f}%)")
        print(f"  Draws: {self.training_stats['draws']} ({100*self.training_stats['draws']/total_games:.1f}%)")
        print(f"  Average game length: {self.training_stats['avg_game_length']:.1f} moves")
        print(f"  Total positions collected: {self.training_stats['positions_collected']}")


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
                args=(num_games, save_interval, model_path, data_path),
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


if __name__ == "__main__":
    # Run the training UI
    ui = TrainingUI()
    ui.run()
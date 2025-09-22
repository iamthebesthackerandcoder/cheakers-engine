# neural_eval.py
# Neural Network Evaluation Module for Checkers Engine
# Now includes a simple supervised training routine.

import numpy as np
import pickle
import os
import random
from copy import deepcopy
from typing import List, Tuple, Dict, Any, Optional, Union, Set

# Type aliases for better readability (avoiding circular imports)
Board = List[int]  # Board state: index 0 is unused, 1-32 are squares
Player = int  # 1 for black, -1 for red
Position = Tuple[int, int]  # (row, col) coordinates
SquareIndex = int  # 1-32 for dark squares
NeuralNetwork = Dict[str, np.ndarray]
TrainingData = Tuple[np.ndarray, np.ndarray]  # (features, targets)
EvaluationCache = Dict[str, float]
AdamState = Dict[str, Dict[str, np.ndarray]]  # Optimizer state
TrainingSample = Tuple[Board, Player, float]  # (board_state, player, target_value)
CacheStats = Dict[str, Union[int, float]]

# Optional imports for optimization
has_numba = False
try:
    import numba
    has_numba = True
except ImportError:
    pass

has_cupy = False
try:
    import cupy as cp
    has_cupy = True
except ImportError:
    pass

has_torch = False
try:
    import torch
    has_torch = True
except ImportError:
    pass

# Import board utilities from the main game module
try:
    from gameotherother import rc, idx_map, SQUARES
except ImportError:
    # Fallback definitions if import fails
    SQUARES = 32

    # Rebuild mappings locally if needed
    rc_of = [None] * (SQUARES + 1)
    idx_map = {}

    def build_mappings():
        i = 1
        for r in range(8):
            for c in range(8):
                if (r + c) % 2 == 1:
                    rc_of[i] = (r, c)
                    idx_map[(r, c)] = i
                    i += 1

    build_mappings()

    def rc(i: int) -> Optional[Tuple[int, int]]:
        return rc_of[i]


class NeuralEvaluator:
    """
    Optimized Neural network-based position evaluation for checkers.
    Vectorized forward pass, optional numba/GPU acceleration.
    Replaces the hand-crafted evaluation function.
    Also supports simple supervised training on self-play data.
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        """Initialize the neural evaluator with optional model loading."""
        self.model: Optional[Any] = None
        self.feature_size: int = 32 + 8  # 32 squares + 8 additional features

        # Optimization flags
        self.has_numba: bool = has_numba
        self.has_cupy: bool = has_cupy
        self.has_torch: bool = has_torch

        if self.has_numba:
            try:
                import numba
                self._board_to_features_fast = numba.jit(nopython=True)(self._board_to_features_fast)
            except Exception as e:
                print(f"Warning: Could not apply Numba JIT to _board_to_features_fast: {e}")

        if self.has_torch:
            try:
                import torch
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            except ImportError:
                pass

        self.directions: List[Tuple[int, int]] = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        self.adj_matrix: np.ndarray = np.full((32, 4), -1, dtype=int)
        for sq in range(32):
            pos: Optional[Tuple[int, int]] = rc(sq + 1)
            if pos is not None:
                r, c = pos
                for d, (dr, dc) in enumerate(self.directions):
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < 8 and 0 <= nc < 8 and (nr + nc) % 2 == 1:
                        if (nr, nc) in idx_map:
                            target_sq = idx_map[(nr, nc)] - 1
                            self.adj_matrix[sq, d] = target_sq

        # Adam optimizer state variables
        self.adam_m: Dict[str, np.ndarray] = {}  # First moment estimates
        self.adam_v: Dict[str, np.ndarray] = {}  # Second moment estimates
        self.adam_t: int = 0   # Time step counter

        # Position evaluation cache
        self.eval_cache: Dict[int, float] = {}  # hash(board_tuple, player) -> evaluation
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.max_cache_size: int = 10000  # Keep cache manageable

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            # Initialize with a simple placeholder network
            self._init_placeholder_network()

    def _init_placeholder_network(self) -> None:
        """Initialize a simple neural network as placeholder"""
        np.random.seed(42)
        self.weights = {
            'W1': (np.random.randn(self.feature_size, 128) * 0.1).astype(np.float32),
            'b1': np.zeros(128, dtype=np.float32),
            'W2': (np.random.randn(128, 64) * 0.1).astype(np.float32),
            'b2': np.zeros(64, dtype=np.float32),
            'W3': (np.random.randn(64, 32) * 0.1).astype(np.float32),
            'b3': np.zeros(32, dtype=np.float32),
            'W4': (np.random.randn(32, 1) * 0.1).astype(np.float32),
            'b4': np.zeros(1, dtype=np.float32)
        }
        self._init_adam_state()

    def board_to_features(self, board: Board, player: Player) -> np.ndarray:
        """
        Convert board position to neural network input features.

        Args:
            board: List representing the checkers board state
            player: Current player (1 for black, -1 for red)

        Returns:
            numpy array of features
        """
        return self.board_to_features_batch(np.array([board]), np.array([player]))[0]

    def _board_to_features_fast(self, boards: np.ndarray, players: np.ndarray) -> np.ndarray:
        """Optimized feature extraction using NumPy (JIT fallback if available)."""
        N = len(boards)
        features = np.zeros((N, self.feature_size), dtype=np.float32)

        board_states = boards[:, 1:33]
        features[:, :32] = board_states

        # Vectorized piece counting - much faster than individual sums
        features[:, 32] = np.sum(board_states == 1, axis=1) / 12.0  # black men
        features[:, 33] = np.sum(board_states == 2, axis=1) / 12.0  # black kings
        features[:, 34] = np.sum(board_states == -1, axis=1) / 12.0 # red men
        features[:, 35] = np.sum(board_states == -2, axis=1) / 12.0 # red kings
        features[:, 36] = np.sum(board_states != 0, axis=1) / 24.0  # total pieces
        features[:, 37] = players  # Current player to move

        # Vectorized mobility calculation for better performance
        features[:, 38] = self._calculate_mobility_batch(board_states, 1)  # Black
        features[:, 39] = self._calculate_mobility_batch(board_states, -1) # Red

        return features

    def board_to_features_batch(self, boards: Union[List[Board], np.ndarray],
                               players: Union[List[Player], np.ndarray]) -> np.ndarray:
        """
        Vectorized feature extraction for batch of boards.

        Args:
            boards: np.ndarray (N, 33) or list of boards
            players: np.ndarray (N,) of players

        Returns:
            np.ndarray (N, feature_size)
        """
        if isinstance(boards, list):
            boards = np.array(boards, dtype=np.float32)
        if isinstance(players, list):
            players = np.array(players, dtype=np.float32)

        features = self._board_to_features_fast(boards, players)
        return features

    def _board_to_features_batch_slow(self, boards: np.ndarray, players: np.ndarray) -> np.ndarray:
        # Fallback without numba
        N = len(boards)
        features = np.zeros((N, self.feature_size), dtype=np.float32)
        # ... same as above
        board_states = boards[:, 1:33]
        features[:, :32] = board_states
        # etc., copy the code above
        features[:, 32] = np.sum(board_states == 1, axis=1) / 12.0
        features[:, 33] = np.sum(board_states == 2, axis=1) / 12.0
        features[:, 34] = np.sum(board_states == -1, axis=1) / 12.0
        features[:, 35] = np.sum(board_states == -2, axis=1) / 12.0
        features[:, 36] = np.sum(board_states != 0, axis=1) / 24.0
        features[:, 37] = players
        for i in range(N):
            features[i, 38] = self._calculate_mobility_feature(board_states[i], 1)
            features[i, 39] = self._calculate_mobility_feature(board_states[i], -1)
        return features

    def augment_position(self, board: Board, player: Player) -> Tuple[Board, Player]:
        """
        Create data augmentation by flipping the board horizontally.

        Args:
            board: Original board state
            player: Current player

        Returns:
            tuple: (flipped_board, flipped_player)
        """
        flipped_board: Board = [0] * len(board)

        flip_map: Dict[int, int] = {}
        for i in range(1, 33):
            pos: Optional[Tuple[int, int]] = rc(i)
            if pos is not None:
                r, c = pos
                flipped_c = 7 - c
                if (r, flipped_c) in idx_map:
                    flip_map[i] = idx_map[(r, flipped_c)]

        for i in range(1, 33):
            if i in flip_map:
                flipped_board[flip_map[i]] = board[i]

        return flipped_board, player

    def _get_position_hash(self, board: List[int], player: int) -> int:
        """Create a hash key for position caching"""
        # Convert board to tuple (hashable) and combine with player
        board_tuple = tuple(board)
        return hash((board_tuple, player))

    def _manage_cache_size(self) -> None:
        """Keep cache size under control by removing oldest entries"""
        if len(self.eval_cache) > self.max_cache_size:
            # Remove 20% of oldest entries (simple FIFO approach)
            items_to_remove = len(self.eval_cache) // 5
            keys_to_remove = list(self.eval_cache.keys())[:items_to_remove]
            for key in keys_to_remove:
                del self.eval_cache[key]
                # Also clean up corresponding Adam state if it exists
                if hasattr(self, 'adam_m') and key in self.adam_m:
                    del self.adam_m[key]
                if hasattr(self, 'adam_v') and key in self.adam_v:
                    del self.adam_v[key]

    def get_cache_stats(self) -> Dict[str, Union[int, float]]:
        """Get cache performance statistics"""
        total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total * 100) if total > 0 else 0
        return {
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.eval_cache)
        }

    def clear_cache(self) -> None:
        """Clear the evaluation cache"""
        self.eval_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0

    def _calculate_mobility_feature(self, board_np: np.ndarray, color: int) -> float:
        """Calculate a simple mobility feature for the neural network"""
        own = (board_np * color > 0).astype(np.float32)
        is_king = (np.abs(board_np) == 2).astype(np.float32)
        is_man = 1.0 - is_king

        empty_adj = np.zeros((32, 4), dtype=np.float32)
        for d in range(4):
            mask = self.adj_matrix[:, d] != -1
            if np.any(mask):
                targets = self.adj_matrix[:, d][mask]
                empty_adj[mask, d] = (board_np[targets] == 0).astype(np.float32)

        total_adj = np.sum(empty_adj, axis=1)
        kings_mobility = np.sum(total_adj * own * is_king)

        if color == 1:
            man_slice = slice(0, 2)
        else:
            man_slice = slice(2, 4)
        men_adj = np.sum(empty_adj[:, man_slice], axis=1)
        men_mobility = np.sum(men_adj * own * is_man)

        mobility = (kings_mobility + men_mobility) / 20.0
        return mobility

    def _calculate_mobility_batch(self, board_states: np.ndarray, color: int) -> np.ndarray:
        """Vectorized mobility calculation for batch of boards."""
        N = len(board_states)
        mobility_scores = np.zeros(N, dtype=np.float32)

        for i in range(N):
            mobility_scores[i] = self._calculate_mobility_feature(board_states[i], color)

        return mobility_scores



    def predict(self, features: np.ndarray) -> float:
        """
        Forward pass for single input.
        """
        return float(self.predict_batch(features.reshape(1, -1))[0])

    def _forward_batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Forward pass for a batch. Supports GPU if available."""
        if self.has_cupy:
            X = cp.asarray(X)
            z1 = X @ cp.asarray(self.weights['W1']) + cp.asarray(self.weights['b1'])
            a1 = cp.maximum(z1, 0)  # ReLU, faster than boolean
            z2 = a1 @ cp.asarray(self.weights['W2']) + cp.asarray(self.weights['b2'])
            a2 = cp.maximum(z2, 0)
            z3 = a2 @ cp.asarray(self.weights['W3']) + cp.asarray(self.weights['b3'])
            a3 = cp.maximum(z3, 0)
            z4 = a3 @ cp.asarray(self.weights['W4']) + cp.asarray(self.weights['b4'])
            y = cp.tanh(z4) * 1000.0
            return [cp.asnumpy(arr) for arr in (z1, a1, z2, a2, z3, a3, z4, y)]
        elif self.has_torch:
            # Fallback to numpy for incomplete torch implementation to avoid dtype errors
            return self._forward_batch_numpy(X)
        else:
            # Numpy
            z1 = X @ self.weights['W1'] + self.weights['b1']
            a1 = np.maximum(z1, 0)  # ReLU
            z2 = a1 @ self.weights['W2'] + self.weights['b2']
            a2 = np.maximum(z2, 0)
            z3 = a2 @ self.weights['W3'] + self.weights['b3']
            a3 = np.maximum(z3, 0)
            z4 = a3 @ self.weights['W4'] + self.weights['b4']
            y = np.tanh(z4) * 1000.0
            return z1, a1, z2, a2, z3, a3, z4, y

    def _forward_batch_numpy(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        z1 = X @ self.weights['W1'] + self.weights['b1']
        a1 = np.maximum(z1, 0)
        z2 = a1 @ self.weights['W2'] + self.weights['b2']
        a2 = np.maximum(z2, 0)
        z3 = a2 @ self.weights['W3'] + self.weights['b3']
        a3 = np.maximum(z3, 0)
        z4 = a3 @ self.weights['W4'] + self.weights['b4']
        y = np.tanh(z4) * 1000.0
        return z1, a1, z2, a2, z3, a3, z4, y

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Predict for a batch of features"""
        if self.has_cupy:
            _, _, _, _, _, _, _, y = self._forward_batch(cp.asarray(X))
            return cp.asnumpy(y).reshape(-1)
        elif self.has_torch:
            # Simplified fallback
            _, _, _, _, _, _, _, y = self._forward_batch(X)
            return y.reshape(-1)
        else:
            _, _, _, _, _, _, _, y = self._forward_batch(X)
            return y.reshape(-1)

    def batch_predict(self, boards: Union[List[Board], np.ndarray],
                     players: Union[List[Player], np.ndarray]) -> np.ndarray:
        """Batch evaluation from boards"""
        features: np.ndarray = self.board_to_features_batch(boards, players)
        return self.predict_batch(features)

    def train_supervised(self, positions: Optional[np.ndarray], targets: Optional[np.ndarray],
                        epochs: int = 1, batch_size: int = 1024, lr: float = 5e-4,
                        l2: float = 1e-6, shuffle: bool = True, verbose: bool = True) -> Dict[str, Union[float, int]]:
        """
        Simple supervised training using MSE loss.
        Default lr=5e-4 as per task.
        """
        if positions is None or targets is None or len(positions) == 0:
            return {"loss": 0.0, "count": 0}

        X: np.ndarray = positions.astype(np.float32)
        t: np.ndarray = targets.astype(np.float32).reshape(-1, 1) * 1000.0
        N: int = X.shape[0]

        indices = np.arange(N)
        np.random.shuffle(indices)
        split_idx = int(0.8 * N)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]

        # Pre-allocate arrays for better memory efficiency
        avg_loss = 0.0
        val_loss = 0.0
        total_count = 0

        for epoch in range(epochs):
            if shuffle:
                np.random.shuffle(train_indices)
            total_loss = 0.0
            total_count = 0

            # Process training data in batches
            for start in range(0, len(train_indices), batch_size):
                idx = train_indices[start:start+batch_size]
                xb = X[idx]
                tb = t[idx]

                # Forward (GPU fallback handled in _forward_batch)
                z1, a1, z2, a2, z3, a3, z4, y = self._forward_batch(xb)
                diff = y - tb
                loss = float(np.mean(diff**2))
                total_loss += loss * len(idx)
                total_count += len(idx)

                # Backprop (numpy for gradients, as GPU weights need sync)
                # Assume numpy for backprop simplicity; full GPU training if needed later
                tanh_z4 = np.tanh(z4)
                dy_dz4 = 1000.0 * (1.0 - tanh_z4**2)
                dL_dy = (2.0 / len(idx)) * diff
                g4 = dL_dy * dy_dz4

                dW4 = a3.T @ g4 + l2 * self.weights['W4']
                db4 = np.sum(g4, axis=0)

                ga3 = g4 @ self.weights['W4'].T
                gz3 = ga3 * (z3 > 0)

                dW3 = a2.T @ gz3 + l2 * self.weights['W3']
                db3 = np.sum(gz3, axis=0)

                ga2 = gz3 @ self.weights['W3'].T
                gz2 = ga2 * (z2 > 0)

                dW2 = a1.T @ gz2 + l2 * self.weights['W2']
                db2 = np.sum(gz2, axis=0)

                ga1 = gz2 @ self.weights['W2'].T
                gz1 = ga1 * (z1 > 0)

                dW1 = xb.T @ gz1 + l2 * self.weights['W1']
                db1 = np.sum(gz1, axis=0)

                gradients = {
                    'W4': dW4, 'b4': db4,
                    'W3': dW3, 'b3': db3,
                    'W2': dW2, 'b2': db2,
                    'W1': dW1, 'b1': db1
                }
                self._adam_update(gradients, lr)

            # Compute validation loss
            val_loss = 0.0
            val_count = len(val_indices)
            if val_count > 0:
                val_xb = X[val_indices]
                val_tb = t[val_indices]
                _, _, _, _, _, _, _, val_y = self._forward_batch(val_xb)
                val_diff = val_y - val_tb
                val_loss = float(np.mean(val_diff**2))

            avg_loss = total_loss / max(1, total_count)

            if verbose:
                print(f"[Neural Train] Epoch {epoch+1}/{epochs} - Train MSE: {avg_loss:.4f} | Val MSE: {val_loss:.4f} over {total_count} train / {val_count} val samples")

        return {"train_loss": avg_loss, "val_loss": val_loss, "count": total_count}

    def evaluate_position(self, board: Board, player: Player) -> float:
        """
        Main evaluation function with caching.
        """
        cache_key: int = self._get_position_hash(board, player)
        if cache_key in self.eval_cache:
            self.cache_hits += 1
            return self.eval_cache[cache_key]

        self.cache_misses += 1

        features: np.ndarray = self.board_to_features(board, player)
        raw_score: float = self.predict(features)
        result: float = player * raw_score

        self.eval_cache[cache_key] = result
        self._manage_cache_size()

        return result

    def _init_adam_state(self) -> None:
        """Initialize Adam optimizer state variables"""
        self.adam_m = {}
        self.adam_v = {}
        for key in self.weights:
            self.adam_m[key] = np.zeros_like(self.weights[key])
            self.adam_v[key] = np.zeros_like(self.weights[key])
        self.adam_t = 0

    def _adam_update(self, gradients: Dict[str, np.ndarray], lr: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        """Apply Adam optimizer update"""
        self.adam_t += 1

        for key in gradients:
            # Update biased first moment estimate
            self.adam_m[key] = beta1 * self.adam_m[key] + (1 - beta1) * gradients[key]

            # Update biased second raw moment estimate
            self.adam_v[key] = beta2 * self.adam_v[key] + (1 - beta2) * (gradients[key] ** 2)

            # Compute bias-corrected first moment estimate
            m_hat = self.adam_m[key] / (1 - beta1 ** self.adam_t)

            # Compute bias-corrected second raw moment estimate
            v_hat = self.adam_v[key] / (1 - beta2 ** self.adam_t)

            # Update parameters
            self.weights[key] -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def save_model(self, filepath: str) -> None:
        """Save the neural network weights and Adam state"""
        model_data = {
            'weights': self.weights,
            'adam_m': self.adam_m,
            'adam_v': self.adam_v,
            'adam_t': self.adam_t
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Neural model saved to: {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load neural network weights and Adam state"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        if isinstance(data, dict) and 'weights' in data:
            # New format with Adam state
            self.weights = data['weights']
            for k in self.weights:
                self.weights[k] = self.weights[k].astype(np.float32)
            self.adam_m = data.get('adam_m', {})
            self.adam_v = data.get('adam_v', {})
            self.adam_t = data.get('adam_t', 0)

            # Initialize Adam state if missing or incomplete
            if not self.adam_m or not self.adam_v:
                self._init_adam_state()
        else:
            # Old format - just weights
            self.weights = data
            for k in self.weights:
                self.weights[k] = self.weights[k].astype(np.float32)
            self._init_adam_state()

        print(f"Neural model loaded from: {filepath}")


# Global neural evaluator instance
_NEURAL_EVALUATOR: Optional[NeuralEvaluator] = None

def get_neural_evaluator() -> NeuralEvaluator:
    """Get or create the neural evaluator singleton"""
    global _NEURAL_EVALUATOR
    if _NEURAL_EVALUATOR is None:
        _NEURAL_EVALUATOR = NeuralEvaluator()
    return _NEURAL_EVALUATOR


# Modified evaluation function that uses neural network
def evaluate_neural(board: Board, player: Player) -> int:
    """
    Neural network-based evaluation function.
    This replaces the original evaluate() function.
    """
    evaluator: NeuralEvaluator = get_neural_evaluator()
    return int(evaluator.evaluate_position(board, player))


# Training data collection helper (for future training)
class TrainingDataCollector:
    """Helper class to collect training data from games"""

    def __init__(self, use_augmentation: bool = True) -> None:
        self.positions: List[np.ndarray] = []
        self.scores: List[float] = []
        self.use_augmentation = use_augmentation  # But will use 50% freq

    def add_position(self, board: Board, player: Player, game_result: float) -> None:
        """
        Add a position with 50% chance of augmentation.
        """
        evaluator: NeuralEvaluator = get_neural_evaluator()

        # Original
        features: np.ndarray = evaluator.board_to_features(board, player)
        score: float = player * game_result

        self.positions.append(features)
        self.scores.append(score)

        # 50% augmentation
        if self.use_augmentation and random.random() < 0.5:
            flipped_board, flipped_player = evaluator.augment_position(board, player)
            flipped_features: np.ndarray = evaluator.board_to_features(flipped_board, flipped_player)
            flipped_score: float = flipped_player * game_result

            self.positions.append(flipped_features)
            self.scores.append(flipped_score)

    def save_training_data(self, filepath: str) -> None:
        """Save to npz compressed format"""
        if filepath.endswith('.pkl'):
            filepath = filepath.replace('.pkl', '.npz')
        positions_np = np.array(self.positions, dtype=np.float32)
        scores_np = np.array(self.scores, dtype=np.float32)
        np.savez_compressed(filepath, positions=positions_np, scores=scores_np)
        print(f"Saved {len(self.positions)} positions to {filepath}")

    def load_training_data(self, filepath: str) -> None:
        """Load from npz only"""
        if not filepath.endswith('.npz'):
            raise ValueError(f"Only .npz format supported: {filepath}")
        data = np.load(filepath, allow_pickle=False)
        self.positions = list(data['positions'])
        self.scores = list(data['scores'])
        print(f"Loaded {len(self.positions)} positions from {filepath}")

    def clear(self) -> None:
        """Clear collected data"""
        self.positions.clear()
        self.scores.clear()


# Integration function
def integrate_neural_evaluation() -> bool:
    """
    Integrate optimized neural evaluation.
    """
    try:
        import gameotherother
        gameotherother.evaluate = evaluate_neural
        print("Optimized neural evaluation activated!")
        return True
    except ImportError:
        print("Could not integrate neural evaluation")
        return False


# Example usage and testing
if __name__ == "__main__":
    # Test the neural evaluator
    evaluator: NeuralEvaluator = get_neural_evaluator()

    # Create a test board (initial position)
    test_board: Board = [0] * 33  # 33 elements (index 0 unused)

    # Set up initial position
    for i in range(1, 33):
        pos: Optional[Tuple[int, int]] = rc(i)
        if pos is not None:
            r, c = pos
            if r <= 2:
                test_board[i] = -1  # Red pieces
            elif r >= 5:
                test_board[i] = 1   # Black pieces

    # Test evaluation
    score: float = evaluator.evaluate_position(test_board, 1)
    print(f"Neural evaluation of initial position: {score}")

    # Test feature extraction
    features: np.ndarray = evaluator.board_to_features(test_board, 1)
    print(f"Feature vector shape: {features.shape}")
    print(f"Sample features: {features[:10]}")


# Example usage and testing
if __name__ == "__main__":
    # Test the neural evaluator
    evaluator = get_neural_evaluator()

    # Create a test board (initial position)
    test_board = [0] * 33  # 33 elements (index 0 unused)

    # Set up initial position
    for i in range(1, 33):
        r, c = rc(i)
        if r <= 2:
            test_board[i] = -1  # Red pieces
        elif r >= 5:
            test_board[i] = 1   # Black pieces

    # Test evaluation
    score = evaluator.evaluate_position(test_board, 1)
    print(f"Neural evaluation of initial position: {score}")

    # Test feature extraction
    features = evaluator.board_to_features(test_board, 1)
    print(f"Feature vector shape: {features.shape}")
    print(f"Sample features: {features[:10]}")

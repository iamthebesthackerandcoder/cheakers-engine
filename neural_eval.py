# neural_eval.py
# Neural Network Evaluation Module for Checkers Engine
# Now includes a simple supervised training routine.

import numpy as np
import pickle
import os
import random
from copy import deepcopy

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
    
    def rc(i):
        return rc_of[i]


class NeuralEvaluator:
    """
    Neural network-based position evaluation for checkers.
    Replaces the hand-crafted evaluation function.
    Also supports simple supervised training on self-play data.
    """
    
    def __init__(self, model_path=None):
        self.model = None
        self.feature_size = 32 + 8  # 32 squares + 8 additional features
        
        self.directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        self.adj_matrix = np.full((32, 4), -1, dtype=int)
        for sq in range(32):
            r, c = rc(sq + 1)
            for d, (dr, dc) in enumerate(self.directions):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < 8 and 0 <= nc < 8 and (nr + nc) % 2 == 1:
                    if (nr, nc) in idx_map:
                        target_sq = idx_map[(nr, nc)] - 1
                        self.adj_matrix[sq, d] = target_sq
        
        # Adam optimizer state variables
        self.adam_m = {}  # First moment estimates
        self.adam_v = {}  # Second moment estimates
        self.adam_t = 0   # Time step counter
        
        # Position evaluation cache
        self.eval_cache = {}  # hash(board_tuple, player) -> evaluation
        self.cache_hits = 0
        self.cache_misses = 0
        self.max_cache_size = 10000  # Keep cache manageable
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            # Initialize with a simple placeholder network
            self._init_placeholder_network()
    
    def _init_placeholder_network(self):
        """Initialize a simple neural network as placeholder"""
        np.random.seed(42)
        self.weights = {
            'W1': np.random.randn(self.feature_size, 128) * 0.1,
            'b1': np.zeros(128, dtype=np.float32),
            'W2': np.random.randn(128, 64) * 0.1,
            'b2': np.zeros(64, dtype=np.float32),
            'W3': np.random.randn(64, 32) * 0.1,
            'b3': np.zeros(32, dtype=np.float32),
            'W4': np.random.randn(32, 1) * 0.1,
            'b4': np.zeros(1, dtype=np.float32)
        }
        self._init_adam_state()
    
    def board_to_features(self, board, player):
        """
        Convert board position to neural network input features.
        
        Args:
            board: List representing the checkers board state
            player: Current player (1 for black, -1 for red)
            
        Returns:
            numpy array of features
        """
        features = np.zeros(self.feature_size, dtype=np.float32)
        
        board_np = np.array(board[1:33], dtype=np.float32)
        features[:32] = board_np
        
        # Piece counts
        features[32] = np.sum(board_np == 1) / 12.0   # black men
        features[33] = np.sum(board_np == 2) / 12.0   # black kings
        features[34] = np.sum(board_np == -1) / 12.0  # red men
        features[35] = np.sum(board_np == -2) / 12.0  # red kings
        features[36] = np.count_nonzero(board_np) / 24.0  # total pieces
        features[37] = player                  # Current player to move
        
        # Simple mobility features
        features[38] = self._calculate_mobility_feature(board_np, 1)   # Black mobility
        features[39] = self._calculate_mobility_feature(board_np, -1)  # Red mobility
        
        return features
    
    def augment_position(self, board, player):
        """
        Create data augmentation by flipping the board horizontally.
        This doubles the effective training data for free.
        
        Args:
            board: Original board state
            player: Current player
            
        Returns:
            tuple: (flipped_board, flipped_player) - the horizontally flipped position
        """
        flipped_board = [0] * len(board)
        
        # Horizontal flip mapping for checkers squares
        # Column mapping: 0->7, 1->6, 2->5, 3->4, 4->3, 5->2, 6->1, 7->0
        flip_map = {}
        for i in range(1, 33):
            r, c = rc(i)
            flipped_c = 7 - c
            if (r, flipped_c) in idx_map:
                flip_map[i] = idx_map[(r, flipped_c)]
        
        # Copy flipped pieces
        for i in range(1, 33):
            if i in flip_map:
                flipped_board[flip_map[i]] = board[i]
        
        return flipped_board, player
    
    def _get_position_hash(self, board, player):
        """Create a hash key for position caching"""
        # Convert board to tuple (hashable) and combine with player
        board_tuple = tuple(board)
        return hash((board_tuple, player))
    
    def _manage_cache_size(self):
        """Keep cache size under control by removing oldest entries"""
        if len(self.eval_cache) > self.max_cache_size:
            # Remove 20% of oldest entries (simple FIFO approach)
            items_to_remove = len(self.eval_cache) // 5
            keys_to_remove = list(self.eval_cache.keys())[:items_to_remove]
            for key in keys_to_remove:
                del self.eval_cache[key]
    
    def get_cache_stats(self):
        """Get cache performance statistics"""
        total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total * 100) if total > 0 else 0
        return {
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.eval_cache)
        }
    
    def clear_cache(self):
        """Clear the evaluation cache"""
        self.eval_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _calculate_mobility_feature(self, board_np, color):
        """Calculate a simple mobility feature for the neural network"""
        own = (board_np * color > 0).astype(np.float32)
        is_king = (np.abs(board_np) == 2).astype(np.float32)
        is_man = 1.0 - is_king
        
        # Compute empty adjacent squares for all directions
        empty_adj = np.zeros((32, 4), dtype=np.float32)
        for d in range(4):
            mask = self.adj_matrix[:, d] != -1
            targets = self.adj_matrix[:, d][mask]
            empty_adj[mask, d] = (board_np[targets] == 0).astype(np.float32)
        
        total_adj = np.sum(empty_adj, axis=1)
        
        # Kings mobility (all directions)
        kings_mobility = np.sum(total_adj * own * is_king)
        
        # Men mobility (forward directions only)
        if color == 1:
            man_slice = slice(0, 2)
        else:
            man_slice = slice(2, 4)
        men_adj = np.sum(empty_adj[:, man_slice], axis=1)
        men_mobility = np.sum(men_adj * own * is_man)
        
        mobility = (kings_mobility + men_mobility) / 20.0
        return mobility
    
    
    
    def predict(self, features):
        """
        Forward pass through the neural network for a single feature vector.
        
        Args:
            features: Input feature vector (shape (feature_size,))
            
        Returns:
            Evaluation score (float), roughly in [-1000, 1000]
        """
        # Layer 1
        z1 = np.dot(features, self.weights['W1']) + self.weights['b1']
        a1 = z1 * (z1 > 0)
        
        # Layer 2
        z2 = np.dot(a1, self.weights['W2']) + self.weights['b2']
        a2 = z2 * (z2 > 0)
        
        # Layer 3
        z3 = np.dot(a2, self.weights['W3']) + self.weights['b3']
        a3 = z3 * (z3 > 0)
        
        # Output layer
        z4 = np.dot(a3, self.weights['W4']) + self.weights['b4']
        output = np.tanh(z4) * 1000  # Scale to reasonable evaluation range
        
        return float(output[0])
    
    def _forward_batch(self, X):
        """Forward pass for a batch. Returns intermediates for backprop."""
        z1 = X @ self.weights['W1'] + self.weights['b1']
        a1 = (z1 > 0) * z1  # ReLU
        z2 = a1 @ self.weights['W2'] + self.weights['b2']
        a2 = (z2 > 0) * z2  # ReLU
        z3 = a2 @ self.weights['W3'] + self.weights['b3']
        a3 = (z3 > 0) * z3  # ReLU
        z4 = a3 @ self.weights['W4'] + self.weights['b4']
        y = np.tanh(z4) * 1000.0
        return z1, a1, z2, a2, z3, a3, z4, y
    
    def predict_batch(self, X):
        """Predict for a batch of features"""
        _, _, _, _, _, _, _, y = self._forward_batch(X)
        return y.reshape(-1)
    
    def train_supervised(self, positions, targets, epochs=1, batch_size=1024, lr=2e-4, l2=1e-6, shuffle=True, verbose=True):
        """
        Simple supervised training using MSE loss between network output and target score.
        targets are expected in [-1, 1] (game outcome from player's perspective), and we scale by 1000.

        Args:
            positions: np.ndarray shape (N, feature_size)
            targets: np.ndarray shape (N,) or (N,1) in [-1,1]
        """
        if positions is None or len(positions) == 0:
            return {"loss": None, "count": 0}
        
        X = positions.astype(np.float32)
        t = targets.astype(np.float32).reshape(-1, 1) * 1000.0  # scale to [-1000, 1000]
        N = X.shape[0]

        order = np.arange(N)

        for epoch in range(epochs):
            if shuffle:
                np.random.shuffle(order)
            total_loss = 0.0
            total_count = 0

            for start in range(0, N, batch_size):
                idx = order[start:start+batch_size]
                xb = X[idx]
                tb = t[idx]

                # Forward
                z1, a1, z2, a2, z3, a3, z4, y = self._forward_batch(xb)
                diff = (y - tb)
                loss = float(np.mean(diff**2))
                total_loss += loss * len(idx)
                total_count += len(idx)

                # Backprop
                # y = 1000 * tanh(z4) => dy/dz4 = 1000 * (1 - tanh(z4)^2)
                tanh_z4 = np.tanh(z4)
                dy_dz4 = 1000.0 * (1.0 - tanh_z4**2)
                dL_dy = (2.0 / len(idx)) * diff  # dL/dy
                g4 = dL_dy * dy_dz4  # dL/dz4

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

                # Adam update
                gradients = {
                    'W4': dW4, 'b4': db4,
                    'W3': dW3, 'b3': db3, 
                    'W2': dW2, 'b2': db2,
                    'W1': dW1, 'b1': db1
                }
                self._adam_update(gradients, lr)

            if verbose:
                avg_loss = total_loss / max(1, total_count)
                print(f"[Neural Train] Epoch {epoch+1}/{epochs} - Avg MSE: {avg_loss:.4f} over {total_count} samples")

        return {"loss": total_loss / max(1, total_count), "count": total_count}
    
    def evaluate_position(self, board, player):
        """
        Main evaluation function that replaces the hand-crafted eval.
        Now includes caching for improved performance.
        
        Args:
            board: Board state
            player: Player to move
            
        Returns:
            Evaluation score from player's perspective (float)
        """
        # Check cache first
        cache_key = self._get_position_hash(board, player)
        if cache_key in self.eval_cache:
            self.cache_hits += 1
            return self.eval_cache[cache_key]
        
        # Cache miss - compute evaluation
        self.cache_misses += 1
        
        features = self.board_to_features(board, player)
        raw_score = self.predict(features)
        # Return score from current player's perspective
        result = player * raw_score
        
        # Store in cache
        self.eval_cache[cache_key] = result
        self._manage_cache_size()
        
        return result
    
    def _init_adam_state(self):
        """Initialize Adam optimizer state variables"""
        self.adam_m = {}
        self.adam_v = {}
        for key in self.weights:
            self.adam_m[key] = np.zeros_like(self.weights[key])
            self.adam_v[key] = np.zeros_like(self.weights[key])
        self.adam_t = 0
    
    def _adam_update(self, gradients, lr, beta1=0.9, beta2=0.999, eps=1e-8):
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
    
    def save_model(self, filepath):
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
    
    def load_model(self, filepath):
        """Load neural network weights and Adam state"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, dict) and 'weights' in data:
            # New format with Adam state
            self.weights = data['weights']
            self.adam_m = data.get('adam_m', {})
            self.adam_v = data.get('adam_v', {})
            self.adam_t = data.get('adam_t', 0)
            
            # Initialize Adam state if missing or incomplete
            if not self.adam_m or not self.adam_v:
                self._init_adam_state()
        else:
            # Old format - just weights
            self.weights = data
            self._init_adam_state()
            
        print(f"Neural model loaded from: {filepath}")


# Global neural evaluator instance
_NEURAL_EVALUATOR = None

def get_neural_evaluator():
    """Get or create the neural evaluator singleton"""
    global _NEURAL_EVALUATOR
    if _NEURAL_EVALUATOR is None:
        _NEURAL_EVALUATOR = NeuralEvaluator()
    return _NEURAL_EVALUATOR


# Modified evaluation function that uses neural network
def evaluate_neural(board, player):
    """
    Neural network-based evaluation function.
    This replaces the original evaluate() function.
    """
    evaluator = get_neural_evaluator()
    return int(evaluator.evaluate_position(board, player))


# Training data collection helper (for future training)
class TrainingDataCollector:
    """Helper class to collect training data from games"""
    
    def __init__(self, use_augmentation=True):
        self.positions = []
        self.scores = []
        self.use_augmentation = use_augmentation
    
    def add_position(self, board, player, game_result):
        """
        Add a position with its eventual game outcome.
        
        Args:
            board: Board state
            player: Player to move
            game_result: Final game result (1.0 for black win, -1.0 for red win, 0.0 for draw)
        """
        evaluator = get_neural_evaluator()
        
        # Original position
        features = evaluator.board_to_features(board, player)
        score = player * game_result  # Score from current player's perspective in [-1, 0, 1]
        
        self.positions.append(features)
        self.scores.append(score)
        
        # Add augmented position if enabled
        if self.use_augmentation:
            flipped_board, flipped_player = evaluator.augment_position(board, player)
            flipped_features = evaluator.board_to_features(flipped_board, flipped_player)
            flipped_score = flipped_player * game_result
            
            self.positions.append(flipped_features)
            self.scores.append(flipped_score)
    
    def save_training_data(self, filepath):
        """Save collected training data"""
        data = {
            'positions': np.array(self.positions, dtype=np.float32),
            'scores': np.array(self.scores, dtype=np.float32)
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved {len(self.positions)} training positions to {filepath}")
    
    def clear(self):
        """Clear collected data"""
        self.positions.clear()
        self.scores.clear()


# Integration function
def integrate_neural_evaluation():
    """
    Call this function to replace the hand-crafted evaluation with neural network.
    You would modify your main game file to call this during initialization.
    """
    try:
        import gameotherother  # Your main game module
        
        # Replace the evaluation function
        gameotherother.evaluate = evaluate_neural
        
        print("Neural network evaluation activated (module-level)!")
        return True
    except ImportError:
        print("Could not import gameotherother module")
        return False


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
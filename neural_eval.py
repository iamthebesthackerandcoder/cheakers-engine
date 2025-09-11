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
        
        # Basic piece representation (squares 1-32)
        for i in range(1, 33):  # checkers squares 1-32
            piece = board[i]
            if piece == 1:      # Black man
                features[i-1] = 1.0
            elif piece == 2:    # Black king
                features[i-1] = 2.0
            elif piece == -1:   # Red man
                features[i-1] = -1.0
            elif piece == -2:   # Red king
                features[i-1] = -2.0
            # Empty squares remain 0
        
        # Additional strategic features (indices 32-39)
        black_men = sum(1 for x in board[1:33] if x == 1)
        black_kings = sum(1 for x in board[1:33] if x == 2)
        red_men = sum(1 for x in board[1:33] if x == -1)
        red_kings = sum(1 for x in board[1:33] if x == -2)
        
        total_pieces = black_men + black_kings + red_men + red_kings
        
        features[32] = black_men / 12.0        # Normalized black men count
        features[33] = black_kings / 12.0      # Normalized black kings count
        features[34] = red_men / 12.0          # Normalized red men count
        features[35] = red_kings / 12.0        # Normalized red kings count
        features[36] = total_pieces / 24.0     # Game phase (endgame indicator)
        features[37] = player                  # Current player to move
        
        # Simple mobility features
        features[38] = self._calculate_mobility_feature(board, 1)   # Black mobility
        features[39] = self._calculate_mobility_feature(board, -1)  # Red mobility
        
        return features.astype(np.float32)
    
    def _calculate_mobility_feature(self, board, player):
        """Calculate a simple mobility feature for the neural network"""
        mobility = 0
        for pos in range(1, 33):
            if board[pos] != 0 and (board[pos] * player > 0):
                # Count basic moves (simplified)
                piece = board[pos]
                r, c = rc(pos)
                
                # Check diagonal moves based on piece type
                if abs(piece) == 2:  # King
                    directions = [(-1,-1), (-1,1), (1,-1), (1,1)]
                elif piece == 1:  # Black man
                    directions = [(-1,-1), (-1,1)]
                else:  # Red man
                    directions = [(1,-1), (1,1)]
                
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 8 and 0 <= nc < 8 and (nr + nc) % 2 == 1:
                        if (nr, nc) in idx_map:
                            target_idx = idx_map[(nr, nc)]
                            if board[target_idx] == 0:
                                mobility += 1
        
        return mobility / 20.0  # Normalize
    
    def _relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def _tanh(self, x):
        """Tanh activation function"""
        return np.tanh(x)
    
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
        a1 = self._relu(z1)
        
        # Layer 2
        z2 = np.dot(a1, self.weights['W2']) + self.weights['b2']
        a2 = self._relu(z2)
        
        # Layer 3
        z3 = np.dot(a2, self.weights['W3']) + self.weights['b3']
        a3 = self._relu(z3)
        
        # Output layer
        z4 = np.dot(a3, self.weights['W4']) + self.weights['b4']
        output = self._tanh(z4) * 1000  # Scale to reasonable evaluation range
        
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
    
    def train_supervised(self, positions, targets, epochs=1, batch_size=1024, lr=1e-3, l2=1e-6, shuffle=True, verbose=True):
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

                # SGD update
                self.weights['W4'] -= lr * dW4
                self.weights['b4'] -= lr * db4
                self.weights['W3'] -= lr * dW3
                self.weights['b3'] -= lr * db3
                self.weights['W2'] -= lr * dW2
                self.weights['b2'] -= lr * db2
                self.weights['W1'] -= lr * dW1
                self.weights['b1'] -= lr * db1

            if verbose:
                avg_loss = total_loss / max(1, total_count)
                print(f"[Neural Train] Epoch {epoch+1}/{epochs} - Avg MSE: {avg_loss:.4f} over {total_count} samples")

        return {"loss": total_loss / max(1, total_count), "count": total_count}
    
    def evaluate_position(self, board, player):
        """
        Main evaluation function that replaces the hand-crafted eval.
        
        Args:
            board: Board state
            player: Player to move
            
        Returns:
            Evaluation score from player's perspective (float)
        """
        features = self.board_to_features(board, player)
        raw_score = self.predict(features)
        # Return score from current player's perspective
        return player * raw_score
    
    def save_model(self, filepath):
        """Save the neural network weights"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.weights, f)
        print(f"Neural model saved to: {filepath}")
    
    def load_model(self, filepath):
        """Load neural network weights"""
        with open(filepath, 'rb') as f:
            self.weights = pickle.load(f)
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
    
    def __init__(self):
        self.positions = []
        self.scores = []
    
    def add_position(self, board, player, game_result):
        """
        Add a position with its eventual game outcome.
        
        Args:
            board: Board state
            player: Player to move
            game_result: Final game result (1.0 for black win, -1.0 for red win, 0.0 for draw)
        """
        evaluator = get_neural_evaluator()
        features = evaluator.board_to_features(board, player)
        
        # Score from current player's perspective in [-1, 0, 1]
        score = player * game_result
        
        self.positions.append(features)
        self.scores.append(score)
    
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
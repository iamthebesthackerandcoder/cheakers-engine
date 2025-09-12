#!/usr/bin/env python3
"""
Enhanced Self-Play Training System with Curriculum Learning
Implements progressive difficulty training and improved data quality
"""

import numpy as np
import pickle
import time
import random
from copy import deepcopy
from collections import deque
import os
import multiprocessing as mp
from multiprocessing import Pool, Manager

# Import the neural evaluator and training components
from neural_eval import NeuralEvaluator, TrainingDataCollector, get_neural_evaluator
from selfplay_trainer import SelfPlayTrainer, ParallelGameRunner

class CurriculumPhase:
    """Defines a curriculum learning phase with specific difficulty parameters"""

    def __init__(self, name, game_count, search_depth_range, noise_level, mutation_rate,
                 min_pieces=None, max_pieces=None, position_importance_threshold=0.0):
        self.name = name
        self.game_count = game_count
        self.search_depth_range = search_depth_range  # (min_depth, max_depth)
        self.noise_level = noise_level
        self.mutation_rate = mutation_rate
        self.min_pieces = min_pieces  # Minimum pieces for complexity filtering
        self.max_pieces = max_pieces  # Maximum pieces for complexity filtering
        self.position_importance_threshold = position_importance_threshold

class PositionScorer:
    """Scores positions for importance in training data"""

    def __init__(self):
        self.center_squares = {5, 6, 9, 10, 14, 15, 18, 19}  # Central squares
        self.king_rows = {1, 2, 3, 4, 29, 30, 31, 32}  # Promotion rows

    def score_position(self, board, player, game_phase=None):
        """
        Score position importance for training
        Returns score between 0-1, higher = more important
        """
        score = 0.0

        # Count pieces
        black_pieces = sum(1 for x in board[1:] if x > 0)
        red_pieces = sum(1 for x in board[1:] if x < 0)
        total_pieces = black_pieces + red_pieces

        # Base score from piece count (prefer mid-game positions)
        if 8 <= total_pieces <= 16:
            score += 0.3
        elif 4 <= total_pieces <= 20:
            score += 0.2

        # Bonus for positions with kings
        black_kings = sum(1 for x in board[1:] if x == 2)
        red_kings = sum(1 for x in board[1:] if x == -2)
        king_bonus = (black_kings + red_kings) * 0.1
        score += min(king_bonus, 0.3)

        # Bonus for center control
        center_control = sum(1 for sq in self.center_squares if board[sq] != 0)
        score += (center_control / 8) * 0.2

        # Bonus for positions near promotion
        promotion_squares = sum(1 for sq in self.king_rows if board[sq] != 0)
        score += (promotion_squares / 8) * 0.15

        # Bonus for complex positions (multiple captures available)
        from gameotherother import legal_moves
        moves = legal_moves(board, player)
        captures = [m for m in moves if len(m) > 2 or (len(m) == 2 and
                    abs(board[m[0]]) == 1 and abs(board[m[1]]) == 0)]  # Simple capture detection

        if len(captures) > 0:
            score += min(len(captures) * 0.1, 0.2)

        # Phase-specific adjustments
        if game_phase == 'opening':
            score += 0.1 if total_pieces >= 20 else 0
        elif game_phase == 'middlegame':
            score += 0.1 if 8 <= total_pieces <= 16 else 0
        elif game_phase == 'endgame':
            score += 0.1 if total_pieces <= 8 else 0

        return min(score, 1.0)

class GameQualityFilter:
    """Filters games based on quality criteria"""

    def __init__(self):
        self.min_game_length = 10  # Minimum moves for a valid game
        self.max_game_length = 150  # Maximum moves to avoid infinite games
        self.min_positions_per_game = 20  # Minimum training positions per game

    def is_game_quality(self, game_result, game_length, positions_collected):
        """Check if a completed game meets quality standards"""
        if game_length < self.min_game_length:
            return False, "Game too short"

        if game_length > self.max_game_length:
            return False, "Game too long (possible infinite loop)"

        if positions_collected < self.min_positions_per_game:
            return False, "Insufficient training positions"

        # Prefer games with decisive results over draws
        if game_result == 0 and game_length < 50:  # Short draws are low quality
            return False, "Short draw game"

        return True, "Good quality"

class CurriculumTrainer(SelfPlayTrainer):
    """Enhanced trainer with curriculum learning capabilities"""

    def __init__(self, base_evaluator=None, resume_from_checkpoint=None):
        super().__init__(base_evaluator, resume_from_checkpoint)

        # Curriculum learning components
        self.position_scorer = PositionScorer()
        self.quality_filter = GameQualityFilter()
        self.curriculum_phases = self._define_curriculum_phases()
        self.current_phase_idx = 0
        self.phase_games_played = 0

        # Enhanced data collection
        self.high_quality_positions = []
        self.position_scores = []
        self.game_qualities = []

        # Training curriculum state
        self.curriculum_enabled = True
        self.adaptive_curriculum = True  # Adjust based on performance

    def _define_curriculum_phases(self):
        """Define the curriculum learning phases"""
        return [
            # Phase 1: Opening games - focus on basic tactics
            CurriculumPhase(
                name="Opening Fundamentals",
                game_count=50,
                search_depth_range=(3, 5),
                noise_level=0.1,
                mutation_rate=0.05,
                min_pieces=18,
                max_pieces=24,
                position_importance_threshold=0.2
            ),

            # Phase 2: Mid-game development - build tactical awareness
            CurriculumPhase(
                name="Mid-game Tactics",
                game_count=100,
                search_depth_range=(4, 6),
                noise_level=0.2,
                mutation_rate=0.1,
                min_pieces=12,
                max_pieces=20,
                position_importance_threshold=0.3
            ),

            # Phase 3: Complex positions - advanced strategy
            CurriculumPhase(
                name="Advanced Strategy",
                game_count=150,
                search_depth_range=(5, 7),
                noise_level=0.3,
                mutation_rate=0.15,
                min_pieces=6,
                max_pieces=16,
                position_importance_threshold=0.4
            ),

            # Phase 4: Endgame mastery - precise play
            CurriculumPhase(
                name="Endgame Mastery",
                game_count=100,
                search_depth_range=(6, 8),
                noise_level=0.25,
                mutation_rate=0.12,
                min_pieces=2,
                max_pieces=10,
                position_importance_threshold=0.5
            )
        ]

    def get_current_phase(self):
        """Get the current curriculum phase"""
        if not self.curriculum_enabled or self.current_phase_idx >= len(self.curriculum_phases):
            return None
        return self.curriculum_phases[self.current_phase_idx]

    def should_advance_phase(self):
        """Check if we should advance to the next curriculum phase"""
        phase = self.get_current_phase()
        if phase is None:
            return False

        return self.phase_games_played >= phase.game_count

    def advance_phase(self):
        """Advance to the next curriculum phase"""
        if self.current_phase_idx < len(self.curriculum_phases) - 1:
            self.current_phase_idx += 1
            self.phase_games_played = 0
            phase = self.get_current_phase()
            print(f"\n🎓 Advanced to curriculum phase: {phase.name}")
            print(f"   Depth range: {phase.search_depth_range}")
            print(f"   Games in phase: {phase.game_count}")
            return True
        return False

    def get_phase_specific_params(self):
        """Get search and opponent parameters for current phase"""
        phase = self.get_current_phase()

        if phase is None:
            # Post-curriculum: use advanced parameters
            return {
                'depth_range': (6, 8),
                'noise_level': 0.3,
                'mutation_rate': 0.15,
                'importance_threshold': 0.4
            }

        return {
            'depth_range': phase.search_depth_range,
            'noise_level': phase.noise_level,
            'mutation_rate': phase.mutation_rate,
            'importance_threshold': phase.position_importance_threshold
        }

    def filter_position_by_phase(self, board, player):
        """Filter positions based on current curriculum phase requirements"""
        phase = self.get_current_phase()
        if phase is None:
            return True  # No filtering in post-curriculum

        # Count pieces
        total_pieces = sum(1 for x in board[1:] if x != 0)

        # Check piece count constraints
        if phase.min_pieces is not None and total_pieces < phase.min_pieces:
            return False
        if phase.max_pieces is not None and total_pieces > phase.max_pieces:
            return False

        return True

    def score_and_collect_position(self, board, player, game_result, game_phase=None):
        """Enhanced position collection with importance scoring"""
        # Skip if doesn't match current phase requirements
        if not self.filter_position_by_phase(board, player):
            return

        # Score position importance
        importance_score = self.position_scorer.score_position(board, player, game_phase)

        # Get current phase threshold
        params = self.get_phase_specific_params()
        threshold = params['importance_threshold']

        # Only collect high-importance positions
        if importance_score >= threshold:
            features = self.base_evaluator.board_to_features(board, player)

            # Convert game result to training target
            if game_result == 0:
                target_score = 0.0  # Draw
            else:
                target_score = 1.0 if player == game_result else -1.0

            self.training_data.add_position(board, player, target_score)

            # Store additional metadata
            self.high_quality_positions.append(features)
            self.position_scores.append(importance_score)

    def create_curriculum_opponent(self):
        """Create opponent with curriculum-appropriate difficulty"""
        params = self.get_phase_specific_params()
        return self.create_opponent_evaluator(
            noise_level=params['noise_level'],
            mutation_rate=params['mutation_rate']
        )

    def _play_single_game_curriculum(self, game_num, shared_tt):
        """Enhanced game playing with curriculum learning"""
        from gameotherother import initial_board, legal_moves, apply_move, is_terminal, SearchEngine

        params = self.get_phase_specific_params()
        depth_range = params['depth_range']

        evaluator1 = self.base_evaluator
        evaluator2 = self.create_curriculum_opponent()

        engine1 = SearchEngine(seed=random.randint(0, 10000), shared_tt=shared_tt)
        engine2 = SearchEngine(seed=random.randint(0, 10000), shared_tt=shared_tt)
        engine1.neural_evaluator = evaluator1
        engine2.neural_evaluator = evaluator2

        board = initial_board()
        player = 1
        move_count = 0
        random_moves_count = 0
        game_positions = []
        game_start = time.time()

        # Determine game phase for position scoring
        game_phase = 'opening'

        while move_count < 200:
            if is_terminal(board, player):
                break

            # Update game phase
            total_pieces = sum(1 for x in board[1:] if x != 0)
            if total_pieces <= 8:
                game_phase = 'endgame'
            elif total_pieces <= 16:
                game_phase = 'middlegame'

            # Store position before move (with curriculum filtering)
            self.score_and_collect_position(board, player, 0, game_phase)  # game_result=0 as placeholder
            game_positions.append((board[:], player))

            # Choose engine based on player
            current_engine = engine1 if player == 1 else engine2

            # Curriculum depth selection
            depth = random.choice(depth_range)

            try:
                if random.random() < self.current_epsilon:
                    moves = legal_moves(board, player)
                    best_move = random.choice(moves) if moves else None
                    random_moves_count += 1
                else:
                    _, best_move = current_engine.search(board, player, depth)
            except Exception:
                moves = legal_moves(board, player)
                best_move = random.choice(moves) if moves else None
                random_moves_count += 1

            if best_move is None:
                break

            board = apply_move(board, best_move)
            player = -player
            move_count += 1

        # Determine final game result
        if is_terminal(board, player):
            game_result = -player
        else:
            game_result = 0

        # Update all position scores with final result
        for board_state, pos_player in game_positions:
            self.score_and_collect_position(board_state, pos_player, game_result, game_phase)

        # Quality check
        is_quality, quality_reason = self.quality_filter.is_game_quality(
            game_result, move_count, len(game_positions)
        )
        self.game_qualities.append((is_quality, quality_reason))

        game_time = time.time() - game_start
        return game_result, len(game_positions), move_count, random_moves_count, game_time

    def run_curriculum_training(self, total_games=400, save_interval=25, num_workers=4,
                               model_save_path="curriculum_model.pkl",
                               data_save_path="curriculum_data.pkl",
                               checkpoint_path="curriculum_checkpoint.pkl"):
        """
        Run curriculum learning training session
        """
        print("🎓 Starting Curriculum Learning Training")
        print("=" * 50)
        print(f"Total games: {total_games}")
        print(f"Curriculum phases: {len(self.curriculum_phases)}")
        print(f"Workers: {num_workers}")
        print()

        games_completed = 0
        start_time = time.time()

        while games_completed < total_games:
            phase = self.get_current_phase()
            if phase is None:
                print("📚 Curriculum completed - entering advanced training")
                break

            phase_games_to_play = min(phase.game_count - self.phase_games_played,
                                    total_games - games_completed)

            print(f"📖 Phase {self.current_phase_idx + 1}: {phase.name}")
            print(f"   Playing {phase_games_to_play} games...")
            print(f"   Depth range: {phase.search_depth_range}")
            print(f"   Progress: {self.phase_games_played}/{phase.game_count}")
            print()

            # Play games for this phase
            for game_idx in range(phase_games_to_play):
                # Update epsilon for exploration
                progress = min(1.0, (games_completed + game_idx) / max(1, total_games))
                self.current_epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * progress

                # Play single game with curriculum parameters
                result, positions, moves, random_moves, game_time = self._play_single_game_curriculum(
                    games_completed + game_idx, None  # No shared TT for sequential
                )

                # Update statistics
                self.training_stats['random_moves_total'] += random_moves
                self.training_stats['random_moves_per_game'].append(random_moves)

                if result == 1:
                    self.training_stats['black_wins'] += 1
                elif result == -1:
                    self.training_stats['red_wins'] += 1
                else:
                    self.training_stats['draws'] += 1

                self.games_played += 1
                self.total_positions += positions
                self.training_stats['positions_collected'] = self.total_positions
                self.phase_games_played += 1

                total_games_all = self.training_stats['black_wins'] + self.training_stats['red_wins'] + self.training_stats['draws']
                if total_games_all > 0:
                    self.training_stats['avg_game_length'] = ((self.training_stats['avg_game_length'] * (total_games_all - 1)) + moves) / total_games_all

                # Progress reporting
                if (game_idx + 1) % 10 == 0 or game_idx == 0:
                    quality_games = sum(1 for q, _ in self.game_qualities[-10:] if q)
                    result_str = "Black wins" if result == 1 else "Red wins" if result == -1 else "Draw"
                    print(f"Game {games_completed + game_idx + 1:3d}: {result_str:10s} | "
                          f"Moves: {moves:3d} | Positions: {positions:3d} | "
                          f"Quality: {quality_games}/10 | Time: {game_time:.2f}s")

                # Periodic training and saving
                if (games_completed + game_idx + 1) % save_interval == 0:
                    if len(self.training_data.positions) > 0:
                        stats = self.train_on_collected_data(epochs=2, batch_size=1024, lr=5e-4, l2=1e-6)
                        if stats and stats.get("loss") is not None:
                            print(f"   📈 Training: MSE {stats['loss']:.4f} on {stats['count']} positions")

                    self.save_training_progress(model_save_path, data_save_path, checkpoint_path)
                    self.print_training_stats()
                    print("-" * 50)

                    # Clear data for next batch
                    self.training_data.clear()
                    self.high_quality_positions.clear()
                    self.position_scores.clear()

            games_completed += phase_games_to_play

            # Check if we should advance phase
            if self.should_advance_phase():
                self.advance_phase()

        # Final training and save
        if len(self.training_data.positions) > 0:
            stats = self.train_on_collected_data(epochs=3, batch_size=1024, lr=5e-4, l2=1e-6)
            if stats and stats.get("loss") is not None:
                print(f"🎯 Final training: MSE {stats['loss']:.4f} on {stats['count']} positions")

        self.save_training_progress(model_save_path, data_save_path, checkpoint_path)

        elapsed_time = time.time() - start_time
        print("\n🎓 Curriculum Learning Complete!")
        print(f"Total time: {elapsed_time:.2f} seconds")
        print(f"Games per second: {games_completed / max(1e-9, elapsed_time):.2f}")
        print(f"Phases completed: {self.current_phase_idx + 1}/{len(self.curriculum_phases)}")
        print(f"High-quality positions collected: {len(self.high_quality_positions)}")
        self.print_training_stats()

    def print_curriculum_stats(self):
        """Print curriculum-specific statistics"""
        print("\n📊 Curriculum Learning Statistics:")
        current_phase = self.get_current_phase()
        if current_phase:
            print(f"Current phase: {current_phase.name}")
            print(f"Phase progress: {self.phase_games_played}/{current_phase.game_count}")
        else:
            print("Current phase: Completed")

        if self.position_scores:
            print(f"Average position importance: {np.mean(self.position_scores):.3f}")
            print(f"Position importance range: {np.min(self.position_scores):.3f} - {np.max(self.position_scores):.3f}")

        quality_games = sum(1 for q, _ in self.game_qualities if q)
        print(f"Quality games: {quality_games}/{len(self.game_qualities)} ({100*quality_games/max(1,len(self.game_qualities)):.1f}%)")

# Example usage
if __name__ == "__main__":
    print("🎓 Curriculum Learning Trainer")
    print("Starting enhanced training with progressive difficulty...")

    trainer = CurriculumTrainer()
    trainer.run_curriculum_training(
        total_games=100,  # Start with smaller number for testing
        save_interval=25,
        num_workers=1,
        model_save_path="curriculum_model.pkl",
        data_save_path="curriculum_data.pkl",
        checkpoint_path="curriculum_checkpoint.pkl"
    )

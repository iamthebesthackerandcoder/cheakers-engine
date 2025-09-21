from __future__ import annotations

import argparse
from config import setup_logging, ensure_dirs, DEFAULT_MODEL_PATH, DEFAULT_DATA_PATH, DEFAULT_CHECKPOINT_PATH
from checkers.training.selfplay import SelfPlayTrainer


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run self-play training session")
    ap.add_argument("--games", type=int, default=100, help="Number of games to play")
    ap.add_argument("--save-interval", type=int, default=10, help="Save progress every N games")
    ap.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    ap.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Model save path")
    ap.add_argument("--data", default=DEFAULT_DATA_PATH, help="Training data save path (npz)")
    ap.add_argument("--ckpt", default=DEFAULT_CHECKPOINT_PATH, help="Checkpoint save path")
    return ap.parse_args()


def main() -> None:
    setup_logging()
    ensure_dirs()
    args = parse_args()

    trainer = SelfPlayTrainer()
    trainer.run_training_session(
        num_games=args.games,
        save_interval=args.save_interval,
        num_workers=args.workers,
        model_save_path=args.model,
        data_save_path=args.data,
        checkpoint_path=args.ckpt,
    )


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    from selfplay_trainer import SelfPlayTrainer
    import time
    import os

    print("=== OPTIMIZED PARALLEL BENCHMARK ===")
    print("Testing performance improvements...")

    start = time.time()
    trainer = SelfPlayTrainer()

    # Use optimized settings
    trainer.run_training_session(
        num_games=30,  # Increased for better measurement
        save_interval=15,  # More frequent saves
        num_workers=4,
        model_save_path='benchmark_par_model.pkl',
        data_save_path='benchmark_par_data.pkl',
        checkpoint_path='benchmark_par_checkpoint.pkl'
    )
    end = time.time()

    elapsed = end - start
    print('\n=== PARALLEL BENCHMARK RESULTS ===')
    print('Total time: {:.2f}s'.format(elapsed))
    print('Games/hour estimate: {:.1f}'.format(30 / elapsed * 3600))

    # Load checkpoint for detailed stats
    if os.path.exists('benchmark_par_checkpoint.pkl'):
        import pickle
        with open('benchmark_par_checkpoint.pkl', 'rb') as f:
            chk = pickle.load(f)
        total_positions = chk.get('total_positions', 0)
        print('Total positions collected: {}'.format(total_positions))
        print('Positions/hour estimate: {:.1f}'.format(total_positions / elapsed * 3600))

        # Calculate efficiency metrics
        if 'training_stats' in chk:
            stats = chk['training_stats']
            total_games = stats.get('black_wins', 0) + stats.get('red_wins', 0) + stats.get('draws', 0)
            if total_games > 0:
                avg_moves = stats.get('avg_game_length', 0)
                print('Average moves per game: {:.1f}'.format(avg_moves))
                print('Positions per game: {:.1f}'.format(total_positions / total_games))

    print('\n=== PERFORMANCE OPTIMIZATIONS APPLIED ===')
    print('✓ Improved transposition table management')
    print('✓ Vectorized feature extraction')
    print('✓ Optimized batch processing')
    print('✓ Enhanced multiprocessing')
    print('✓ Better memory management')
    print('✓ Increased cache sizes')

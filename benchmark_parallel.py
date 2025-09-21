if __name__ == "__main__":
    from selfplay_trainer import SelfPlayTrainer
    import time
    start = time.time()
    trainer = SelfPlayTrainer()
    trainer.run_training_session(num_games=20, save_interval=20, num_workers=4, model_save_path='benchmark_par_model.pkl', data_save_path='benchmark_par_data.pkl', checkpoint_path='benchmark_par_checkpoint.pkl')
    end = time.time()
    print('\n=== PARALLEL BENCHMARK TOTAL TIME: {:.2f}s ==='.format(end - start))
    print('Games/hour estimate: {:.1f}'.format(20 / (end - start) * 3600))
    import pickle
    with open('benchmark_par_checkpoint.pkl', 'rb') as f:
        chk = pickle.load(f)
    print('Total positions collected: {}'.format(chk.get('total_positions', 0)))
    print('Positions/hour estimate: {:.1f}'.format(chk.get('total_positions', 0) / (end - start) * 3600))

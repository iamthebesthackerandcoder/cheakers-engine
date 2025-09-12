from selfplay_trainer import SelfPlayTrainer
trainer = SelfPlayTrainer()
trainer.run_training_session(num_games=10, save_interval=5, model_save_path='profile_model.pkl', data_save_path='profile_data.pkl', checkpoint_path='profile_checkpoint.pkl')
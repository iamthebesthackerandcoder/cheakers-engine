from selfplay_trainer import SelfPlayTrainer
import time

start_time = time.time()
trainer = SelfPlayTrainer()
trainer.run_training_session(num_games=2, save_interval=1, num_workers=2, model_save_path='profile_model.pkl', data_save_path='profile_data.npz', checkpoint_path='profile_checkpoint.pkl')
end_time = time.time()
print(f"Profile run completed in {end_time - start_time:.2f} seconds")
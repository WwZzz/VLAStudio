import vlastudio as vla
dataset = vla.load_dataset("examples/07_custom_sim_env_libero_plus/config/task_object.yaml")
policy = vla.load_policy("examples/07_custom_sim_env_libero_plus/config/smolvla.yaml")
vla.train(policy, dataset, "smolvla", output_dir="checkpoints/example07_smolvla", overrides={"training.max_steps": 5000, "training.per_device_train_batch_size": 16, "training.dataloader_num_workers": 4, "training.save_steps": 5000, "training.save_total_limit": 2})
vla.load_env("examples/07_custom_sim_env_libero_plus/config/env_object.yaml").evaluate(policy, output_dir="results/example07_plus", num_rollout=1, action_manager="examples/07_custom_sim_env_libero_plus/config/action_manager.yaml")

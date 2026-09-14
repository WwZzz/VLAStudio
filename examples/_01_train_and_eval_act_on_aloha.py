import vlastudio as vla
dataset = vla.load_dataset("sim_transfer_cube_scripted")
policy = vla.load_policy("act")
vla.train(policy, dataset, "default", output_dir="checkpoints/act_aloha")
vla.load_env("aloha_transfer").evaluate(policy, output_dir="results/act_aloha")

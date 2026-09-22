import vlastudio as vla

policy = vla.connect_policy("127.0.0.1:5000")
vla.load_env("tabletop_sim.dish_drainer").evaluate(
    policy, output_dir="results/pi05_tabletop", num_rollout=4, batch_size=0,
)

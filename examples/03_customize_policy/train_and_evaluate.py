"""Train and evaluate a policy implemented outside VLAStudio."""

import vlastudio as vla


dataset = vla.load_dataset("sim_transfer_cube_scripted")
policy = vla.load_policy("examples/03_customize_policy/my_policy.yaml")
vla.train(policy, dataset, "default", output_dir="checkpoints/custom_policy")
bench = vla.load_env("aloha_transfer")
bench.evaluate(policy, output_dir="results/custom_policy")

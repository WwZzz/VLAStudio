"""Train SmolVLA on LIBERO-Object and evaluate every object task."""

import vlastudio as vla


dataset = vla.load_dataset("libero.libero_object")
policy = vla.load_policy("examples/04_multitask_policy/smolvla.yaml")
vla.train(policy, dataset, "smolvla", output_dir="checkpoints/smolvla_libero")
bench = vla.load_env("libero.object")
bench.evaluate(policy, output_dir="results/smolvla_libero", num_rollout=4, batch_size=0)

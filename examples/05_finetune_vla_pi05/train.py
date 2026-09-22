"""LoRA-finetune π0.5 on Tabletop-Sim dish-drainer."""

import vlastudio as vla


dataset = vla.load_dataset("tabletop_sim.dish_drainer")
policy = vla.load_policy("examples/05_finetune_vla_pi05/pi05.yaml")
vla.train(policy, dataset, "openpi_lora", output_dir="checkpoints/pi05_tabletop")

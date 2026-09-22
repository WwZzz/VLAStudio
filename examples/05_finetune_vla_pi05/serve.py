import vlastudio as vla

policy = vla.load_policy(
    "examples/05_finetune_vla_pi05/pi05.yaml",
    checkpoint="checkpoints/pi05_tabletop",
)
vla.serve(policy, address="0.0.0.0:5000")

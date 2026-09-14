import vlastudio as vla

policy = vla.load_policy("act", checkpoint="checkpoints/act_aloha")
vla.serve(policy, address="0.0.0.0:5000")

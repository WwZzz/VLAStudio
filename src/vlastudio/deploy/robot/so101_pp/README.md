# SO101++ (`so101_pp`) — parallel gripper

Same 7-motor layout as `so101_plus`, but calibration handles a **parallel gripper** that cannot be opened/closed by hand.

## Calibration (two phases)

```bash
# Leader (example: ttyACM0)
python -m deploy.robot.so101_pp --port /dev/ttyACM0 --id so101_pp_leader --calibrate

# Follower (example: ttyACM1)
python -m deploy.robot.so101_pp --port /dev/ttyACM1 --id so101_pp_follower --calibrate
```

1. **Arm joints** (no gripper): move by hand through ROM, Enter to finish (same as so101_plus).
2. **Gripper**: a Tkinter slider window opens  
   - close fully → **Set min (closed)** (or Enter)  
   - open fully → **Set max (open)** (or Enter)  
   - Home is set to **closed** (normalized 0), not mid.

Calibration files:
`~/.cache/huggingface/lerobot/calibration/robots/so101_pp/<id>.json`

## Teleop

```bash
python collect_data.py \
  -r configs/robot/so101_pp.yaml \
  -t configs/teleop/so101_pp_leader.yaml \
  -o data/so101_pp_teleop \
  --no-record-teleop
```

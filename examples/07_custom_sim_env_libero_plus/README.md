# 07 — Custom benchmark: LIBERO-Plus

Requires Linux, Python 3.10 and an NVIDIA GPU. Run from the repository root:

```bash
bash examples/07_custom_sim_env_libero_plus/setup.sh
source "${VLASTUDIO_CACHE:-$HOME/.cache/vlastudio}/envs/example07/activate.sh"
python examples/07_custom_sim_env_libero_plus/train_and_evaluate.py
```

- Cache: `~/.cache/vlastudio`; override with `VLASTUDIO_CACHE`.
- Setup options and native dependencies: [setup.sh](setup.sh) (`--help`, `-i URL`).
- Training budget and output paths: [train_and_evaluate.py](train_and_evaluate.py). Use fresh evaluation outputs.
- Custom benchmark: implement [libero_plus_env.py](libero_plus_env.py), reference it in [env_object.yaml](config/env_object.yaml).

## Verified results · 2026-10-08

RTX 4090, 5,000 steps, batch 16: training and evaluation completed (exit 0).
Training took 28.8 minutes; the seven default scenes scored **2/7**.
Same checkpoint, ten Object tasks × five trials per condition:

| Condition | Success |
| --- | ---: |
| Standard LIBERO-Object | 19/50 (38%) |
| Background textures | 7/50 (14%) |
| Camera viewpoints | 3/50 (6%) |
| Language instructions | 0/50 (0%) |
| Light conditions | 15/50 (30%) |
| Object layouts | 16/50 (32%) |
| Robot initial states | 3/50 (6%) |
| Sensor noise | 8/50 (16%) |

Language control with canonical instructions: **17/50 (34%)**.
Sampled results, not the full benchmark; cached dependencies were reused.
Sensor noise used pixel/RNG-equivalent glass-blur acceleration.
[Full validation details](https://github.com/WwZzz/VLAStudio/blob/ffdde23cd53bc7e7457c1fb6becc4a5bb980189d/examples/07_custom_sim_env_libero_plus/README.md#verified-run-2026-10-08).

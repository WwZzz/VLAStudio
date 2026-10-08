# 08 — FastWAM on LIBERO

Linux, Python 3.10 and an NVIDIA GPU with EGL support. From the repository root:

```bash
bash examples/08_fastwam_libero/setup.sh
source "${VLASTUDIO_CACHE:-$HOME/.cache/vlastudio}/envs/example08/activate.sh"
python examples/08_fastwam_libero/evaluate.py
```

Uses the official FastWAM Base checkpoint; no retraining. Cache: `VLASTUDIO_CACHE`
(default `~/.cache/vlastudio`). Setup accepts `-i URL`. Evaluation accepts
`--suite spatial|object|goal|10`, `--checkpoint PATH`, and `--output-dir PATH`.
Use fresh outputs. Configuration files are in `config/`.

## Paper comparison

50 initial states per task, 10 tasks per suite (2,000 trials total).
30 settling steps; 32 predicted / 10 executed actions; 10 denoising steps,
scheduler shift 5.0, seed 42. Two 256px cameras rotated 180 degrees, resized
to 224px and concatenated; 400-step limit (700 for Long). The simulator is
reused between trials to preserve the official reset behavior.

| Suite | Paper | VLAStudio |
| --- | ---: | ---: |
| Spatial | 98.2% | 96.0% (480/500) |
| Object | 100.0% | 99.4% (497/500) |
| Goal | 97.0% | 96.6% (483/500) |
| Long | 95.2% | 94.0% (470/500) |
| Average | 97.6% | 96.5% (1930/2000) |

Validation uses eager BF16 inference on RTX 4090 48GB with PyTorch 2.7.1+cu126.
Reset images/states and predicted actions were checked against the official code.

Sources: [paper, Table 2](https://arxiv.org/html/2603.16666v1),
[official code](https://github.com/yuantianyuan01/FastWAM),
[released checkpoint](https://huggingface.co/yuanty/fastwam).

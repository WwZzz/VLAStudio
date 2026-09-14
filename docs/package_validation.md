# Packaging and GPU validation

Validation date: 2026-09-14. Branch: `codex/package-runtime-isolation`.
Base: `db79c4c12c4858e310a6990c5c4c9f4b4d76f381` (main, PR #39).

## Packaging and extension contracts

31 local tests pass, including file-based extensions across multiprocessing spawn,
config/argument compatibility, cache overrides, dependency-lock consistency,
interrupted-install recovery, and custom OpenVLA pretrained checkpoint paths.
Wheel and source distributions build. The wheel keeps legacy implementations in
`vlastudio/_legacy` and does not install generic top-level `policy`/`utils` packages.
The lightweight public import does not import torch or create environments.

Windows wheel installation, two-step CPU training, custom dataset/config loading,
custom device start/close, custom runtime entrypoints, and offline environment reuse
were also verified. GPU tests use the installed wheel from outside the source tree.

## GPU execution

Host: QZ Linux, RTX 4090 (49,140 MiB reported), NVIDIA driver 550.163.01,
glibc 2.35. Each model runs in its managed environment, with no host ML environment
on the worker import path. See `gpu_validation.json` for measured results.

| Check | Torch / CUDA | Peak allocated MiB | Result |
| --- | --- | ---: | --- |
| MLP train/inference | 2.4.0+cu121 / 12.1 | 17.3 | Pass |
| ACT train/inference | 2.4.0+cu121 / 12.1 | 238.1 | Pass |
| Original train.py, two GPU steps | 2.4.0+cu121 / 12.1 | 17.3 | Pass |
| OpenVLA reduced model, checkpoint load/train/inference | 2.4.0+cu121 / 12.1 | 148.6 | Pass |
| OpenPI full architecture, LoRA step/compiled inference | 2.7.1+cu126 / 12.6 | 7069.6 | Pass |

All model checks assert CUDA execution, finite loss/gradients, an optimizer update,
and finite inference output. The MLP CLI check executes the original `train.py`
for two GPU steps with external YAML/dataset files and saves checkpoints to a
user-selected directory containing spaces.

OpenVLA uses a small randomly initialized architecture, a synthetic tokenizer,
and a local checkpoint, including the real factory and `select_action` path.
OpenPI uses the 3.5B-parameter architecture with random weights and rank-2 LoRA,
batch size 1, action horizon 2, token length 8, and one compiled denoising step. These are execution
checks, not pretrained model quality or robotics task-success evaluations.

## Fixes found through integration

- Preserve incomplete environments on retry and mark readiness only after success.
- Increase uv cache-lock waiting for large CUDA downloads; respect download-cache
  overrides and route TorchInductor, Triton and OpenPI asset caches under the cache root.
- Match OpenVLA's required timm/transformers/tokenizers versions, preserve its custom
  pretrained path, and lazy-load the optional RLDS data pipeline.
- Declare pytest and chex, which the pinned OpenPI model/tokenizer import at runtime.
- Ship the validated Linux ACT/MLP lock alongside the other platform locks.

The QZ default download route was slow. Official wheel bytes were fetched through
an installation mirror and verified against PyPI SHA-256 metadata before uv
installed the locked dependencies. Validation caches were relocated when the shared
HDD quota filled; no user datasets or working checkouts were modified.

## Remaining coverage

Pretrained OpenVLA-7B/OpenPI checkpoint quality, real robot/device SDKs, multi-GPU
training, Windows GPU execution, RLDS ingestion, and simulation backends are not
covered by these checks. Unsupported policies still require a runtime manifest or
an existing environment. A managed `Policy.load().predict()` Python proxy is outside
this first version; use task dispatch or existing policy-server clients.

This is a development package, not a PyPI release. Install the built wheel or the
branch to try it. Reproduction instructions are in `package_runtime.md` and the
runnable scripts under `examples/`.

## Python orchestration API (2026-09-14)

The follow-up package adds `load_dataset`, `load_policy`, `train` and `load_env`
configuration handles, with `bench.evaluate(policy)` dispatch. All public arguments
and the training result use the name `policy`.

- 34 packaging/API tests pass, including real subprocess execution with a custom
  test entrypoint, checkpoint handoff, evaluation JSON retrieval, failure propagation,
  and argument forwarding. Environment preparation is mocked in that subprocess
  contract test; its synthetic metrics are not simulator measurements.
- Wheel and sdist build successfully. The wheel contains the public API; the sdist
  includes the Python examples and preserved legacy README.
- The installed wheel's `examples/train_mlp.py --gpu` completed two actual
  training steps on the existing RTX 4090 instance through the managed environment.
  Reported training loss: 0.32710614800453186. Checkpoint and trainer state were saved
  under `api-gpu-checkpoints` in the dedicated validation directory.

This follow-up does not claim real simulator evaluation coverage. Managed evaluation
requires a complete policy-plus-simulator manifest; the Python API does not expose
live Torch objects in the calling process.

## Native package layout

Implementations and built-in configs now live directly under `src/vlastudio`.
The wheel has no `_legacy` directory or forced copy of root-level implementations.
The 40 packaging tests cover native layout, namespace/legacy alias identity,
runtime-profile compatibility and the existing API contracts. GPU smoke checks for
MLP, ACT, the full MLP training entrypoint and OpenVLA passed again after migration.
The root scripts remain thin entrypoints; training implementation is in
`vlastudio.entrypoints.train`, keeping the public `vlastudio.train` function distinct.

## ACT on ALOHA end-to-end example

`examples/_01_train_and_eval_act_on_aloha.py --smoke` completed two ACT training
steps and a five-step ALOHA transfer-cube MuJoCo rollout on the RTX 4090. The
managed evaluation environment reported PyTorch 2.4.0+cu121 with CUDA available,
loaded the saved 336 MB checkpoint on `cuda`, processed four policy inferences,
and wrote evaluation JSON, state/action samples, a camera image and an MP4 video.
The random smoke policy scored 0/1, as expected for an execution check.

The ALOHA runtime pins `dm-control==1.0.34` with `mujoco==3.3.6`; allowing the
resolver to select MuJoCo 3.13.0 failed because its model fields are incompatible
with that dm-control release. The example defaults environment creation to the
Aliyun PyPI mirror and exposes `--package-index` for callers to replace it.

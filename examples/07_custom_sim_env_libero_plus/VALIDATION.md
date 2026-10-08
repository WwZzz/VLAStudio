# Validation record — 2026-09-25

This record distinguishes integration checks, a new training run, and a separate
execution-horizon diagnostic. The reported samples do not reproduce a 97% score.

## Revisions and runtime

- VLAStudio development base: `79bd6d428debcd254a09a05c7d716ebd88c5917c`.
- Working branch: `codex/example07-libero-plus`.
- LIBERO-Plus: `4976dc30028e805ff8094b55501d532c48fec182`.
- Standard LIBERO: `8f1084e3132a39270c3a13ebe37270a43ece2a01`.
- One NVIDIA RTX 4090, Python 3.10, PyTorch 2.7.1 with CUDA 12.6.
- LeRobot 0.3.3, Transformers 4.51.3, MuJoCo 3.3.2, robosuite 1.4.1,
  NumPy 1.26.4.

## Training

The supplied `train.py` completed 5,000 optimizer steps with batch size 16,
BF16, seed 0, and both main and wrist cameras. It used the original Object
HDF5 demonstrations: 10 tasks, 500 episodes, 74,507 transitions. This amounts
to approximately 1.074 epochs; it is a short training run, not evidence of
convergence or equivalence to a published SmolVLA checkpoint.

- Initialization: pretrained `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` vision
  and language backbone; the supplied SmolVLA action-policy training recipe.
- Duration: 1,965.806 seconds (32.8 minutes).
- Last logged training loss: 0.1033; mean training loss: 0.159594.
- Checkpoint: `checkpoints/example07_smolvla`.
- Training log: `results/train5000.log`.

## Integration checks

- Named-environment execution and relative/absolute runtime manifest selection passed.
- Six focused regression checks passed: reset ordering/state iteration, observation
  layout, success vs. timeout, initial-state paths, unique rollout indices, and
  execution of exactly 16 out of 50 predicted actions.
- All seven sample perturbations passed real EGL reset/render/step checks.
- The full Object generator produced 2,518 tasks; all corresponding initial-state
  files were found. The full 2,518-task evaluation was **not** run.
- A real robot-pose check found that the upstream full-state restore overwrites
  the named robot variant with standard joint positions. Reapplying the variant
  after object-state restoration reduced the joint target error from 0.110866
  to 1.31e-7 before settling. This correction is part of this adapter's protocol.

## New checkpoint evaluation

All runs use 256 × 256 main and wrist images, explicit 180-degree image rotation
in both training and evaluation, native 8D end-effector state and 7D delta actions,
16 executed actions per 50-action prediction, seed 0, ten settling actions,
and a maximum of 280 policy-controlled environment steps.

The Plus sample contains seven predetermined alphabet-soup variants, one per
category. The final run completed with **2/7 successes (28.6%)**: the language
and lighting variants succeeded. Artifacts: `results/trained5000_plus_final`.
This is a small teaching subset of one base task, not a full-suite robustness
score, and is not directly comparable to the ten-task standard Object rate.

The first provisional Plus run fed filename-derived strings containing
perturbation parameters to the policy. It is retained in
`results/trained5000_plus` as a diagnostic, **not the final example score**.
The final adapter removes those suffixes and preserves the canonical training
instruction; deliberate language variants retain their BDDL rewrite. All 2,164
non-language Object variants were checked against the HDF5 instruction metadata
and matched. The remaining 354 variants intentionally change language.

Standard Object completed with **18/50 successes (36%)**, using all ten tasks
and initial states 0–4 per task. Artifacts: `results/trained5000_standard`.
The final adapter retains the same canonical instruction for standard tasks.
The short 5,000-step policy therefore has limited success even without Plus
perturbations; this run cannot isolate a full-suite robustness gap.

Each completed evaluation writes `all_envs_summary.json`, per-task JSON,
`protocol.json`, and rollout videos. Results and checkpoints are not Git inputs.

## Separate diagnostic using an existing checkpoint

An existing 30,000-step, single-camera checkpoint was evaluated on the same
standard Object tasks and initial states 0 and 1 per task (20 trials per setting).
Both runs used the corrected state-reset sequence and the same seed.

| Execution setting | Successes | Rate |
| --- | ---: | ---: |
| Execute all 50 predictions before observing again | 3/20 | 15% |
| Execute the first 16 predictions before observing again | 10/20 | 50% |

Artifacts: `results/control50` and `results/control16`. This small controlled
comparison supports checking the execution horizon before attributing a low
score entirely to simulator difficulty. It is separate from the new two-camera
5,000-step training run and must not be pooled with it.


## Matched-checkpoint audit — 2026-09-26

A separate full-HDF5 SmolVLA checkpoint previously scored 96/100 in the legacy
VLAStudio branch. The four original job logs were verified: 26/30, 30/30,
20/20 and 20/20. That policy used 100,000 updates, global batch 64, an explicitly
unfrozen vision tower, one main camera, raw image orientation, 128x128 inputs,
and 16 predicted/executed actions. It is not the 5,000-step policy above.

The same saved 100k weights and their own normalization were then used in both
implementations on a fresh 4090. No model was retrained for this audit.

| Check | Result |
| --- | --- |
| Fixed observations from all ten tasks, identical dependency runtime | All 70 recorded arrays exactly equal; maximum absolute difference 0 |
| Same observations, example07 dependency runtime | Preprocessing exactly equal; final action maximum absolute difference 0.0034 |
| Checkpoint loading in all three combinations | No missing, unexpected or mismatched weights |
| Legacy closed loop, ten tasks, initial states 0 and 1 | 18/20 (90%) |
| Packaged closed loop with example07 adapter, same states | 18/20 (90%); all 20 success/failure outcomes match |
| Python API `load_policy` / `load_env` / `evaluate`, first two tasks | 3/4; all four outcomes match the corresponding direct-entrypoint episodes |

The closed-loop comparison used raw main-camera images rendered at 256x256
then resized to 128x128 with OpenCV INTER_LINEAR, the standard task language,
checkpoint normalization, a basic manager executing the 16-action prediction,
seed 90670, ten settling actions and a 280-step horizon. The adapter's default
two-camera/rotated configuration was overridden only for this checkpoint's
matching protocol. Do not feed these raw images to the separately trained
5k checkpoint or change its normalization.

These paired 20-episode results do not reproduce the original 100-episode
estimate or prove all configurations equivalent. They show that the inspected
packaged pipeline can preserve this policy's high success under matched
conditions. The earlier 2/7 Plus sample cannot establish that packaging or
Plus difficulty caused the gap; the underlying trained policies differ.

Two independent bugs were reproduced and fixed during the audit:

- Horizon-dependent action statistics `(T,A)` were applied after swapping
  `(B,T,A)` to `(T,B,A)`. The normalizer now preserves batch-first layout and
  rejects shape expansion. Six numerical cases pass. Both policies in this
  comparison use `(7,)` action statistics, so this defect did not explain their
  performance difference.
- An already partitioned DistributedSampler was partitioned again by Accelerate.
  A real two-rank CPU test covered only 16/32 samples before the fix and 32/32
  afterward. Train and eval loaders now retain their existing rank partition;
  unpartitioned custom loaders still pass through Accelerate and cover 32/32.
  This issue does not explain the single-GPU 5k run.

Audit outputs live under `results/parity0926`; they are not Git inputs.

## Release checks — 2026-10-08

The example's training script, shell wrapper and documented command now default
consistently to batch size 16, matching the recorded 5,000-step GPU run.

- Re-ran all six focused adapter/evaluation regressions on CPU, isolating
  simulator and GPU imports while executing the actual source implementations.
- Re-ran six numerical action-denormalization cases across batch sizes 1, 2
  and 16 with vector and per-horizon statistics; all passed.
- Parsed every example Python file and YAML configuration successfully.
- Verified `--help` for setup, configuration generation, training and evaluation.
- Checked training API dispatch with the training call mocked: 5,000 steps and
  batch size 16 reach the API as intended.
- Checked that the example documentation is English and generated results,
  checkpoints and regression scripts are excluded from the commit.

These are release checks, not a new GPU training or rollout run. The GPU results
above remain the September measurements; the short training recipe has not been
shown to achieve the matched 100k checkpoint's success rate.

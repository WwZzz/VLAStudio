# Replay policy (`policy.replay`)

Play back dataset actions from a **pseudo checkpoint** directory. `select_action` ignores images/state. `MetaPolicy` still wraps the policy, but `normalize.json` uses **identity** state/action normalizers: `replay_chunks.npz` stores **raw** actions as in the dataset (no training stats). Trajectories are built with `load_raw_dataset_for_task`; task YAML `action_normalize` / `state_normalize` are **not** applied when extracting actions.

## Task name

Use dot form like **`local.t0325`** (same as `train.py -t` / `ConfigLoader`), i.e. `configs/task/local/t0325.yaml`. Avoid `local/t0325` path syntax unless you pass a real `.yaml` path.

## Build a pseudo checkpoint

```bash
python scripts/generate_replay_checkpoint.py \
  --output_dir ckpt/replay_demo \
  --task local.t0325 \
  --episode_id 0 \
  --chunk_size 100 \
  --replay_key action_first
```

- **`--episode_id`**: multiple ids (`0 1 2`) or **`all`** (sorted global episode ids; do not pass `all 0`).
- **`--loop`** is **on** by default; use **`--no-loop`** for a single pass.
- **Transitions**: if L∞ between last frame of episode A and first of B exceeds **`--transition_linf_thresh`**, linear bridge frames are inserted (count clamped by **`--transition_min_insert`** / **`--transition_max_insert`**). Threshold `0` skips episode-to-episode bridges; with **`--loop`**, a wrap bridge from last→first is still added (effective step size derived from gap / `transition_max_insert` when thresh is 0).
- **`chunk_size`** is written to `config.json`; if you edit `config.json` only, keep it equal to the time dimension in `replay_chunks.npz`.

List episode ids:

```bash
python -m policy.replay --list-episodes local.t0325
```

## Run

Same as any other checkpoint:

```bash
python start_policy_server.py -m ckpt/replay_demo -p 5000
python eval_real.py -r <robot> -m ckpt/replay_demo -am basic
```

Use an action manager that aligns with your chunk length (e.g. `basic`, `sync_chunk`). If **`loop`** is off and **`on_exhausted`** is **`repeat_last`**, after the last chunk the policy repeats **only that last chunk** forever—not the full demo. Default loading treats missing **`loop`** in `replay_spec.json` as **true**.

`normalize.json` has a single synthetic entry **`dataset_id`: `replay`**. No `*.pkl` stats files. For `start_policy_server`, leave **`--dataset_id`** empty or set it to **`replay`**.

## Diagnostics: replay vs trained ckpt (`eval_real`)

To check whether odd ACT behavior comes from the **observation pipeline** (sync, `obs2meta`) vs the **policy**, record the same inputs under `eval_real` for replay and for the trained checkpoint, then diff by `trigger_t`:

```bash
python eval_real.py -r <robot> -m ckpt/replay_demo -am basic --infer_record_dir /tmp/rec_replay
python eval_real.py -r <robot> -m ckpt/act_right_pick -am basic --infer_record_dir /tmp/rec_act
PYTHONPATH=. python scripts/compare_infer_records.py /tmp/rec_replay /tmp/rec_act \
  --label_a replay --label_b act
```

Each inference writes `inference/data/*.npz` with **`state`** / **`image`** (snapshot **before** `MetaPolicy` state normalization), full **`actions`** chunk, and **`action_first`**. `inference/index.jsonl` includes `trigger_t`, `model_name_or_path`, etc. Use the same robot config and action manager so `trigger_t` lines up.

## Files in the pseudo directory

| File | Role |
|------|------|
| `policy_metadata.json` | `policy_module`: `policy.replay` |
| `normalize.json` | Identity normalizers for `replay` |
| `config.json` | `chunk_size`, `action_dim`, … |
| `replay_spec.json` | `loop`, `on_exhausted`, `ctrl_space` / `ctrl_type`, episodes, transition settings |
| `replay_chunks.npz` | `chunks` `(N, chunk_size, dim)` in **raw** dataset action space |

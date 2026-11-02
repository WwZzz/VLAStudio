# Persistent execution

`vlastudio.open_execution()` runs observation acquisition, policy inference and
action publication in native subprocesses. `vlastudio.connect_target()` keeps a
simulation or robot connected across sessions. Neither API depends on an agent
framework. Install this source checkout in the application's Python environment
with `python -m pip install -e .`.

```python
from vlastudio import open_execution

def main():
    with open_execution("execution.yaml", output_dir="runs/task") as session:
        stage = session.update_context(
            {"instruction": "Move to the target"},
            limits={"max_control_steps": 200, "timeout_s": 30},
        )
        result = session.wait(stage["version"], timeout_s=40)
        print(result["state"], result["video"])

if __name__ == "__main__":
    main()
```

Configuration can be a dictionary or YAML path. A minimal session selects:

```yaml
backend:
  type: sim                       # or robot
  config: configs/env/target.yaml  # load_env config; robot uses load_robot config
policy:
  checkpoint: path/to/checkpoint
  device: cpu
action_manager: basic
control_hz: 20
record_hz: 20
```

The selected model and target require their own dependencies and configuration.
`policy.loader` optionally names an importable `module:function` accepting native
inference arguments. `policy.config_path` passes configuration to that loader.
This is a model construction hook, not a second action scheduler.

## Context and lifecycle

Observations and context use separate shared-memory channels. Context changes
only when submitted; the inference worker reads its latest snapshot alongside
observations. `update_context(payload, limits)` replaces the entire context,
assigns a new version and invalidates buffered actions. The policy hook
`set_execution_context(snapshot)` receives arbitrary application fields plus
runtime metadata. Language is optional. Without that hook, only instruction
text is supported through the existing observation `raw_lang` field; custom
payload fields cause an explicit error.

The runtime owns `version`, `epoch`, `active`, `min_obs_timestamp` and
`max_observation_age_s`. Applications must not use these names for payload data.
`instruction` and `language` are normalized text fields; optional `label` is used
in recordings. `set_intent()` remains an alias for `update_context()`.

`observe()` reads the latest observation snapshot. `status()` reports state and
versions. `pause()` revokes pending actions; `resume()` retains the context
version but changes its epoch and resets the policy's temporal state. Results
from old versions or epochs cannot be published. `cancel()` stops the session's
current execution. These controls stop new publications; physical stopping
behavior remains the robot driver's responsibility.

An external evaluator can call
`complete(version, epoch, evidence=..., reason="external_completion")`.
Stale completion signals are rejected. Reaching a limit yields `exhausted`,
not task success. The runtime does not infer application-level success.
`close()` releases workers and shared memory and finalizes trace, result and
video artifacts in `output_dir`. Use a fresh output directory for each session.

## Reuse a connection

```python
from vlastudio import connect_target, open_execution

def main(target_config, runtime_config):
    with connect_target(target_config) as target:
        for index, instruction in enumerate(["First command", "Second command"]):
            with target.execution_backend() as backend:
                config = {**runtime_config, "backend": backend}
                with open_execution(config, output_dir=f"runs/task-{index}") as session:
                    stage = session.update_context(
                        {"instruction": instruction}, limits={"timeout_s": 30})
                    session.wait(stage["version"], timeout_s=40)
```

`target_config` is the backend configuration shown above. Only one session may
borrow a target at a time. Closing a borrowing session keeps the target connected
and preserves the scene; a terminated simulation requires a new connection.
Each session still loads its own inference worker.

Native ActionManager's existing blocking API is unchanged. Persistent execution
uses its nonblocking selection capability; a manager overriding selection must
explicitly support that capability rather than silently use another scheduler.

## Verification

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:tests python -m unittest \
  test_execution_api test_execution_session test_execution_recording -v
```

These CPU tests exercise real processes, shared memory, recording and both target
paths with deterministic model/device substitutes. They do not establish physical
robot task success. Set `VLA_NATIVE_ACCEPTANCE_OUTPUT` to a fresh directory to
retain test recordings.

# Remote inference with separate environments

This example runs policy inference and simulation evaluation in separate processes.
The policy server uses policy dependencies; the evaluation client uses simulation
and communication dependencies. The two environments may use incompatible versions
of PyTorch, Transformers, MuJoCo, or other libraries.

From the repository root:

```bash
examples/02_remote_inference/run.sh
```

`run.sh` starts `serve.py` in the background and then runs `evaluate.py`.
It needs the ACT checkpoint from `examples/01_train_and_eval_act_on_aloha/run.sh`.

To run the two processes yourself:

## 1. Policy server

In the first terminal:

```bash
python serve.py
```

`serve.py` loads `checkpoints/act_aloha` and listens on TCP port 5000 on all
interfaces. The server continues running until interrupted with `Ctrl+C`.

## 2. Simulation evaluation

Keep the server running and use a second terminal:

```bash
python evaluate.py
```

ACT and ALOHA share the base interpreter but run in separate processes.
Example 05 (π0.5 / Tabletop-Sim) uses the same serve plus evaluate split because
those components cannot share an environment. Example 04 evaluates SmolVLA on
LIBERO in-process and does not need a server.

On one machine, connect to `127.0.0.1:5000`. Across machines, change the client
address to a reachable policy-server IP:

```python
policy = vla.connect_policy("192.168.1.20:5000")
```

You can also pass an address directly to evaluation:

```python
bench = vla.load_env("aloha_transfer")
bench.evaluate("192.168.1.20:5000", output_dir="results/remote")
```

## Supported transports

| Transport | Server address | Evaluation address | Scope |
| --- | --- | --- | --- |
| TCP | `0.0.0.0:5000` | `host:5000` | Default; local or across machines |
| HTTP | `http://0.0.0.0:8000` | `http://host:8000` | JSON/HTTP; across machines |
| HTTPS | `https://0.0.0.0:8443` | `https://host:8443` | Requires certificate configuration |
| SHM | `shm://act_policy` | `shm://act_policy` | Same machine only |

HTTP/HTTPS servers need additional dependencies installed in the server environment:

```bash
vlastudio env run --policy act -- python -m pip install -e ".[serve-http]"
```

Update the address in both scripts. For HTTPS, set `ILSTD_SSL_KEYFILE` and
`ILSTD_SSL_CERTFILE` to the certificate paths. SHM does not use the network;
both processes must have access to the same shared memory.

TCP transport uses Python pickle and should run on a trusted network or locally.
For untrusted networks, use HTTPS or a trusted tunnel.

## Troubleshooting

- `vlastudio: command not found`: from the repository root, `python -m pip install -e .`
- `python: can't open file 'examples/...'`: run `run.sh` from the repository root, not from this directory.
- Missing `checkpoints/act_aloha`: run `examples/01_train_and_eval_act_on_aloha/run.sh` first.
- `Connection refused` / evaluate starts before the server listens: start `serve.py` in another terminal and wait until it is listening, then run `evaluate.py`.
- Port 5000 already in use: stop the leftover server (`kill %1` from the same shell, or kill the process bound to 5000).
- Stale `No module named 'examples/...'` imports: `unset PYTHONPATH` and rerun.

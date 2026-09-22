# 5. Evaluation with Policy Server

The **Policy Server** is a key feature for robust and flexible deployment. It decouples the policy inference from the evaluation environment, allowing you to run the computationally heavy model on a powerful machine (the "server") while controlling a robot or simulation from a lighter machine (the "client").

## Use Cases

*   **Resource Constraints**: Run evaluation on a robot's onboard computer, which may lack a powerful GPU, while inference happens on a nearby workstation.
*   **Centralized Model Serving**: A single, powerful server can provide inference for multiple client robots.
*   **Simplified Client Setup**: The client machine does not need to have the model, its dependencies, or large datasets.

## Architecture

```
+--------------------------+                           +-------------------------+
|      Client Machine      |                           |      Server Machine     |
| (e.g., Robot NUC or PC)  |                           | (e.g., Workstation w/ GPU)|
+--------------------------+                           +-------------------------+
|      `vlastudio infer`      |                           | `vlastudio serve`|
|            or            |                           |                         |
|      `vlastudio evaluate`       |                           |      Loads Model &      |
|                          |                           |       Normalizers       |
|      `PolicyClient`      |                           |                         |
|                          |                           |      `PolicyServer`     |
| (Connects to server)     |                           |   (Listens for clients) |
+------------+-------------+                           +------------+------------+
             |                                                        ^
             |                (Network: TCP/IP)                       |
             +------------------------->------------------------------+
             |      Sends `MetaObs`      (Serialized with pickle)      |
             |                                                        |
             +-------------------------<------------------------------+
             | Receives `MetaAction[]` (Serialized with pickle)      |
             |                                                        |

```

## Step 1: Start the Policy Server

On a machine with a GPU and access to the model checkpoints:
```bash
# Listen on all network interfaces on port 5000
vlastudio serve --runtime current \
    -m ckpt/act_sim_transfer_cube_scripted_zscore_example \
    --host 0.0.0.0 \
    --port 5000
```
The server will load the model and print `🚀 Policy Server started...`, waiting for clients to connect.

## Step 2: Run the Evaluation Client

On the client machine (which can be the same machine or a different one on the same network), run your evaluation script, but replace the model path with the server's IP address and port.

### Example: Simulation Client
```bash
# Replace 192.168.1.101 with your server's IP address
vlastudio evaluate --runtime current \
    -m 192.168.1.101:5000 \
    -e aloha \
    --num_rollout 10
```

### Example: Real-World Client
```bash
# Replace 192.168.1.101 with your server's IP address
vlastudio infer --runtime current \
    --model_name_or_path 192.168.1.101:5000 \
    -r agilex_aloha \
    --task agilex_transfer_cube
```

The scripts automatically detect that the `-m` argument is a network address and will instantiate `PolicyClient` instead of loading the model locally.

## Key Arguments for `vlastudio serve`

*   `-m, --model_name_or_path`: Path to the model checkpoint directory.
*   `--host`: The IP address to bind to. `0.0.0.0` makes it accessible to other machines on the network.
*   `--port`: The network port to listen on.
*   `--dataset_id`: If the model was trained on multiple datasets, specifies which dataset's normalizer to use.
*   `--device`: The device to run inference on (e.g., `cuda`, `cpu`).

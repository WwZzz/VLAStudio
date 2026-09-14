# Installation
```shell
cd benchmark/libero
uv venv
source .venv/bin/activate
uv sync
cd ../../third_party/libero
uv pip install -e .
uv pip install mujoco==3.3.2
cd ../..
```

### TroubleShooting
If the installation failed due to error like `Compatibility with CMake < 3.5 has been removed from CMake. Update the VERSION argument <min> value.`, please set the configurations by
```shell
export CMAKE_POLICY_VERSION_MINIMUM=X.X # your cmake version

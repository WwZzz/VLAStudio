# Installation
```shell
cd benchmark/robomimic
uv sync
uv pip install git+https://github.com/ARISE-Initiative/robomimic.git # 0.4.0
uv pip install mujoco==2.3.2 mujoco-python-viewer
```

# TroubleShooting
If the installation failed due to error like `Compatibility with CMake < 3.5 has been removed from CMake. Update the VERSION argument <min> value.`, please set the configurations by
```shell
export CMAKE_POLICY_VERSION_MINIMUM=X.X # your cmake version
```
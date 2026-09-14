# Installation
```shell
# Linux only or mac
pip install gymnasium-robotics
```

# TroubleShooting
for error `mujoco.FatalError: an OpenGL platform library has not been loaded into this process, this most likely means that a valid OpenGL context has not been created before mjr_makeContext was called` on Linux, please run
```shell
export MUJOCO_GL=egl
```

# Data Preparation
```shell
# install minari
pip install minari

# download dataset
minari download D4RL/door/human-v2 # data will be stored at ~/.minari/datasets/D4RL/door/human-v2
minari download D4RL/door/expert-v2
minari download D4RL/door/clone-v2

# view dataset
minari show D4RL/door/human-v2

# 


```

# Installation
```
cd benchmark/metaworld
uv sync
uv pip install metaworld
```

# Trouble Shooting
- `mujoco.FatalError: gladLoadGL error`: running the command in the shell `export MUJOCO_GL=egl`

- Wrong view from top instead of the corner4: 
This is as unexpected for the environment. Please copy the line below:
```
<camera name="corner4" fovy="60" mode="fixed" pos="0.75 0.075 0.7" euler="3.9 2.3 0.6"/>
```
Then, you need to paste this line to 
`path_to_env/lib/python/site-packages/metaworld/assets/objects/assets/xyz_base.xml`
```xml
...
<camera pos="0 0.5 1.5" name="topview" />
<camera name="corner" mode="fixed" pos="-1.1 -0.4 0.6" xyaxes="-1 1 0 -0.2 -0.2 -1"/>
<camera name="corner2" fovy="60" mode="fixed" pos="1.3 -0.2 1.1" euler="3.9 2.3 0.6"/>
<camera name="corner3" fovy="45" mode="fixed" pos="0.9 0 1.5" euler="3.5 2.7 1"/>
<camera name="corner4" fovy="60" mode="fixed" pos="0.75 0.075 0.7" euler="3.9 2.3 0.6"/> <!--Paste it here-->
...
```
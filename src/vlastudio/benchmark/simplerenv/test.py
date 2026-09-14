import os
os.environ['MS2_REAL2SIM_ASSET_DIR'] = os.path.join(os.path.dirname(__file__), 'ManiSkill2_real2sim', 'data')
import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
import cv2

env = simpler_env.make('google_robot_pick_coke_can')
obs, reset_info = env.reset()
instruction = env.get_language_instruction()
print("Reset info", reset_info)
print("Instruction", instruction)

done, truncated = False, False
while not (done or truncated):
   # action[:3]: delta xyz; action[3:6]: delta rotation in axis-angle representation;
   # action[6:7]: gripper (the meaning of open / close depends on robot URDF)
   image = get_image_from_maniskill2_obs_dict(env, obs)
   # 图像已经是 RGB 格式，OpenCV imshow 需要 BGR 格式，所以转换为 BGR
   image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
   # 可视化图像
   cv2.imshow('Environment View', image_bgr)
   cv2.waitKey(1)  # 非阻塞，允许程序继续运行
   action = env.action_space.sample() # replace this with your policy inference
   obs, reward, done, truncated, info = env.step(action) # for long horizon tasks, you can call env.advance_to_next_subtask() to advance to the next subtask; the environment might also autoadvance if env._elapsed_steps is larger than a threshold
   new_instruction = env.get_language_instruction()
   if new_instruction != instruction:
      # for long horizon tasks, we get a new instruction when robot proceeds to the next subtask
      instruction = new_instruction
      print("New Instruction", instruction)

episode_stats = info.get('episode_stats', {})
print("Episode stats", episode_stats)
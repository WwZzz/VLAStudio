"""Isolated dependency/import and optional execution checks for env check."""
import importlib
import json
import sys


IMPORTS = {
    'data': ['numpy', 'h5py', 'pyarrow.parquet', 'av', 'torch', 'torchvision'],
    'base': ['torch', 'torchvision', 'transformers', 'h5py', 'vlastudio.policy.act'],
    'diffusion_policy': ['diffusers', 'vlastudio.policy.diffusion_policy'],
    'smolvla': ['lerobot.policies.smolvla.modeling_smolvla', 'vlastudio.policy.smolvla'],
    'openpi': ['openpi.models_pytorch.pi0_pytorch', 'vlastudio.policy.openpi'],
    'openvla': ['vlastudio.policy.openvla'],
    'fastwam': ['vlastudio.policy.fastwam'],
    'aloha': ['dm_control.mujoco', 'vlastudio.benchmark.aloha'],
    'libero': ['libero.libero.benchmark', 'vlastudio.benchmark.libero'],
    'metaworld': ['metaworld', 'vlastudio.benchmark.metaworld'],
    'rlbench': ['pyrep', 'rlbench', 'vlastudio.benchmark.rlbench'],
    'behavior1k': ['omnigibson'],
    'robotwin': ['sapien', 'mplib', 'pytorch3d.ops', 'curobo.wrap.reacher.motion_gen', 'vlastudio.benchmark.robotwin'],
    'carla': ['carla', 'cv2', 'py_trees', 'vlastudio.benchmark.carla'],
}


def smoke(name):
    if name == 'carla':
        from vlastudio.benchmark.carla.server import CarlaServer
        from queue import Queue
        import carla
        import numpy as np
        with CarlaServer() as server:
            world = server.connect().get_world()
            settings = world.get_settings()
            settings.synchronous_mode, settings.fixed_delta_seconds = True, .05
            world.apply_settings(settings)
            vehicle = camera = None
            try:
                library = world.get_blueprint_library()
                vehicle = world.spawn_actor(library.find('vehicle.lincoln.mkz_2020'), world.get_map().get_spawn_points()[0])
                blueprint = library.find('sensor.camera.rgb')
                blueprint.set_attribute('image_size_x', '160')
                blueprint.set_attribute('image_size_y', '90')
                camera = world.spawn_actor(blueprint, carla.Transform(carla.Location(x=1.5, z=2.)), attach_to=vehicle)
                frames = Queue()
                camera.listen(frames.put)
                vehicle.apply_control(carla.VehicleControl(throttle=.3))
                for _ in range(20):
                    world.tick()
                    image = np.frombuffer(frames.get(timeout=10).raw_data, dtype=np.uint8).reshape(90, 160, 4)
                assert image[:, :, :3].std() > 1, 'CARLA camera did not render a scene'
            finally:
                if camera is not None:
                    camera.stop()
                    camera.destroy()
                if vehicle is not None:
                    vehicle.destroy()
    elif name == 'aloha':
        from vlastudio.benchmark.aloha.sim_env import make_sim_env, BOX_POSE
        from vlastudio.benchmark.aloha.utils import sample_box_pose
        BOX_POSE[0] = sample_box_pose()
        env = make_sim_env('sim_transfer_cube_scripted')
        try:
            obs = env.reset()
            assert obs.observation['images']['top'].shape == (480, 640, 3)
            env.step(obs.observation['qpos'])
        finally:
            env.close()
    elif name == 'metaworld':
        import metaworld
        import numpy as np
        benchmark = metaworld.ML1('reach-v3', seed=0)
        env = benchmark.train_classes['reach-v3'](render_mode='rgb_array')
        try:
            env.set_task(benchmark.train_tasks[0])
            env.reset()
            env.step(np.zeros(env.action_space.shape))
            image = env.render()
            assert image.ndim == 3 and image.shape[-1] == 3
        finally:
            env.close()
    elif name == 'libero':
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        from vlastudio.benchmark.libero import load_task_init_states
        from pathlib import Path
        import numpy as np
        suite = benchmark.get_benchmark_dict()['libero_spatial']()
        task = suite.get_task(0)
        env = OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file), camera_heights=128, camera_widths=128)
        try:
            env.reset()
            env.set_init_state(load_task_init_states(suite, 0)[0])
            obs, _, _, _ = env.step(np.zeros(7))
            assert obs['agentview_image'].shape == (128, 128, 3)
        finally:
            env.close()
    elif name == 'rlbench':
        from rlbench.environment import Environment
        from rlbench.observation_config import ObservationConfig
        from rlbench.action_modes.action_mode import MoveArmThenGripper
        from rlbench.action_modes.arm_action_modes import JointVelocity
        from rlbench.action_modes.gripper_action_modes import Discrete
        from rlbench.tasks import ReachTarget
        import numpy as np
        config = ObservationConfig()
        config.set_all(True)
        # OpenCV's wheel sets its own Qt path on import; CoppeliaSim needs its Qt.
        import cv2
        import os
        os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = os.environ['COPPELIASIM_ROOT']
        env = Environment(MoveArmThenGripper(JointVelocity(), Discrete()), obs_config=config, headless=True)
        try:
            env.launch()
            task = env.get_task(ReachTarget)
            _, obs = task.reset()
            assert obs.front_rgb is not None
            task.step(np.r_[np.zeros(7), 1.])
        finally:
            env.shutdown()
    elif name == 'robotwin':
        from types import SimpleNamespace
        from vlastudio.benchmark.robotwin import RoboTwinEnv
        config = SimpleNamespace(task='adjust_bottle', max_timesteps=10, seed=0,
                                 use_planner=True, expert_check=False, image_size=[128, 128])
        env = RoboTwinEnv(config)
        try:
            obs = env.reset()
            assert obs.image is not None and obs.image.shape[-2:] == (128, 128)
            assert obs.image.max() > 0, 'Rendered image is empty'
            assert env.use_planner, 'Planner fallback is not a successful planner check'
            for arm in ('left', 'right'):
                assert getattr(env.task_env.robot, arm + '_planner').motion_gen is not None
            env.step(obs.state)
        finally:
            env.close()
    elif name == 'behavior1k':
        import omnigibson as og
        try:
            env = og.Environment(configs={'scene': {'type': 'Scene'}})
            env.reset()
            og.sim.step()
            print('VLASTUDIO_SMOKE_COMPLETED:behavior1k', flush=True)
        finally:
            og.shutdown(due_to_signal=True)
    else:
        raise ValueError(f'No execution smoke check for {name}; import checks do not prove model inference or simulator reset.')
    print(f'VLASTUDIO_SMOKE_COMPLETED:{name}', flush=True)


def main():
    import os
    os.environ.setdefault('MUJOCO_GL', 'egl' if sys.platform == 'linux' else 'glfw')
    from vlastudio.deploy.interpreter.dependencies import inspect_environment
    with open(sys.argv[1], encoding='utf-8') as source:
        profile = json.load(source)
    issues, _ = inspect_environment(profile)
    if issues:
        raise RuntimeError('\n'.join(issues))
    for module in ['vlastudio', 'loguru', 'yaml']:
        importlib.import_module(module)
    checks = profile.get('checks', [])
    for name in checks:
        if name.startswith('import:'):
            importlib.import_module(name.split(':', 1)[1])
            print(f'PASS {name}', flush=True)
            continue
        if name not in IMPORTS:
            raise ValueError(f'Unknown check {name}')
        for module in IMPORTS[name]:
            importlib.import_module(module)
        print(f'PASS imports: {name}', flush=True)
    if '--smoke' in sys.argv:
        executed = False
        # Each simulator has its own process in normal use; do not mix their graphics contexts.
        simulation = [name for name in checks if name in ('aloha', 'libero', 'metaworld', 'rlbench', 'behavior1k', 'robotwin', 'carla')]
        import subprocess
        for name in simulation:
            # First Isaac startup can take several minutes to initialize rendering.
            timeout = 900 if name == 'behavior1k' else 300
            result = subprocess.run([sys.executable, '-c', f'from vlastudio.deploy.interpreter.probe import smoke; smoke({name!r})'], capture_output=True, text=True, timeout=timeout)
            print(result.stdout, end='', flush=True)
            print(result.stderr, end='', file=sys.stderr, flush=True)
            result.check_returncode()
            if f'VLASTUDIO_SMOKE_COMPLETED:{name}' not in result.stdout.splitlines():
                raise RuntimeError(f'{name} exited before completing reset/step; exit code zero alone does not prove success.')
            print(f'PASS reset/step: {name}', flush=True)
            executed = True
        if any(name in checks for name in ('base', 'smolvla', 'openpi', 'fastwam', 'openvla', 'diffusion_policy')):
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA unavailable; GPU execution was not verified.')
            a = torch.randn(64, 64, device='cuda', requires_grad=True)
            (a @ a.T).square().mean().backward()
            torch.cuda.synchronize()
            print('PASS CUDA forward/backward (not a policy inference test)', flush=True)
            executed = True
        if not executed:
            raise ValueError('No execution smoke check declared by this recipe; only imports were checked.')


if __name__ == '__main__':
    main()

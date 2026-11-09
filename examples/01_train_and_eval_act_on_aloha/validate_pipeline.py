"""Small pipeline regression using recorded simulation motion, not expert data.

Run with the ACT/ALOHA dependencies installed. This tests training plumbing,
checkpoint saving and simulator contracts; it does not measure learned success.
"""
import argparse
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import h5py
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    data = out / 'sim_transfer_cube_scripted'
    data.mkdir()
    import vlastudio as vla
    from vlastudio.benchmark.aloha import AlohaSimEnv
    config = SimpleNamespace(task='sim_transfer_cube_scripted', ctrl_space='joint',
                             ctrl_type='abs', camera_ids=[0], image_size=[640, 480])
    files = []
    for episode in range(2):
        env = AlohaSimEnv(config)
        try:
            obs = env.reset()
            states, actions, images = [], [], []
            initial = obs.state.copy()
            for step in range(8):
                action = initial.copy()
                action[0] += .02 * np.sin((step + 1) * np.pi / 8)
                states.append(obs.state.copy())
                images.append(obs.image[0].transpose(1, 2, 0).copy())
                actions.append(action)
                result = env.step({'action': action})
                assert len(result) == 4
                obs = SimpleNamespace(**result[0])
            path = data / f'episode_{episode}.hdf5'
            with h5py.File(path, 'w') as f:
                f.attrs['sim'] = True
                f.create_dataset('action', data=actions)
                f.create_dataset('observations/qpos', data=states)
                f.create_dataset('observations/images/top', data=images)
            files.append(str(path))
        finally:
            # Free the renderer while its executor is still alive.
            env.env.physics.free()
            env.close()
    root = Path(__file__).resolve().parents[2]
    task = yaml.safe_load((root / 'src/vlastudio/configs/task/sim_transfer_cube_scripted.yaml').read_text())
    task['datasets'][0]['args'].update(dataset_path_list=files, expected_episodes=2)
    (out / 'task.yaml').write_text(yaml.safe_dump(task))
    training = yaml.safe_load((root / 'examples/01_train_and_eval_act_on_aloha/training_config.yaml').read_text())
    training.update(max_steps=2, per_device_train_batch_size=1, dataloader_num_workers=0,
                    save_steps=2, logging_steps=1, report_to='none', use_cpu=args.cpu,
                    resume_from_checkpoint=False, dataloader_pin_memory=not args.cpu)
    (out / 'training.yaml').write_text(yaml.safe_dump(training))
    policy = vla.load_policy('act')
    dataset = vla.load_dataset(out / 'task.yaml')
    vla.train(policy, dataset, out / 'training.yaml', output_dir=out / 'checkpoint')
    environment = yaml.safe_load((root / 'src/vlastudio/configs/env/aloha_transfer.yaml').read_text())
    environment.pop('runtime', None)
    environment['args']['max_timesteps'] = 8
    (out / 'environment.yaml').write_text(yaml.safe_dump(environment))
    evaluation = vla.load_env(out / 'environment.yaml').evaluate(
        policy, output_dir=out / 'inference', num_rollout=1,
        device='cpu' if args.cpu else 'cuda')
    (out / 'result.json').write_text(json.dumps(dict(
        training_steps=2, episodes=2, frames_per_episode=8, expert_data=False,
        source=str(vla.__file__), checkpoint=str(policy.checkpoint),
        inference=evaluation.metrics), indent=2))
    print('TRAINING_AND_INFERENCE_PIPELINE_PASSED', flush=True)


if __name__ == '__main__':
    main()

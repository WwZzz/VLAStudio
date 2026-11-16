"""Exercise real simulator devices through the WebUI session supervisor.

Use the chosen simulator's Python and graphics environment. The TCP test policy
holds the current pose; this validates deployment plumbing, not learned success.
"""
import argparse
import json
from pathlib import Path
import threading
import time
import uuid

import h5py
import numpy as np

from vlastudio.webui.session import Session
from vlastudio.webui.profiles import ROOT, catalog
from vlastudio.deploy.comm.server import PolicyServer


def wait(session, action=None, timeout=90):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = session.state()
        session.preview.state()  # Same preview heartbeat as a polling browser.
        if not state['active']:
            raise RuntimeError('\n'.join(state['logs'][-35:]))
        if action is None or action in state['actions']:
            return state
        time.sleep(.05)
    raise TimeoutError(json.dumps(session.state(), ensure_ascii=False))


class HoldPolicy:
    def __init__(self, profile):
        self.profile, self.calls = profile, 0

    def inference(self, obs):
        self.calls += 1
        if self.profile == 'metaworld':
            action, space, kind = np.array([0., 0., 0., -1.]), 'ee', 'delta'
        elif self.profile == 'aloha_sim_joint':
            action, space, kind = np.asarray(obs.state_joint).reshape(-1), 'joint', 'abs'
        else:
            action, space, kind = np.asarray(obs.state_ee).reshape(-1), 'ee', 'abs'
        return [np.array([dict(action=action, ctrl_space=space, ctrl_type=kind)], dtype=object)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['metaworld', 'rlbench', 'aloha_sim_joint'], required=True)
    parser.add_argument('--task')
    parser.add_argument('--result', type=Path, required=True)
    args = parser.parse_args()
    settings = catalog()['robot_options'][args.profile]
    task = args.task or settings['tasks'][0]
    result = dict(profile=args.profile, task=task, learned_policy=False)
    if args.profile != 'aloha_sim_joint':
        session = Session()
        name = f'verify-{args.profile}-{uuid.uuid4().hex[:8]}'
        try:
            session.start('collect', dict(robot=args.profile, task=task, output=name,
                                         teleop=settings['teleop'], rate=20))
            wait(session, 'tele_run')
            session.action(session.sid, 'tele_run')
            wait(session, 'record')
            session.action(session.sid, 'record')
            wait(session, 'finish')
            for _ in range(12):
                session.keyboard(dict(id=session.sid, owner='validation', keys=['w']))
                time.sleep(.05)
            session.keyboard(dict(id=session.sid, owner='validation', keys=[]))
            deadline = time.monotonic() + 45
            while not session.preview.images and time.monotonic() < deadline:
                wait(session)
                time.sleep(.1)
            wait(session, 'finish')
            assert session.preview.images, f'No WebUI camera preview: {session.preview.state()}'
            session.action(session.sid, 'tele_pause')
            wait(session, 'save')
            session.action(session.sid, 'save')
            wait(session, 'home')
            session.action(session.sid, 'home')
            wait(session, 'tele_run')
            result['collect_pause_save_reset'] = 'passed'
            result['collection_runtime'] = session.runtime
        finally:
            session.shutdown()
        files = list((ROOT / 'data/webui' / name).rglob('*.hdf5'))
        assert files, 'No saved HDF5 episode'
        with h5py.File(files[0]) as f:
            shapes = {}
            f.visititems(lambda n, o: shapes.update({n: list(o.shape)}) if isinstance(o, h5py.Dataset) else None)
        result['recorded_shapes'] = shapes
        result['dataset'] = str(files[0])
    policy = HoldPolicy(args.profile)
    # Let the server bind an available port atomically.
    server = PolicyServer(policy, host='127.0.0.1', port=0)
    thread = threading.Thread(target=server.start, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while not server.running and time.monotonic() < deadline:
        time.sleep(.01)
    assert server.running
    port = server.server_socket.getsockname()[1]
    session = Session()
    try:
        session.start('infer', dict(robot=args.profile, task=task,
                                   policy=f'127.0.0.1:{port}', rate=20, output=''))
        wait(session, 'run')
        session.action(session.sid, 'run')
        wait(session, 'pause')
        deadline = time.monotonic() + 15
        while policy.calls < 3 and time.monotonic() < deadline:
            wait(session)
            time.sleep(.1)
        assert policy.calls >= 3, 'No policy observation/action round trips'
        session.action(session.sid, 'pause')
        wait(session, 'reset')
        session.action(session.sid, 'reset')
        wait(session, 'run')
        result['tcp_inference_pause_reset'] = 'passed'
        result['inference_runtime'] = session.runtime
        result['policy_calls'] = policy.calls
    finally:
        session.shutdown()
        server.stop()
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2))
    print('WEBUI_SESSION_PASSED', result, flush=True)


if __name__ == '__main__':
    main()

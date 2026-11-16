"""Inference trial boundaries, recovery and recording provenance."""
import io
import json
import os
import threading
from types import SimpleNamespace

import pytest

from vlastudio.webui.inference import InferenceEpisodes, is_simulation_robot


class Recorder:
    def __init__(self, enabled=True):
        self.has_recorder = enabled
        self.commands = []
        self.index = 0
        self.fail_save = False

    def send_recorder_cmd(self, command):
        self.commands.append(command)
        if command == 'save_episode' and self.fail_save:
            return None
        if command == 'discard_episode':
            self.index += 1
        return ({'resume_recording':'resumed', 'pause_recording':'paused',
                 'save_episode':'saved', 'exit_discard':'exit_ok', 'discard_episode':'discarded'}[command], 10, self.index)


def controller(enabled=True, simulation=False):
    rt = Recorder(enabled)
    calls, phases = [], []
    manager = SimpleNamespace(reset=lambda:calls.append('policy reset'))
    c = InferenceEpisodes(rt, manager, lambda:calls.append('robot reset'), phases.append,
                          reset_after_trial=simulation)
    return c, rt, calls, phases


def test_pause_resume_keeps_trial_and_stop_waits_for_decision():
    c, rt, calls, phases = controller()
    assert rt.commands == []
    c.handle('');c.handle('')
    assert c.state == 'paused' and rt.commands == ['resume_recording', 'pause_recording']
    c.handle('')
    assert c.state == 'running' and rt.index == 0 and len(calls) == 1
    c.handle('finish')
    assert c.state == 'infer_save' and 'save_episode' not in rt.commands
    c.handle('')
    assert c.state == 'infer_save'  # Cannot resume a stopped trial before deciding.
    c.handle('s')
    assert c.state == 'infer_ready' and calls == ['policy reset']
    c.handle('')
    assert c.state == 'running' and rt.index == 1
    assert rt.commands[-2:] == ['discard_episode', 'resume_recording']
    assert phases.count('infer_save') >= 1


def test_discard_finishes_only_current_trial_without_resetting_robot():
    c, rt, calls, _ = controller()
    c.handle('');c.handle('finish');c.handle('d')
    assert c.state == 'infer_ready'
    assert rt.commands[-1] == 'exit_discard' and 'robot reset' not in calls
    c.handle('')
    assert c.state == 'running' and rt.index == 1


def test_preview_without_recording_never_prompts_or_creates_recordings():
    c, rt, calls, phases = controller(False)
    c.handle('');c.handle('');c.handle('');c.handle('finish')
    assert c.state == 'infer_ready' and 'infer_save' not in phases
    assert rt.commands == []


def test_failed_save_preserves_pending_trial_and_allows_retry():
    c, rt, calls, _ = controller()
    c.handle('');c.handle('finish');rt.fail_save = True;c.handle('s')
    assert c.state == 'infer_save' and not c.needs_episode
    rt.fail_save = False;c.handle('s')
    assert c.state == 'infer_ready' and c.needs_episode


def test_failed_pause_or_stop_never_reenables_policy_control():
    for command in ('', 'finish'):
        c, rt, _, _ = controller()
        c.handle('')
        rt.send_recorder_cmd = lambda _:None
        c.handle(command)
        assert c.state == ('paused' if command == '' else 'infer_save')


@pytest.mark.parametrize('decision', ['s','d'])
def test_simulation_decision_resets_after_recording_is_finished(decision):
    c,rt,calls,_=controller(simulation=True)
    c.on_reset=lambda:calls.append(('robot reset',rt.commands[-1]))
    c.handle('');c.handle('finish')
    assert not any(isinstance(x,tuple) for x in calls)
    c.handle(decision)
    assert c.state=='infer_ready' and c.needs_episode
    assert calls[-1]==('robot reset','save_episode' if decision=='s' else 'exit_discard')
    c.handle('');assert c.state=='running' and rt.index==1


def test_simulation_paused_reset_discards_and_finishes_trial():
    c,rt,calls,phases=controller(simulation=True)
    c.handle('');c.handle('');c.handle('r')
    assert c.state=='infer_ready' and c.needs_episode
    assert rt.commands[-1]=='exit_discard' and 'save_episode' not in rt.commands
    assert calls[-1]=='robot reset' and 'infer_save' not in phases


def test_recording_failure_prevents_automatic_reset_and_reset_failure_does_not_repeat_save():
    c,rt,calls,_=controller(simulation=True)
    c.handle('');c.handle('finish');rt.fail_save=True;c.handle('s')
    assert c.state=='infer_save' and 'robot reset' not in calls
    rt.fail_save=False
    def failed_reset():raise RuntimeError('reset not acknowledged')
    c.on_reset=failed_reset;c.handle('s')
    assert c.state=='infer_ready' and c.needs_episode
    count=rt.commands.count('save_episode');c.handle('s')
    assert rt.commands.count('save_episode')==count


def test_simulation_detection_never_uses_profile_names_or_camera_types():
    sim={'type':'vlastudio.deploy.robot.bench_robot.BenchmarkRobot'}
    real={'type':'vlastudio.deploy.robot.alicia_d.AliciaD','name':'simulation_demo'}
    camera={'type':'vlastudio.deploy.camera.dummy_camera.DummyCamera'}
    assert is_simulation_robot([sim,camera])
    assert not is_simulation_robot([real]) and not is_simulation_robot([sim,real])
    assert not is_simulation_robot([camera])


@pytest.mark.skipif(os.name != 'posix', reason='Linux dataset storage')
def test_recording_origins_are_distinct_and_legacy_collection_is_reused(tmp_path, monkeypatch):
    from vlastudio.webui import datasets
    monkeypatch.setattr(datasets, 'data_directory', lambda:tmp_path)
    collection = datasets.scope('dummy')
    inference = datasets.scope('dummy', recording_source='inference')
    a,b = datasets.identity(collection,''), datasets.identity(inference,'')
    assert a['dataset'] != b['dataset'] and b['dataset'].startswith('inference_')
    legacy = dict(a, dataset='old-collection');legacy.pop('recording_source')
    folder = tmp_path/legacy['dataset'];folder.mkdir();datasets.write_manifest(folder,legacy)
    assert datasets.identity(collection,'')['dataset'] == legacy['dataset']
    datasets.ensure(b)
    assert {r['recording_source'] for r in datasets.choices(inference)} == {'inference'}
    assert {r['recording_source'] for r in datasets.choices(collection)} == {'collection'}
    # Legacy inference is recognized from its producer manifest, not its name.
    (folder/'eval_real_manifest.json').write_text(json.dumps({'script':'vlastudio infer'}))
    assert datasets.recording_source(folder) == 'inference'
    assert datasets.identity(collection,'')['dataset'] != legacy['dataset']


@pytest.mark.skipif(os.name != 'posix', reason='Linux session supervisor')
def test_stop_pauses_control_and_keeps_policy_process_connected():
    from vlastudio.webui.session import Session
    from vlastudio.webui.bridge import PHASES, RUN, PAUSE
    preview=SimpleNamespace(lock=threading.RLock(),devices={})
    s=Session(preview,lambda _:None)
    terminated=[]
    proc=SimpleNamespace(stdin=io.BytesIO(),terminate=lambda:terminated.append(True))
    s.proc=proc;s.sid='session';s.mode='infer';s.robot='dummy';s.output='inference-recording'
    s.options={'record_inference':True};s.box=bytearray([RUN,RUN,PHASES.index('running'),0,255])
    s.reset_after_pause=False;s.reset_ok=False;s.gated=False;s.retaking=False;s.episode=0
    s.keyboard_arms=0;s.keyboard_bindings=[];s.keyboard_base_bindings=[];s.keyboard_axes=[]
    assert 'save' not in s.state()['actions']
    s.action('session','stop')
    assert s.proc is proc and not terminated and not s.stopping
    assert s.box[0]==PAUSE and proc.stdin.getvalue()==b'finish\n'
    assert 'disconnect' not in s.state()['actions']
    s.box[2]=PHASES.index('paused');s.box[3]=1;s.pending=None
    assert 'save' not in s.state()['actions'] and 'stop' in s.state()['actions']
    s.box[2]=PHASES.index('infer_save')
    assert {'save','discard'} <= set(s.state()['actions'])
    assert 'run' not in s.state()['actions']
    from vlastudio.webui.bridge import RESET_A, RESETTING
    s.box[2]=PHASES.index('infer_ready');s.box[0]=RESET_A;s.box[1]=RESETTING;s.box[4]=RESET_A
    assert s.state()['phase']=='homing' and 'run' not in s.state()['actions']
    s.box[1]=PAUSE
    assert s.state()['phase']=='infer_ready' and 'run' in s.state()['actions']


@pytest.mark.skipif(os.name != 'posix', reason='Linux dataset storage')
def test_saved_inference_videos_visible_while_policy_stays_loaded(tmp_path, monkeypatch):
    from vlastudio.webui import episodes
    monkeypatch.setattr(episodes, 'root_directory', lambda:tmp_path)
    directory=tmp_path/'inference-trials';(directory/'video').mkdir(parents=True)
    (directory/'eval_real_manifest.json').write_text(json.dumps({'script':'vlastudio infer'}))
    for n in range(2):(directory/'video'/f'real_eval_episode_{n:04d}.mp4').write_bytes(b'video placeholder')
    rows=episodes.catalog('inference-trials',active_episode=1)
    assert [r['index'] for r in rows]==[0]
    assert rows[0]['recording_source']=='inference' and not rows[0]['rerecordable']
    assert [r['index'] for r in episodes.catalog()]==[0,1]

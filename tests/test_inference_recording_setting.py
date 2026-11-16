import json
import threading
import time
from types import SimpleNamespace
import pytest

pytest.importorskip('fcntl')
from vlastudio.webui.recording_control import RecordingControl, write_json
from vlastudio.webui.session import Session


def test_live_recording_toggle_preserves_policy_robot_and_episode_decisions(tmp_path,monkeypatch):
    from vlastudio.webui import session as module
    monkeypatch.setattr(module,'data_directory',lambda:tmp_path)
    events=[]
    runtime=SimpleNamespace(has_recorder=False,send_recorder_cmd=lambda _:('paused',0,None))
    def stop():runtime.has_recorder=False;events.append('stop recorder')
    runtime.stop_session_recorder=stop
    def start(destination):runtime.has_recorder=True;events.append(('start recorder',destination))
    controller=SimpleNamespace(state='infer_ready',runtime=runtime,needs_episode=True,publish_phase=lambda _:None)
    control=RecordingControl(tmp_path,controller,start)
    session=object.__new__(Session);session.lock=threading.RLock();session.mode='infer';session.sid='policy'
    session.proc=object();policy=session.proc;session.mailbox_path=tmp_path/'mailbox';session.output=''
    session.options={'record_inference':False};session.state=lambda:dict(phase=controller.state,recording_configurable=True)
    stop_poll=threading.Event()
    def poll():
        while not stop_poll.is_set():control.poll();time.sleep(.005)
    worker=threading.Thread(target=poll);worker.start()
    try:
        for enabled in (True,False,True):
            session.configure_recording('policy',enabled,'dataset')
            assert session.proc is policy and session.sid=='policy'
            assert session.options['record_inference']==enabled
            assert runtime.has_recorder==enabled
            assert session.output==('dataset' if enabled else '')
        assert len(events)==3
        assert controller.needs_episode is False
        for phase in ('running','paused','infer_save'):
            controller.state=phase
            with pytest.raises(ValueError):session.configure_recording('policy',False,'dataset')
        assert len(events)==3
    finally:stop_poll.set();worker.join(2)


def test_worker_rejects_midtrial_and_preserves_loaded_policy(tmp_path):
    runtime=SimpleNamespace(has_recorder=True,stop_session_recorder=lambda:pytest.fail('must not stop'))
    controller=SimpleNamespace(state='infer_save',runtime=runtime,publish_phase=lambda _:None)
    control=RecordingControl(tmp_path,controller,lambda _:pytest.fail('must not start'))
    write_json(tmp_path/'recording-request.json',dict(id='change',enabled=False,output=''))
    control.poll()
    response=json.loads((tmp_path/'recording-response.json').read_text())
    assert not response['ok'] and runtime.has_recorder


def test_worker_failed_start_reports_error(tmp_path):
    runtime=SimpleNamespace(has_recorder=False)
    controller=SimpleNamespace(state='infer_ready',runtime=runtime,publish_phase=lambda _:None)
    def start(_):raise OSError('Cannot create recording')
    control=RecordingControl(tmp_path,controller,start)
    write_json(tmp_path/'recording-request.json',dict(id='change',enabled=True,output='dataset'))
    control.poll()
    response=json.loads((tmp_path/'recording-response.json').read_text())
    assert not response['ok'] and not runtime.has_recorder

"""Exercise the real video and HDF5 writers with an isolated sensor source."""
import queue
import sys
import threading
import time
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from vlastudio.deploy import runtime, shm_utils
from vlastudio.webui import episodes


def test_web_inference_enables_dataset_recording(monkeypatch, tmp_path):
    from vlastudio.entrypoints.eval_real import parse_param
    monkeypatch.setenv('VLASTUDIO_WEB_GATE', 'infer')
    monkeypatch.setattr(sys, 'argv', ['infer', '-o', str(tmp_path)])
    assert parse_param().save_as_dataset == 'hdf5'
    monkeypatch.delenv('VLASTUDIO_WEB_GATE')
    assert parse_param().save_as_dataset is None


def test_full_recording_video_alignment_pause_discard_and_playback(tmp_path, monkeypatch):
    root=tmp_path/'data';folder=root/'recording';video=folder/'video';video.mkdir(parents=True)
    # Existing video-only episode must survive the switch to complete datasets.
    old=video/'real_eval_episode_0004.mp4';old.write_bytes(b'existing video must survive')
    monkeypatch.setattr(shm_utils,'SharedMemoryChannel',lambda *a,**k:SimpleNamespace(shm=True,destroy=lambda:None))
    class Source:
        def __init__(self,**kwargs):self.index=0
        def get_synced_frame_blocking(self,**kwargs):
            self.index+=1
            return {'sensor':{'qpos':np.arange(14,dtype=np.float32)+self.index/100,
                              'qvel':np.ones(14,dtype=np.float32),
                              'state':np.array([self.index,1.],dtype=np.float32),
                              'image':np.full((32,32,3),self.index%255,dtype=np.uint8)}}
    monkeypatch.setattr(shm_utils,'SharedMemoryDataSynchronizer',Source)
    stop=threading.Event();ready=threading.Event();commands=queue.Queue();acks=queue.Queue()
    worker=threading.Thread(target=runtime.eval_real_session_recorder_main,kwargs=dict(
        device_shm_names=['sensor'],video_path=str(video/'real_eval.mp4'),dataset_output_dir=str(folder),
        dataset_format='hdf5',task='test',episode_id=-1,sensing_rate=30,stop_event=stop,
        cmd_queue=commands,ack_queue=acks,recorder_ready_event=ready,initial_recording_active=False))
    def command(value):
        commands.put(value);return acks.get(timeout=15)
    worker.start()
    try:
        assert ready.wait(15)
        assert not list(folder.glob('*.hdf5'))
        assert command('resume_recording')[2]==5
        time.sleep(.3);command('pause_recording')
        time.sleep(.15);ack=command('save_episode');assert ack[0]=='saved' and ack[1]>0 and ack[2]==5
        path=folder/'episode_0005.hdf5';movie=video/'real_eval_episode_0005.mp4'
        assert movie.exists() and old.read_bytes()==b'existing video must survive'
        with h5py.File(path) as f:
            count=len(f['timestamps']);assert count==ack[1]
            assert f['observation/qpos'].shape==(count,14)
            assert f['observation/qvel'].shape==(count,14)
            assert f['observation/state'].shape==(count,2)
            assert f['observation/images/sensor'].shape==(count,32,32,3)
            assert f['observation/images/sensor'].compression=='lzf'
            assert np.all(np.diff(f['timestamps'][:])>0)
            assert f.attrs['recording_source']=='inference'
        import cv2
        cap=cv2.VideoCapture(str(movie));assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==count;cap.release()
        monkeypatch.setattr(episodes,'root_directory',lambda:root)
        version=episodes.revision(episodes.files('recording',5))
        frame=episodes.read('recording',5,version,frame=0)
        assert set(frame['values'])>={'qpos','qvel','state'} and frame['images']
        assert command('discard_episode')[2]==6
        command('resume_recording');time.sleep(.15);command('pause_recording');command('exit_discard')
        assert not (folder/'episode_0006.hdf5').exists()
        assert not (video/'real_eval_episode_0006.mp4').exists()
    finally:
        stop.set();worker.join(timeout=20)
    assert not worker.is_alive()

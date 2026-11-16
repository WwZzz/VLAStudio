import hashlib
from types import SimpleNamespace
import pytest

pytest.importorskip('fcntl')
import h5py
from vlastudio.webui import episodes, data_management as dm
from vlastudio.webui.__main__ import Handler


@pytest.fixture
def recording(tmp_path, monkeypatch):
    monkeypatch.setattr(episodes, 'root_directory', lambda:tmp_path)
    folder=tmp_path/'sample';folder.mkdir()
    path=folder/'episode_0000.hdf5'
    with h5py.File(path,'w') as f:f['timestamps']=[0.,.1,.2]
    return path,dict(dataset='sample',index=0,version=episodes.revision([path]))


def test_quick_label_preserves_recording_segments_and_history(recording):
    path,item=recording;original=hashlib.sha256(path.read_bytes()).hexdigest()
    assert dm.outcomes('sample')[0]['outcome']=='unknown'
    doc=dm.annotations(**item)['document'];records=doc['records']
    records.append(dict(id='part',scope='segment',start=0,end=3,language='keep me',outcome='failure',reviewed=True))
    saved=dm.save(dict(item,revision=None,records=records))
    result=dm.save_outcome(dict(item,outcome='success',revision=saved['document']['revision']))
    assert result['document']['records'][1]==records[1]
    assert len(result['history'])==2
    assert dm.outcomes('sample')[0]['outcome']=='success'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==original
    with pytest.raises(ValueError,match='another window'):
        dm.save_outcome(dict(item,outcome='failure',revision=None))
    with pytest.raises(ValueError):dm.save_outcome(dict(item,outcome='invalid'))
    with h5py.File(path,'a') as f:f.attrs['retake']=True
    assert dm.outcomes('sample')[0]['outcome']=='unknown'
    with pytest.raises(ValueError):dm.save_outcome(dict(item,outcome='failure'))


def test_outcome_does_not_decode_media_and_binds_frames_when_opened(recording, monkeypatch):
    _, item = recording
    read = episodes.read
    def unavailable(*args, **kwargs):
        raise ValueError('Video is unavailable')
    monkeypatch.setattr(episodes, 'read', unavailable)
    first = dm.save_outcome(dict(item, outcome='success', revision=None))
    assert dm.outcomes('sample')[0]['outcome'] == 'success'
    second = dm.save_outcome(dict(item, outcome='failure', revision=first['document']['revision']))
    assert dm.outcomes('sample')[0]['outcome'] == 'failure'
    monkeypatch.setattr(episodes, 'read', read)
    doc = dm.annotations(**item)['document']
    assert doc['records'][0]['end'] == 3 and doc['records'][0]['outcome'] == 'failure'
    assert doc['revision'] == second['document']['revision']
    doc['records'][0]['outcome'] = 'unknown'
    dm.save(dict(item, revision=doc['revision'], records=doc['records']))
    assert dm.outcomes('sample')[0]['outcome'] == 'unknown'


@pytest.mark.parametrize('mode',['collect','infer'])
@pytest.mark.parametrize('outcome',['success','failure','unknown'])
def test_save_labels_the_finished_episode_after_recorder_confirmation(tmp_path,monkeypatch,mode,outcome):
    monkeypatch.setattr(episodes,'root_directory',lambda:tmp_path)
    folder=tmp_path/'sample';folder.mkdir();path=folder/'episode_0007.hdf5'
    state=dict(id='session',actions=['save'],options={'output':'sample'},episode=7,phase='save' if mode=='collect' else 'infer_save')
    def action(selected,sid,action):
        assert (selected,sid,action)==(mode,'session','save')
        with h5py.File(path,'w') as f:f['timestamps']=[0.,.1]
        state.update(phase='ready',episode=8,actions=[])
    handler=object.__new__(Handler);handler.server=SimpleNamespace(api_upstream_port=None,session=SimpleNamespace(state=lambda _:state,action=action))
    result=handler.save_episode_outcome(dict(mode=mode,id='session',outcome=outcome))
    assert result['index']==7 and dm.outcomes('sample')[0]['outcome']==outcome
    with pytest.raises(ValueError,match='no longer'):handler.save_episode_outcome(dict(mode=mode,id='session',outcome=outcome))

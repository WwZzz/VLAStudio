"""Session-private recorder control, processed on the idle inference loop."""
import json
import os
from pathlib import Path


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value), encoding='utf-8')
    temporary.replace(path)


class RecordingControl:
    def __init__(self, directory, controller, start_recorder):
        self.directory = Path(directory)
        self.controller = controller
        self.start_recorder = start_recorder
        self.handled = None
        write_json(self.directory/'recording-capability.json', {'version':1, 'pid':os.getpid()})

    def poll(self):
        path = self.directory/'recording-request.json'
        if not path.exists():
            return
        request = json.loads(path.read_text(encoding='utf-8'))
        if request['id'] == self.handled:
            return
        self.handled = request['id']
        control = self.controller
        response = dict(id=request['id'], ok=False)
        try:
            if control.state != 'infer_ready':
                raise ValueError('Finish this inference trial before changing recording')
            if type(request['enabled']) is not bool:
                raise ValueError('Invalid recording setting')
            control.publish_phase('busy')
            if request['enabled']:
                if not control.runtime.has_recorder:
                    self.start_recorder(request['output'])
                    ack = control.runtime.send_recorder_cmd('pause_recording')
                    if not ack or ack[0] != 'paused':
                        control.runtime.stop_session_recorder()
                        raise RuntimeError('Recorder did not become ready')
            elif control.runtime.has_recorder:
                control.runtime.stop_session_recorder()
            control.needs_episode = False
            response.update(ok=True, enabled=control.runtime.has_recorder)
        except (ValueError, RuntimeError, OSError) as exc:
            response['error'] = str(exc)
        finally:
            control.publish_phase(control.state)
            write_json(self.directory/'recording-response.json', response)

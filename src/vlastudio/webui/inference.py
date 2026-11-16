"""Web inference trials keep the loaded policy separate from episode decisions."""
from .bridge import phase


def is_simulation_robot(configs):
    """Use robot implementation types, never a user-editable profile name."""
    types = [row.get('type', '') for row in configs if '.deploy.robot.' in row.get('type', '')]
    return bool(types) and all(
        typ in ('vlastudio.deploy.robot.bench_robot.BenchmarkRobot',
                'vlastudio.deploy.robot.bessica_sim.BessicaSimRobot')
        or typ.startswith('vlastudio.deploy.robot.sim_pybullet.') for typ in types)


class InferenceEpisodes:
    def __init__(self, runtime, manager, on_reset, publish_phase=phase, *, reset_after_trial=False):
        self.runtime, self.manager, self.on_reset = runtime, manager, on_reset
        self.publish_phase = publish_phase
        self.state = 'infer_ready'
        self.needs_episode = False
        self.reset_after_trial = reset_after_trial
        self.publish_phase(self.state)

    def _record(self, command, expected):
        if not self.runtime.has_recorder:
            return
        ack = self.runtime.send_recorder_cmd(command)
        if not ack or ack[0] != expected:
            raise RuntimeError('Recorder did not confirm ' + command)
        if command == 'resume_recording' and (len(ack) < 3 or ack[2] is None):
            raise RuntimeError('Recorder has no open inference episode')

    def handle(self, command):
        previous = self.state
        committed = False
        try:
            if command == '' and self.state in ('infer_ready', 'paused'):
                if self.state == 'infer_ready':
                    self.manager.reset()
                    if self.needs_episode:
                        self._record('discard_episode', 'discarded')
                        self.needs_episode = False
                self._record('resume_recording', 'resumed')
                self.state = 'running'
            elif command == '' and self.state == 'running':
                self._record('pause_recording', 'paused')
                self.state = 'paused'
            elif command == 'finish' and self.state in ('running', 'paused'):
                self._record('pause_recording', 'paused')
                self.state = 'infer_save' if self.runtime.has_recorder else 'infer_ready'
                if self.reset_after_trial and not self.runtime.has_recorder:
                    committed = True
                    self.manager.reset()
                    self.on_reset()
            elif command in ('s', 'd') and self.state == 'infer_save':
                self._record('save_episode' if command == 's' else 'exit_discard',
                             'saved' if command == 's' else 'exit_ok')
                self.needs_episode = True
                self.state = 'infer_ready'
                committed = True
                if self.reset_after_trial:
                    self.manager.reset()
                    self.on_reset()
            elif command == 'r' and self.state in ('infer_ready', 'paused'):
                if self.reset_after_trial:
                    self._record('exit_discard', 'exit_ok')
                    self.needs_episode = True
                    self.state = 'infer_ready'
                    committed = True
                else:
                    self._record('discard_episode', 'discarded')
                    self.needs_episode = False
                self.manager.reset()
                self.on_reset()
            else:
                raise ValueError('Inference state changed; refresh and retry')
        except Exception as error:
            # Keep an unsaved trial and the policy available when a recorder fails.
            if committed:
                self.state = 'infer_ready'  # A save/discard must never be repeated after a reset error.
            elif previous == 'running' and command in ('', 'finish'):
                self.state = 'infer_save' if command == 'finish' and self.runtime.has_recorder else 'paused'
            else:
                self.state = previous
            print('[Inference] ' + str(error), flush=True)
        finally:
            self.publish_phase(self.state)

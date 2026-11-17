"""Official sensor preprocessing connected to the VLAStudio policy worker."""
import numpy as np
import carla

from team_code.tcp_b2d_agent import TCPAgent
from TCP.config import GlobalConfig
from vlastudio.benchmark.carla import bridge
from vlastudio.data_utils.datasets.bench2drive import compose_images, driving_observation, local_points


def get_entry_point():
    return 'VLAStudioDrivingAgent'


class VLAStudioDrivingAgent(TCPAgent):
    def setup(self, configuration):
        from leaderboard.autoagents.autonomous_agent import Track
        self.track = Track.SENSORS
        self.config = GlobalConfig()
        self.initialized, self.step = False, -1
        self.save_name = configuration.split('+')[-1]

    def sensors(self):
        # The upstream BEV camera is a debugging visualization outside the sensor
        # track. Policy observations use only the allowed front cameras and ego sensors.
        return [sensor for sensor in super().sensors() if sensor['id'] != 'bev']

    def tick(self, input_data):
        import cv2
        self.step += 1
        images = [cv2.cvtColor(input_data[name][1][:, :, :3], cv2.COLOR_BGR2RGB)
                  for name in ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT']]
        rgb = compose_images(images, jpeg=True).transpose(1, 2, 0)
        compass = float(input_data['IMU'][1][-1])
        if not np.isfinite(compass):
            compass = 0.
        position = self.gps_to_location(input_data['GPS'][1][:2])
        waypoint, command = self._route_planner.run_step(position)
        target = local_points(np.asarray(waypoint)[None], position, compass)[0]
        return {'rgb': rgb, 'speed': float(input_data['SPEED'][1]['speed']),
                'target_point': target, 'next_command': command.value}

    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()
        values = self.tick(input_data)
        if self.step < self.config.seq_len:
            return carla.VehicleControl()
        observation = {**driving_observation(values['speed'], values['target_point'], values['next_command']),
                       'image': values['rgb'].transpose(2, 0, 1)[None].copy(),
                       'timestep': self.step-self.config.seq_len}
        bridge.AGENT_PIPE.send(('observation', observation))
        command, value = bridge.AGENT_PIPE.recv()
        if command == 'close':
            raise KeyboardInterrupt()
        if command != 'action':
            raise ValueError('Expected one vehicle control action')
        steer, throttle, brake, reverse = np.asarray(value, dtype=float)
        return carla.VehicleControl(steer=float(steer), throttle=float(throttle), brake=float(brake), reverse=bool(reverse))

    def destroy(self):
        # ScenarioManager owns sensors; this process never constructs a policy.
        pass

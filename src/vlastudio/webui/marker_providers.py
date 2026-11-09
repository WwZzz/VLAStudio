"""Simulator-specific, read-only marker adapters owned by the UI."""
from .ee_overlay import axes_markers


class RLBenchMarkerProvider:
    def __init__(self, environment, image_names):
        from pyrep.objects.vision_sensor import VisionSensor
        native_names = dict(front='cam_front', wrist='cam_wrist', overhead='cam_overhead',
                            left_shoulder='cam_over_shoulder_left', right_shoulder='cam_over_shoulder_right')
        self.tip = environment.rlbench_env._robot.arm.get_tip()
        self.cameras = []
        for image, name in zip(image_names, environment.camera_names):
            try:
                sensor = VisionSensor(native_names[name])
                calibration = dict(intrinsics=sensor.get_intrinsic_matrix().tolist(),
                                   image_size=sensor.get_resolution(), forward_axis=1)
                self.cameras.append((image, sensor, calibration))
            except (KeyError, RuntimeError):
                continue

    def get_marker_from_observation(self, observation):
        # Native FK is already computed by the simulator. No state_ee layout dependency.
        effectors = [dict(label='EE', position=self.tip.get_position())]
        images = {}
        for image, sensor, calibration in self.cameras:
            transform = sensor.get_matrix()
            camera = dict(calibration, position=transform[:3, 3], rotation=transform[:3, :3])
            images[image] = axes_markers(camera, effectors, calibration['image_size'])
        return dict(version=1, images=images)


class RobosuiteMarkerProvider:
    def __init__(self, environment, image_names):
        self.base = environment.env.base_env
        names = [key.removesuffix('_image') for key in environment.img_keys]
        self._configure(environment, image_names, names)

    def _configure(self, environment, image_names, names, flip_x=False):
        from robosuite import macros
        self.sim = self.base.sim
        self.cameras = []
        for source_index, (image, name) in enumerate(zip(image_names, names)):
            try:
                index = self.sim.model.camera_name2id(name)
                if index >= 0:
                    self.cameras.append((image, source_index, index, dict(fovy=float(self.sim.model.cam_fovy[index]),
                        flip_x=flip_x, flip_y=macros.IMAGE_CONVENTION == 'opencv')))
            except (AttributeError,KeyError,ValueError,IndexError):
                continue
        self.sites = []
        for robot_index, robot in enumerate(self.base.robots):
            sites = robot.eef_site_id
            sites = sites if isinstance(sites, dict) else {'arm': sites}
            for arm, site in sites.items():
                self.sites.append((f'R{robot_index}/{arm}', site))

    def get_marker_from_observation(self, observation):
        effectors = [dict(label='EE' if len(self.sites) == 1 else label,
                          position=self.sim.data.site_xpos[site]) for label, site in self.sites]
        rgb = observation.get('image')
        images = {}
        for image, image_index, index, calibration in self.cameras:
            if rgb is None or image_index >= len(rgb):
                continue
            height, width = rgb[image_index].shape[-2:]
            camera = dict(calibration, position=self.sim.data.cam_xpos[index],
                          rotation=self.sim.data.cam_xmat[index].reshape(3, 3))
            images[image] = axes_markers(camera, effectors, [width, height])
        return dict(version=1, images=images)


class LiberoMarkerProvider(RobosuiteMarkerProvider):
    def __init__(self, environment, image_names):
        self.base = environment.env.env
        names = ['agentview'] + (['robot0_eye_in_hand'] if environment.use_wrist else [])
        self._configure(environment, image_names, names, flip_x=True)


class AlohaMarkerProvider:
    def __init__(self, environment, image_names):
        self.environment = environment
        self.physics = environment.env.physics
        self.cameras = [(image, ['top', 'left_wrist', 'right_wrist'][int(index)])
                        for image, index in zip(image_names, environment.camera_ids)]

    def get_marker_from_observation(self, observation):
        physics = self.physics
        effectors = [dict(label=side[0].upper(), position=physics.named.data.xpos[f'vx300s_{side}/gripper_link'])
                     for side in ('left', 'right')]
        images = {}
        for i, (image, camera_name) in enumerate(self.cameras):
            camera = dict(position=physics.named.data.cam_xpos[camera_name],
                          rotation=physics.named.data.cam_xmat[camera_name].reshape(3, 3),
                          fovy=float(physics.named.model.cam_fovy[camera_name]))
            height, width = observation['image'][i].shape[-2:]
            images[image] = axes_markers(camera, effectors, [width, height])
        return dict(version=1, images=images)

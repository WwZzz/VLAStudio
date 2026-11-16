"""Bind optional UI marker providers without changing simulator interfaces."""
from loguru import logger
from typing import Protocol

from vlastudio.utils.extensions import resolve


class MarkerProvider(Protocol):
    def get_marker_from_observation(self, observation: dict) -> dict | None:
        """Return a v1 packet of typed 2D markers, or no display information."""
        ...


class MarkerBinding:
    def __init__(self, device, specification):
        self.device = device
        self.specification = specification
        self.original_cache = device._cache
        self.provider = None
        self.episode = None
        self.error = None

    def cache(self, observation, *args, **kwargs):
        self.original_cache(observation, *args, **kwargs)
        try:
            episode = (id(self.device.env), self.device.episode)
            if episode != self.episode:
                # Rebind after Reset too: native camera/site handles can change.
                self.episode = episode
                self.provider = None
                spec = self.specification
                spec = {'type': spec} if isinstance(spec, str) else spec
                factory = resolve(spec['type'])
                self.provider = factory(
                    self.device.env,
                    image_names=list(self.device.observation_config.get('cameras', [])),
                    **spec.get('args', {}),
                )
            if self.provider is not None:
                packet = self.provider.get_marker_from_observation(self.device.raw)
                if packet and packet.get('version') == 1 and isinstance(packet.get('images'), dict):
                    self.device.cached['__preview__'] = packet
        except Exception as error:
            # Display extensions must never stop simulation, policy control or saving.
            message = str(error)
            if message != self.error:
                logger.warning('UI marker provider unavailable: {}', error)
                self.error = message

    def detach(self):
        if getattr(self.device, '_ui_marker_binding', None) is not self:
            return
        self.device._cache = self.original_cache
        if self.device.cached:
            self.device.cached.pop('__preview__', None)
        del self.device._ui_marker_binding


def bind_marker_provider(device, specification):
    """Attach a UI observer to a BenchmarkRobot; the environment stays untouched.

    A factory receives (environment, image_names=..., **args). Its bound provider
    implements get_marker_from_observation(observation), returning None or a v1
    packet of normalized 2D markers per image. Only WebUI workers call this hook.
    """
    if not hasattr(device, '_cache'):
        return None
    if hasattr(device, '_ui_marker_binding'):
        device._ui_marker_binding.detach()
    if not specification:
        return None
    binding = MarkerBinding(device, specification)
    device._ui_marker_binding = binding
    device._cache = binding.cache
    return binding

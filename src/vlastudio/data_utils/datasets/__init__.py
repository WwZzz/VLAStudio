"""Dataset classes, imported only when the selected dataset needs them."""
from importlib import import_module


_EXPORTS = {
    "EpisodicDataset": (".base", "EpisodicDataset"),
    "AlohaSimDataset": (".aloha_sim", "AlohaSimDataset"),
    "AlohaSIIDataset": (".aloha_sii", "AlohaSIIDataset"),
    "AlohaSIIv2Dataset": (".aloha_sii_v2", "AlohaSIIv2Dataset"),
    "RobomimicDataset": (".robomimic_dataset", "RobomimicDataset"),
    "KochDataset": (".koch_dataset", "KochDataset"),
    "D4RLDataset": (".d4rl", "D4RLDataset"),
    "WrappedLerobotDataset": (".lerobot_wrapper", "WrappedLerobotDataset"),
    "RLBenchDataset": (".rlbench_dataset", "RLBenchDataset"),
    "WrappedLerobotV20Dataset": (".lerobotv20_wrapper", "WrappedLerobotV20Dataset"),
    "WrappedLerobotV21Dataset": (".lerobotv21_wrapper", "WrappedLerobotV21Dataset"),
    "WrappedLerobotV30Dataset": (".lerobotv30_wrapper", "WrappedLerobotV30Dataset"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))

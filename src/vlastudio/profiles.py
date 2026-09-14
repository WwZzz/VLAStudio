"""Environment declarations are data: inspecting one never imports the policy."""
import copy
from pathlib import Path
import yaml
from .configuration import read_config

TORCH = ["torch==2.4.0", "torchvision==0.19.0", "numpy==1.26.4", "transformers==4.45.2", "accelerate==1.0.1", "loguru==0.7.3", "h5py==3.12.1", "pillow==11.3.0", "opencv-python==4.9.0.80", "einops==0.8.1", "scipy==1.14.1", "timm==1.0.15", "torchdata==0.9.0", "matplotlib==3.9.4", "imageio==2.37.0", "imageio-ffmpeg==0.6.0", "websockets==13.1", "psutil==6.1.1", "tensorboardX==2.6.4"]
BUILTINS = {name: {"python": "3.10", "requirements": TORCH} for name in ("policy.act", "policy.mlp")}
REMOTE_EVAL = ["torch==2.4.0", "numpy==1.26.4", "pillow==11.3.0",
               "opencv-python==4.9.0.80", "imageio==2.37.0",
               "imageio-ffmpeg==0.6.0", "tqdm==4.67.1", "requests==2.32.5",
               "psutil==6.1.1"]
BUILTINS["policy.remote"] = {"python": "3.10", "requirements": REMOTE_EVAL}


def environment_for(config, path, manifest=None):
    declaration = manifest or config.get("runtime")
    if not declaration:
        module = config.get("type") or config.get("module_path")
        if module and module.startswith('vlastudio.'):
            module = module[len('vlastudio.'):]
        if module not in BUILTINS:
            raise ValueError(f"No managed runtime for {module!r}. Declare runtime in the config, use --runtime-manifest, or --runtime current.")
        declaration = copy.deepcopy(BUILTINS[module])
        import sys, platform
        family = "torch310" if module in ("policy.act", "policy.mlp") else module.split(".")[-1]
        lock = Path(__file__).parent / "locks" / f"{family}-{sys.platform}-{platform.machine().lower()}.txt"
        if lock.is_file():
            declaration["lock"] = lock.read_text(encoding="utf-8")
    if isinstance(declaration, str):
        manifest_path = Path(declaration).expanduser()
        if not manifest_path.is_absolute():
            manifest_path = path.parent / manifest_path
        path = manifest_path.resolve()
        declaration = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(declaration, dict):
        raise ValueError("runtime must be a mapping or a YAML manifest path")
    unknown = set(declaration) - {"python", "requirements", "lockfile", "lock", "entrypoint", "overrides", "overlays", "platforms", "min_glibc"}
    if unknown:
        raise ValueError(f"Unknown runtime fields: {sorted(unknown)}")
    profile = copy.deepcopy(declaration)
    import sys
    if profile.get("platforms") and sys.platform not in profile["platforms"]:
        raise ValueError(f"This runtime supports {profile['platforms']}; current platform is {sys.platform}")
    if profile.get("min_glibc"):
        import platform
        libc, version = platform.libc_ver()
        if libc != "glibc" or tuple(map(int, version.split("."))) < tuple(map(int, profile["min_glibc"].split("."))):
            raise ValueError(f"Runtime requires glibc >= {profile['min_glibc']}; found {libc} {version}")
    profile.setdefault("python", "3.10")
    requirements = profile.setdefault("requirements", [])
    if not isinstance(requirements, list) or not all(isinstance(r, str) for r in requirements):
        raise ValueError("runtime.requirements must be a list of requirement strings")
    if profile.get("lockfile"):
        if requirements:
            raise ValueError("Use either a complete runtime.lockfile or runtime.requirements")
        lock = Path(profile.pop("lockfile")).expanduser()
        if not lock.is_absolute():
            lock = path.parent / lock
        profile["lock"] = lock.read_text(encoding="utf-8")
    if ".py:" in profile.get("entrypoint", ""):
        filename, attribute = profile["entrypoint"].rsplit(":", 1)
        entry = Path(filename).expanduser()
        if not entry.is_absolute():
            entry = path.parent / entry
        profile["entrypoint"] = str(entry.resolve()) + ":" + attribute
    profile["python"] = str(profile["python"])
    return profile


def merge_component_runtimes(profile, config, path):
    """Combine dependency declarations for components sharing one worker process."""
    result = copy.deepcopy(profile)
    def visit(value):
        if isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, dict):
            if value.get("runtime"):
                extra = environment_for(value, path)
                if extra["python"] != result["python"]:
                    raise ValueError("Components in one process require different Python versions; use a compatible manifest or separate services")
                if extra.get("entrypoint"):
                    raise ValueError("Only the top-level runtime may select the task entrypoint")
                if extra.get("lock"):
                    raise ValueError("Use one complete lockfile instead of multiple component lockfiles")
                if result.get("lock") and extra["requirements"]:
                    # Keep the validated base pins while resolving additional plugin dependencies.
                    result["requirements"] = [line for line in result.pop("lock").splitlines() if line.strip() and not line.lstrip().startswith("#")]
                for requirement in extra["requirements"]:
                    if requirement not in result["requirements"]:
                        result["requirements"].append(requirement)
            for key, child in value.items():
                if key != "runtime":
                    visit(child)
    visit(config)
    return result


_OPENPI_REV = "0d3eb2db836e443a6ca9debb242d0aabe4628f1f"
_OPENPI_URL = "https://github.com/Physical-Intelligence/openpi/archive/" + _OPENPI_REV + ".tar.gz"
_OPENPI_DEPS = [r for r in TORCH if not r.startswith(("torch==", "torchvision==", "transformers==", "opencv-python=="))]
BUILTINS["policy.openpi"] = {
    "python": "3.11", "platforms": ["linux"], "min_glibc": "2.31",
    "requirements": _OPENPI_DEPS + [
        "torch==2.7.1", "torchvision==0.22.1", "transformers==4.53.2", "opencv-python==4.10.0.84",
        "openpi @ " + _OPENPI_URL,
        "openpi-client @ " + _OPENPI_URL + "#subdirectory=packages/openpi-client",
        "lerobot @ git+https://github.com/huggingface/lerobot.git@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5",
        "peft==0.17.1", "rich==14.0.0", "draccus==0.10.0", "tyro==1.0.3", "pytest==8.3.5", "chex==0.1.89",
    ],
    "overrides": ["ml-dtypes==0.4.1", "tensorstore==0.1.74"],
    "overlays": [{"source_module": "openpi", "source": "models_pytorch/transformers_replace", "target_module": "transformers"}],
}
BUILTINS["policy.openvla"] = {
    "python": "3.10", "platforms": ["linux"],
    "requirements": [r for r in TORCH if not r.startswith(("timm==", "transformers=="))] + ["timm==0.9.16", "transformers==4.40.1", "tokenizers==0.19.1", "peft==0.13.2", "sentencepiece==0.2.0", "draccus==0.10.0", "rich==13.9.4"],
}

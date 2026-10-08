"""Pin the upstream sources, download released weights, and write a VLAStudio bundle."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import yaml
from huggingface_hub import hf_hub_download, snapshot_download

FASTWAM_REVISION = "7faa71108368fbb3b6885649f112af607427a2d4"
LIBERO_REVISION = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
WEIGHTS_REVISION = "8eaceeb24c3cc92ff2a9c9a9d266a4941b836705"


def checkout(url, root, revision):
    if not root.exists():
        subprocess.run(["git", "clone", url, str(root)], check=True)
        subprocess.run(["git", "-C", str(root), "checkout", revision], check=True)
    actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if actual != revision:
        raise RuntimeError(f"Expected {revision} in {root}, found {actual}; use a separate checkout")
    return root


def main():
    import sys
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", help="Existing pinned vanilla LIBERO checkout")
    parser.add_argument("--assets", help="Existing LIBERO assets")
    args = parser.parse_args()
    cache = Path(os.getenv("VLASTUDIO_CACHE", "~/.cache/vlastudio")).expanduser().resolve()
    sources = cache / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    upstream = checkout("https://github.com/yuantianyuan01/FastWAM.git", sources/"FastWAM", FASTWAM_REVISION)
    libero = checkout("https://github.com/Lifelong-Robot-Learning/LIBERO.git", Path(args.root).expanduser().resolve() if args.root else sources/"LIBERO-fastwam", LIBERO_REVISION)
    # The inference environment installs only the dependencies in runtime.yaml.
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "-e", str(upstream)], check=True)
    package = libero / "libero/libero"
    assets = Path(args.assets).expanduser().resolve() if args.assets else package/"assets"
    if not assets.is_dir() or not any(assets.rglob("*.xml")):
        raise FileNotFoundError(f"LIBERO assets missing: {assets}")
    (cache/"data/libero").mkdir(parents=True, exist_ok=True)
    work = cache / "fastwam"
    config_dir = work / "libero_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir/"config.yaml").write_text(yaml.safe_dump(dict(benchmark_root=str(package),bddl_files=str(package/"bddl_files"),init_states=str(package/"init_files"),assets=str(assets),datasets=str(cache/"data/libero"))))
    bundle = work / "bundle"
    bundle.mkdir(exist_ok=True)
    released = {}
    for key, filename in (("weights", "libero_uncond_2cam224.pt"), ("dataset_stats", "libero_uncond_2cam224_dataset_stats.json")):
        released[key] = hf_hub_download("yuanty/fastwam", filename, revision=WEIGHTS_REVISION, cache_dir=str(cache/"models/huggingface/hub"))
    for repo, revision, patterns in [
        ("Wan-AI/Wan2.2-TI2V-5B", "921dbaf3f1674a56f47e83fb80a34bac8a8f203e", ["Wan2.2_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth"]),
        ("Wan-AI/Wan2.1-T2V-1.3B", "37ec512624d61f7aa208f7ea8140a131f93afc9a", ["google/umt5-xxl/*"]),
    ]:
        snapshot_download(repo, revision=revision, allow_patterns=patterns,
                          local_dir=str(cache/"models/fastwam"/repo))
    spec = yaml.safe_load((Path(__file__).parent/"config/fastwam.yaml").read_text())["args"]
    spec.update(released, upstream_root=str(upstream), upstream_revision=FASTWAM_REVISION, weights_revision=WEIGHTS_REVISION)
    (bundle/"fastwam.json").write_text(json.dumps(spec, indent=2))
    (bundle/"policy_metadata.json").write_text(json.dumps({"policy_module":"vlastudio.policy.fastwam","policy_name":"fastwam"}))
    (bundle/"config.json").write_text(json.dumps({"chunk_size":32,"action_dim":7,"state_dim":8}))
    (bundle/"normalize.json").write_text(json.dumps({"datasets":[{"dataset_id":"fastwam","ctrl_space":"ee","ctrl_type":"delta"}],"state":{"fastwam":"identity"},"action":{"fastwam":"identity"}},indent=2))
    values = {"LIBERO_ROOT":str(libero),"LIBERO_CONFIG_PATH":str(config_dir),"MUJOCO_GL":"egl","DIFFSYNTH_MODEL_BASE_PATH":str(cache/"models/fastwam"),"DIFFSYNTH_DOWNLOAD_SOURCE":"huggingface"}
    for key in ("LD_LIBRARY_PATH", "__EGL_VENDOR_LIBRARY_FILENAMES"):
        if os.environ.get(key):
            values[key] = os.environ[key]
    (work/"env.sh").write_text("".join(f"export {k}={shlex.quote(v)}\n" for k,v in values.items()))
    print(f"Prepared bundle: {bundle}")


if __name__ == "__main__":
    main()

"""Prepare an isolated, pinned LIBERO-Plus source/configuration under the cache."""
import argparse
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess
import zipfile

PLUS_REVISION = "4976dc30028e805ff8094b55501d532c48fec182"
ASSETS_REVISION = "dd2bd61b7d9a6fef1abc52d606e983b41886a149"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["plus", "standard"], default="plus")
    parser.add_argument("--root", help="Existing simulator checkout")
    parser.add_argument("--assets", help="Existing extracted assets directory")
    parser.add_argument("--download-assets", action="store_true")
    args = parser.parse_args()
    cache = Path(os.environ.get("VLASTUDIO_CACHE", "~/.cache/vlastudio")).expanduser().resolve()
    repo = "sylvestf/LIBERO-plus" if args.backend == "plus" else "Lifelong-Robot-Learning/LIBERO"
    revision = PLUS_REVISION if args.backend == "plus" else "8f1084e3132a39270c3a13ebe37270a43ece2a01"
    root = Path(args.root).expanduser().resolve() if args.root else cache / "sources" / repo.split("/")[-1]
    if not root.exists():
        subprocess.run(["git", "clone", f"https://github.com/{repo}.git", str(root)], check=True)
        subprocess.run(["git", "-C", str(root), "checkout", revision], check=True)
    package = root / "libero/libero"
    if not (package / "__init__.py").is_file():
        raise ValueError(f"Not a simulator checkout: {root}")
    if args.backend == "plus" and not (package / "benchmark/task_classification.json").is_file():
        raise ValueError(f"Not a LIBERO-Plus checkout: {root}")
    assets = Path(args.assets).expanduser().resolve() if args.assets else package / "assets"
    if args.download_assets and args.backend == "plus":
        from huggingface_hub import hf_hub_download
        archive = hf_hub_download("Sylvest/LIBERO-plus", "assets.zip", repo_type="dataset", revision=ASSETS_REVISION,
                                  cache_dir=str(cache / "models/huggingface/hub"))
        # Upstream includes its build machine's directory prefix before assets/.
        # Extract only the asset subtree, never recreate that machine's paths.
        with zipfile.ZipFile(archive) as zipped:
            members = []
            for item in zipped.infolist():
                parts = PurePosixPath(item.filename).parts
                if ".." in parts or "assets" not in parts:
                    raise ValueError(f"Unexpected archive member: {item.filename}")
                relative = Path(*parts[parts.index("assets") + 1:])
                target = (assets / relative).resolve()
                if not target.is_relative_to(assets.resolve()):
                    raise ValueError(f"Unsafe archive member: {item.filename}")
                members.append((item, target))
            for index, (item, target) in enumerate(members):
                if item.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zipped.open(item) as src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                if index % 25000 == 0:
                    print(f"Extracted {index}/{len(members)} asset entries", flush=True)
    if not assets.is_dir():
        raise FileNotFoundError("Provide --assets or use --download-assets.")
    import yaml
    config = cache / ("libero-plus" if args.backend == "plus" else "libero-standard") / "config"
    config.mkdir(parents=True, exist_ok=True)
    paths = dict(benchmark_root=str(package), bddl_files=str(package / "bddl_files"),
                 init_states=str(package / "init_files"), assets=str(assets),
                 datasets=str(cache / "data/libero"))
    (config / "config.yaml").write_text(yaml.safe_dump(paths), encoding="utf-8")
    env_file = config.parent / "env.sh"
    env_file.write_text(
        "export LIBERO_ROOT=" + shlex.quote(str(root)) + "\n"
        "export LIBERO_CONFIG_PATH=" + shlex.quote(str(config)) + "\n"
        "export MUJOCO_GL=egl\n", encoding="utf-8")
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    print(f"{args.backend} revision: {revision}")
    print(f"source {shlex.quote(str(env_file))}")


if __name__ == "__main__":
    main()

"""Explicit GPU integration checks using an installed VLAStudio distribution.

python run_smoke.py --cache-dir /scratch/cache --output /scratch/results mlp act openpi openvla
"""
import argparse
from pathlib import Path
import subprocess
import sys
import yaml

from vlastudio.profiles import environment_for


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("cases", nargs="+", choices=("mlp", "act", "openpi", "openvla", "train_mlp"))
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache_dir).resolve()
    for case in args.cases:
        module = "mlp" if case == "train_mlp" else case
        profile = environment_for({"type": "policy." + module}, Path.cwd() / "config.yaml")
        profile["entrypoint"] = str(Path(__file__).with_name("smoke.py").resolve()) + ":run"
        manifest = output / f"{case}-runtime.yaml"
        manifest.write_text(yaml.safe_dump(profile), encoding="utf-8")
        with (output / f"{case}.log").open("w", encoding="utf-8") as log:
            result = subprocess.run([sys.executable, "-m", "vlastudio", "train",
                "--runtime-manifest", str(manifest), "--cache-dir", str(cache),
                "--case", case, "--result", str(output / f"{case}.json"),
                *(["--offline"] if args.offline else [])],
                cwd=output, stdout=log, stderr=subprocess.STDOUT)
        print(f"{case}: exit {result.returncode}; log: {output / (case + '.log')}", flush=True)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

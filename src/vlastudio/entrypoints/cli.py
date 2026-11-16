"""CLI dispatch with the same policy/task/output argument semantics."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import subprocess
import yaml
from vlastudio.core.paths import cache_root, runtime_env
from vlastudio.deploy.interpreter.recipes import environment_for
from vlastudio.deploy.interpreter.runtime import execute


PUBLIC_COMMANDS = ["device", "train", "evaluate", "collect", "infer", "serve", "doctor"]
LEGACY_ALIASES = {"eval-sim": "evaluate", "eval-real": "infer"}


def parse(argv):
    argv = list(argv)
    if argv and argv[0] in LEGACY_ALIASES:
        argv[0] = LEGACY_ALIASES[argv[0]]
    parser = argparse.ArgumentParser(prog="vlastudio", allow_abbrev=False)
    parser.add_argument("command", choices=PUBLIC_COMMANDS)
    parser.add_argument("--runtime", choices=["current"], default="current")
    parser.add_argument("--runtime-manifest", help="Environment YAML, resolved from the current directory")
    parser.add_argument("--cache-dir")
    parser.add_argument("--model-cache-dir")
    parser.add_argument("--data-cache-dir", help="Dataset caches, separate from model and environment caches")
    parser.add_argument("--plugin-path", action="append", default=[])
    parser.add_argument("--config-path", action="append", default=[])
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    # Parse help separately so train flags can be described without ML imports.
    if any(x in argv for x in ("-h", "--help")):
        parser.epilog = "train forwards -p/--policy (config name or YAML path), -t/--task, -c/--training_config, -o/--output_dir and dotted overrides unchanged. evaluate forwards the simulation evaluation flags; infer forwards the real-world inference flags; serve forwards -m/--model_name_or_path (checkpoint path)."
    return parser.parse_known_args(argv)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'env':
        from vlastudio.deploy.interpreter.cli import main as env_main
        return env_main(argv[1:])
    options, args = parse(sys.argv[1:] if argv is None else argv)
    try:
        cache = cache_root(options.cache_dir)
        env = runtime_env(cache, options.model_cache_dir, options.data_cache_dir)
        for option, variable in ((options.plugin_path, "VLASTUDIO_PLUGIN_PATH"), (options.config_path, "VLASTUDIO_CONFIG_PATH")):
            if option:
                paths = [str(Path(x).expanduser().resolve()) for x in option]
                if env.get(variable):
                    paths.append(env[variable])
                env[variable] = os.pathsep.join(paths)
        if options.offline:
            env.update(UV_OFFLINE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        if options.command == "doctor":
            print(json.dumps({"python": sys.executable, "cache": str(cache), "environments": [p.parent.name for p in (cache / "envs").glob("*/.ready")]}, indent=2))
            return 0
        # Apply config lookup paths only while resolving metadata, not policy imports.
        previous = os.environ.get("VLASTUDIO_CONFIG_PATH")
        os.environ["VLASTUDIO_CONFIG_PATH"] = env.get("VLASTUDIO_CONFIG_PATH", "")
        try:
            profile = None
            if options.runtime_manifest:
                profile = environment_for({}, Path.cwd() / "config.yaml", str(Path(options.runtime_manifest).expanduser().resolve()))

        finally:
            if previous is None:
                os.environ.pop("VLASTUDIO_CONFIG_PATH", None)
            else:
                os.environ["VLASTUDIO_CONFIG_PATH"] = previous
        if options.dry_run:
            preview = dict(profile) if profile else None
            if preview and preview.get("lock"):
                preview["lock"] = {"sha256": hashlib.sha256(preview["lock"].encode()).hexdigest(), "lines": len(preview["lock"].splitlines())}
            print(json.dumps({"command": options.command, "args": args, "runtime": options.runtime, "profile": preview, "cache": str(cache), "cwd": os.getcwd()}, indent=2))
            return 0
        from vlastudio.core.paths import package_source_root
        if profile:
            env["VLASTUDIO_RUNTIME_JSON"] = json.dumps({k: v for k, v in profile.items() if k != "entrypoint"})
        return execute(options.command, args, Path(sys.executable), package_source_root().parent,
                       env, (profile or {}).get("entrypoint"), isolated=False)

    except KeyboardInterrupt:
        return 130
    except (ValueError, RuntimeError, OSError, KeyError, yaml.YAMLError, subprocess.CalledProcessError) as error:
        print(f"vlastudio: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

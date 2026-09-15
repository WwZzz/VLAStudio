"""CLI dispatch with the same train.py policy/task/output argument semantics."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import subprocess
import yaml
from .configuration import read_config
from .paths import cache_root, runtime_env
from .profiles import environment_for, merge_component_runtimes
from .runtime import prepare, snapshot, execute
from .worker import ENTRIES
from .deploy.comm import is_server_address


def parse(argv):
    parser = argparse.ArgumentParser(prog="vlastudio", allow_abbrev=False)
    parser.add_argument("command", choices=[*ENTRIES, "prepare", "doctor"])
    parser.add_argument("--runtime", choices=["managed", "current"], default="managed")
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
        parser.epilog = "Train forwards -p/--policy (config name or YAML path), -t/--task, -c/--training_config, -o/--output_dir and dotted overrides unchanged. Serve forwards -m/--model_name_or_path (checkpoint path)."
    return parser.parse_known_args(argv)


def selected_config(command, args):
    selector = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    if command in ("train", "prepare"):
        selector.add_argument("-p", "--policy", default="act")
        options, _ = selector.parse_known_args(args)
        config, path = read_config(options.policy, "policy")
        # Runtime selection must agree with a CLI policy type override.
        for i, token in enumerate(args):
            for key in ("type", "module_path"):
                flag = "--policy." + key
                if token == flag and i + 1 < len(args):
                    config[key] = args[i + 1]
                elif token.startswith(flag + "="):
                    config[key] = token.split("=", 1)[1]
        return config, path
    if command == "device":
        selector.add_argument("-c", "--config", required=True)
        options, _ = selector.parse_known_args(args)
        return read_config(options.config, "device")
    if command in ("serve", "eval-sim"):
        selector.add_argument("-m", "--model_name_or_path", default="ckpt/act_sim_transfer_cube_scripted_zscore_example")
        options, _ = selector.parse_known_args(args)
        if command == "eval-sim" and is_server_address(options.model_name_or_path):
            return {"type": "vlastudio.policy.remote"}, Path.cwd()
        checkpoint = Path(options.model_name_or_path).expanduser().resolve()
        metadata_root = checkpoint.parent if checkpoint.name.startswith("checkpoint-") else checkpoint
        metadata = metadata_root / "policy_metadata.json"
        data = json.loads(metadata.read_text(encoding="utf-8"))
        # Persisted runtime supports external checkpoint/policy extensions.
        return {"type": data["policy_module"], "runtime": data.get("runtime")}, metadata
    raise ValueError(f"{command} requires --runtime-manifest or --runtime current until an environment is declared")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'env':
        from .env_cli import main as env_main
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
            if options.runtime == "managed":
                if options.runtime_manifest:
                    profile = environment_for({}, Path.cwd() / "config.yaml", str(Path(options.runtime_manifest).expanduser().resolve()))
                else:
                    config, path = selected_config(options.command, args)
                    profile = environment_for(config, path)
                    if options.command == "train":
                        task_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
                        task_parser.add_argument("-t", "--task", default="sim_transfer_cube_scripted")
                        task_args, _ = task_parser.parse_known_args(args)
                        task_config, task_path = read_config(task_args.task, "task")
                        profile = merge_component_runtimes(profile, task_config, task_path)
                    elif options.command == "eval-sim":
                        env_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
                        env_parser.add_argument("-e", "--env", default="aloha_transfer")
                        env_args, _ = env_parser.parse_known_args(args)
                        env_config, env_path = read_config(env_args.env, "env")
                        profile = merge_component_runtimes(profile, env_config, env_path)
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
        python = prepare(profile, cache, env, options.offline) if profile else Path(sys.executable)
        if options.command == "prepare":
            print(f"Environment ready: {python}")
            return 0
        if profile:
            env["VLASTUDIO_RUNTIME_JSON"] = json.dumps({k: v for k, v in profile.items() if k != "entrypoint"})
        app = snapshot(cache)
        return execute(options.command, args, python, app, env, (profile or {}).get("entrypoint"), isolated=options.runtime == "managed")
    except KeyboardInterrupt:
        return 130
    except (ValueError, RuntimeError, OSError, KeyError, yaml.YAMLError, subprocess.CalledProcessError) as error:
        print(f"vlastudio: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

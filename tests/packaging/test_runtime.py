import json
from pathlib import Path
import os
import subprocess
import sys
import pytest

from vlastudio.cli import main
from vlastudio.configuration import resolve_config
from vlastudio.extensions import create, resolve, register
from vlastudio.paths import cache_root, runtime_env
from vlastudio.profiles import environment_for
from vlastudio.runtime import prepare


def test_import_is_light():
    result = subprocess.run([sys.executable, "-c", "import vlastudio, sys; assert not any(x in sys.modules for x in ('torch', 'transformers', 'tensorflow', 'configs'))"], capture_output=True)
    assert result.returncode == 0, result.stderr


def test_train_preserves_paths_and_overrides(tmp_path, capsys):
    config = tmp_path / "custom policy.yaml"
    config.write_text("type: policy.act\n")
    args = ["train", "--dry-run", "--policy", str(config), "-o", "my checkpoints", "--policy.args.chunk_size", "12"]
    assert main(args) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["args"] == args[2:]
    assert plan["profile"]["python"] == "3.10"


def test_config_outside_checkout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert resolve_config("act", "policy").is_file()
    assert resolve_config("configs/policy/act.yaml", "policy").is_file()


def test_custom_config_search(tmp_path, monkeypatch):
    (tmp_path / "policy").mkdir()
    custom = tmp_path / "policy/act.yaml"
    custom.write_text("type: external.policy")
    monkeypatch.setenv("VLASTUDIO_CONFIG_PATH", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    assert resolve_config("act", "policy") == custom


@pytest.mark.parametrize("kind", ["policy", "device", "robot", "action_manager", "dataset"])
def test_file_extensions(kind, tmp_path):
    file = tmp_path / "my device.py"
    file.write_text("class Component:\n    def __init__(self, value): self.value = value\n")
    component = create({"type": f"{file}:Component", "args": {"value": 5}}, kind=kind)
    assert component.value == 5


def test_registration():
    register("device", "test_device", dict)
    assert resolve("@device/test_device") is dict
    with pytest.raises(ValueError):
        resolve("@device/test_device", kind="policy")


def test_config_extension(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("type: policy.act")
    register("config", "test_config", lambda: path)
    assert resolve_config("@config/test_config", "policy") == path


def test_cache_precedence(tmp_path, monkeypatch):
    monkeypatch.setenv("VLASTUDIO_CACHE_DIR", str(tmp_path / "env"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "existing"))
    assert cache_root() == tmp_path / "env"
    cache = cache_root(tmp_path / "explicit")
    assert cache == tmp_path / "explicit"
    assert runtime_env(cache)["HF_HOME"] == str(tmp_path / "existing")
    assert runtime_env(cache, tmp_path / "model")["HF_HOME"] == str(tmp_path / "model")


def test_relative_lockfile(tmp_path):
    (tmp_path / "deps.lock").write_text("PyYAML==6.0.2\n")
    profile = environment_for({"runtime": {"python": "3.11", "lockfile": "deps.lock"}}, tmp_path / "policy.yaml")
    assert profile["lock"] == "PyYAML==6.0.2\n"


def test_offline_never_prepares(tmp_path):
    with pytest.raises(RuntimeError, match="not prepared"):
        prepare({"python": "3.11", "requirements": []}, tmp_path, os.environ.copy(), offline=True)


def test_unknown_policy_does_not_guess_runtime(tmp_path):
    with pytest.raises(ValueError, match="No managed runtime"):
        environment_for({"type": "custom.policy"}, tmp_path / "config.yaml")


def test_combined_component_conflict(tmp_path):
    from vlastudio.profiles import merge_component_runtimes
    profile = {"python": "3.10", "requirements": []}
    with pytest.raises(ValueError, match="different Python"):
        merge_component_runtimes(profile, {"datasets": [{"runtime": {"python": "3.11"}}]}, tmp_path / "task.yaml")


def test_cli_override_selects_environment(tmp_path, capsys):
    cfg = tmp_path / "policy.yaml"
    cfg.write_text("type: nonexistent.policy")
    assert main(["train", "--dry-run", "-p", str(cfg), "--policy.type", "policy.mlp"]) == 0
    assert json.loads(capsys.readouterr().out)["profile"]["python"] == "3.10"


def test_entrypoint_extension(monkeypatch):
    from importlib.metadata import EntryPoint
    monkeypatch.setattr("vlastudio.extensions.entry_points", lambda **kw: [EntryPoint(name="mapping", value="builtins:dict", group="vlastudio.device")])
    assert create({"type": "@device/mapping", "args": {"value": 8}}) == {"value": 8}


def test_device_resolver_is_used_by_legacy_loader(tmp_path):
    # Read just the standalone resolver through AST: no camera/GPU dependencies needed.
    import ast
    from vlastudio.paths import legacy_root
    tree = ast.parse((legacy_root() / "deploy/base.py").read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_resolve_device_class")
    scope = {"resolve": resolve}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "device_loader", "exec"), scope)
    assert scope["_resolve_device_class"]({"type": "builtins:dict"}) is dict


def test_external_checkpoint_module_is_not_rewritten(tmp_path):
    import ast
    from types import SimpleNamespace
    from vlastudio.paths import legacy_root
    tree = ast.parse((legacy_root() / "policy/direct_loader.py").read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    nodes = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ("detect_policy_type", "load_policy_module")]
    scope = {"Path": Path, "json": json, "resolve": resolve}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "loader", "exec"), scope)
    (tmp_path / "policy_metadata.json").write_text(json.dumps({"policy_module": "json"}))
    loader = SimpleNamespace(_loaded_modules={})
    reference = scope["detect_policy_type"](loader, str(tmp_path))
    assert scope["load_policy_module"](loader, reference) is json


def test_dataset_plugin_not_prefixed():
    import ast
    from vlastudio.paths import legacy_root
    tree = ast.parse((legacy_root() / "data_utils/utils.py").read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_import_class_from_path")
    # Supply the public compatibility import without loading the data subsystem.
    import types
    compat = types.ModuleType("utils.extensions")
    compat.resolve = resolve
    previous = sys.modules.get("utils.extensions")
    sys.modules["utils.extensions"] = compat
    register("dataset", "test_mapping", dict)
    try:
        scope = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "dataset_loader", "exec"), scope)
        assert scope["_import_class_from_path"]("@dataset/test_mapping") is dict
    finally:
        if previous is None:
            sys.modules.pop("utils.extensions", None)
        else:
            sys.modules["utils.extensions"] = previous


def test_sigterm_forwarded_to_worker(tmp_path, monkeypatch):
    from vlastudio.runtime import execute
    import signal
    calls = []
    handlers = {}
    class Child:
        pid = 123456
        def poll(self): return None
        def terminate(self): calls.append("terminate")
        def wait(self, timeout=None):
            handlers[signal.SIGTERM](signal.SIGTERM, None)
            return 143
    def install(sig, handler):
        handlers[sig] = handler
        return signal.SIG_DFL
    monkeypatch.setattr("vlastudio.runtime.subprocess.Popen", lambda *a, **k: Child())
    monkeypatch.setattr("vlastudio.runtime.signal.signal", install)
    if os.name != "nt":
        monkeypatch.setattr("vlastudio.runtime.os.killpg", lambda pid, sig: calls.append((pid, sig)))
    assert execute("train", [], Path(sys.executable), tmp_path, {}) == 143
    assert calls


def test_saved_settings(tmp_path, monkeypatch):
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"cache_dir": str(tmp_path / "saved")}))
    monkeypatch.setenv("VLASTUDIO_SETTINGS", str(settings))
    monkeypatch.delenv("VLASTUDIO_CACHE_DIR", raising=False)
    monkeypatch.delenv("ILSTD_CACHE", raising=False)
    assert cache_root() == tmp_path / "saved"


def test_file_extension_survives_spawn(tmp_path):
    module = tmp_path / "extension.py"
    module.write_text("class Value:\n    answer = 42\n")
    script = tmp_path / "spawn_check.py"
    script.write_text("""import multiprocessing as mp
from vlastudio import resolve

def child(value, connection):
    connection.send(value.answer)
    connection.close()

if __name__ == '__main__':
    value = resolve(REPLACE)()
    ctx = mp.get_context('spawn')
    parent, child_pipe = ctx.Pipe()
    process = ctx.Process(target=child, args=(value, child_pipe))
    process.start()
    assert parent.poll(15)
    assert parent.recv() == 42
    process.join(15)
    assert process.exitcode == 0
""".replace("REPLACE", repr(str(module) + ":Value")))
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=40)
    assert result.returncode == 0, result.stderr

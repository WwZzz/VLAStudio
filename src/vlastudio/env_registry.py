"""Persistent environment inventory and shell activation."""
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
from filelock import FileLock
from .runtime import CORE, run, uv_command, snapshot


def read(cache):
    path = cache / 'environments.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def register(cache, name, python, kind):
    cache.mkdir(parents=True, exist_ok=True)
    with FileLock(str(cache / 'environments.lock')):
        entries = read(cache)
        python = os.path.abspath(os.path.expanduser(str(python)))
        if name in entries and entries[name]['python'] != python:
            import hashlib
            name += '-' + hashlib.sha256(python.encode()).hexdigest()[:8]
        entries[name] = {'python': python, 'kind': kind}
        path = cache / 'environments.json'
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(entries, indent=2), encoding='utf-8')
        os.replace(temp, path)
        (cache / 'last-environment').write_text(name, encoding='utf-8')
    return name


def install(profile, python, cache, env):
    # Install is additive: never synchronize away the user's other packages.
    with tempfile.TemporaryDirectory(prefix='vlastudio-install-') as directory:
        requirements = Path(directory) / 'requirements.txt'
        requirements.write_text(profile.get('lock') or '\n'.join(CORE + profile.get('requirements', [])), encoding='utf-8')
        args = [*uv_command(), 'pip', 'install', '--python', str(python), '-r', str(requirements)]
        if profile.get('overrides'):
            overrides = Path(directory) / 'overrides.txt'
            overrides.write_text('\n'.join(profile['overrides']), encoding='utf-8')
            args += ['--override', str(overrides)]
        run(args, env)
        for overlay in profile.get('overlays', []):
            script = '''import importlib.util, pathlib, shutil, sys
source_module, target_module, relative = sys.argv[1:]
source_root = pathlib.Path(next(iter(importlib.util.find_spec(source_module).submodule_search_locations))).resolve()
target = pathlib.Path(next(iter(importlib.util.find_spec(target_module).submodule_search_locations))).resolve()
source = (source_root / relative).resolve()
source.relative_to(source_root)
target.relative_to(pathlib.Path(sys.prefix).resolve())
shutil.copytree(source, target, dirs_exist_ok=True)
'''
            run([python, '-c', script, overlay['source_module'], overlay['target_module'], overlay['source']], env)
        run([python, '-c', 'import yaml, platformdirs, filelock, loguru'], env)


def activation(cache, name, shell):
    if name is None:
        name = (cache / 'last-environment').read_text(encoding='utf-8')
    entry = read(cache).get(name)
    if entry is None:
        raise ValueError(f'Unknown environment: {name}. Use vlastudio env list.')
    python = Path(entry['python'])
    if not python.is_file():
        raise ValueError(f'Environment interpreter is missing: {python}')
    app = snapshot(cache)
    values = {'PATH': str(python.parent) + os.pathsep + os.environ.get('PATH', ''),
              'VLASTUDIO_ACTIVE_PYTHON': str(python), 'VLASTUDIO_ENV': name,
              'PYTHONPATH': str(app) + os.pathsep + os.environ.get('PYTHONPATH', '')}
    root = python.parent.parent if python.parent.name in ('bin', 'Scripts') else python.parent
    if (root / 'pyvenv.cfg').is_file():
        values['VIRTUAL_ENV'] = str(root)
    if shell == 'powershell':
        return '\n'.join("$env:" + key + " = '" + value.replace("'", "''") + "'" for key, value in values.items())
    return '\n'.join('export ' + key + '=' + shlex.quote(value) for key, value in values.items())


def hook(shell):
    if shell == 'powershell':
        exe = sys.executable.replace("'", "''")
        return """function global:vlastudio {
  if ($args.Count -ge 2 -and $args[0] -eq 'env' -and $args[1] -eq 'activate') {
    $code = & 'EXE' -m vlastudio @args --shell powershell
    if ($LASTEXITCODE -eq 0) { Invoke-Expression ($code -join "`n") }
  } else {
    & 'EXE' -m vlastudio @args
    if ($LASTEXITCODE -eq 0 -and $args.Count -ge 2 -and $args[0] -eq 'env' -and $args[1] -eq 'create' -and $args -notcontains '--dry-run') {
      $tail = @($args | Select-Object -Skip 2)
      $code = & 'EXE' -m vlastudio env activate @tail --shell powershell
      if ($LASTEXITCODE -eq 0) { Invoke-Expression ($code -join "`n") }
    }
  }
}""".replace('EXE', exe)
    exe = shlex.quote(sys.executable)
    return '''vlastudio() {
  if [ "$1" = env ] && [ "$2" = activate ]; then
    local code
    code=$(EXE -m vlastudio "$@" --shell bash) || return $?
    eval "$code"
  else
    EXE -m vlastudio "$@" || return $?
    if [ "$1" = env ] && [ "$2" = create ]; then
      local code
      case " $* " in *" --dry-run "*) return 0 ;; esac
      code=$(EXE -m vlastudio env activate "${@:3}" --shell bash) || return $?
      eval "$code"
    fi
  fi
}'''.replace('EXE', exe)

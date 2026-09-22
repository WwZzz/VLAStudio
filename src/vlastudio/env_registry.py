"""Persistent environment inventory and shell activation."""
import base64
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
from filelock import FileLock
from .runtime import CORE, run, uv_command, build_environment, install_application, ensure_source_layout_packages


def read(cache):
    path = cache / 'environments.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def ensure_base(cache):
    """Record the launcher's environment without installing or changing packages."""
    cache.mkdir(parents=True, exist_ok=True)
    with FileLock(str(cache / 'environments.lock')):
        entries = read(cache)
        if 'base' not in entries:
            python = Path(os.environ.get('VLASTUDIO_BASE_PYTHON') or sys.executable).expanduser().absolute()
            if not python.is_file():
                raise ValueError(f'Base interpreter does not exist: {python}')
            entries['base'] = {'python': str(python), 'kind': 'default'}
            path = cache / 'environments.json'
            temp = path.with_suffix('.tmp')
            temp.write_text(json.dumps(entries, indent=2), encoding='utf-8')
            os.replace(temp, path)
        return entries['base']


def existing_python(cache, name):
    """Return a usable interpreter for a named environment, or None."""
    if not name:
        return None
    entry = read(cache).get(name)
    if entry:
        python = Path(entry['python']).expanduser()
        if python.is_file():
            return python
    python = cache / 'envs' / ('named-' + name) / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    return python if python.is_file() else None


def register(cache, name, python, kind):
    cache.mkdir(parents=True, exist_ok=True)
    with FileLock(str(cache / 'environments.lock')):
        entries = read(cache)
        python = os.path.abspath(os.path.expanduser(str(python)))
        if name in entries and entries[name]['python'] != python:
            import hashlib
            if name == 'base':
                old = entries[name]
                entries['base-' + hashlib.sha256(old['python'].encode()).hexdigest()[:8]] = old
            else:
                name += '-' + hashlib.sha256(python.encode()).hexdigest()[:8]
        entries[name] = {'python': python, 'kind': kind}
        path = cache / 'environments.json'
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(entries, indent=2), encoding='utf-8')
        os.replace(temp, path)
        (cache / 'last-environment').write_text(name, encoding='utf-8')
    return name


def install(profile, python, cache, env):
    env = build_environment(profile, env)
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
        install_application(python, env)
        ensure_source_layout_packages(python, profile, env)
        run([python, '-c', 'import yaml, platformdirs, filelock, loguru, vlastudio'], env)


def activation(cache, name, shell):
    name = name or 'base'
    ensure_base(cache)
    entry = read(cache).get(name)
    if entry is None:
        raise ValueError(f'Unknown environment: {name}. Use vlastudio env create or env install to register base; env list shows registered environments.')
    python = Path(entry['python'])
    if not python.is_file():
        raise ValueError(f'Environment interpreter is missing: {python}')
    from .paths import drop_app_snapshots, package_source_root
    previous = os.environ.get('VLASTUDIO_ACTIVE_PYTHON')
    paths = os.environ.get('PATH', '').split(os.pathsep)
    if previous:
        paths = [p for p in paths if os.path.normcase(p) != os.path.normcase(str(Path(previous).parent))]
    pythonpath = drop_app_snapshots(os.environ.get('PYTHONPATH', ''), cache)
    live = str(package_source_root().parent)
    pythonpath = os.pathsep.join(filter(None, [live, pythonpath]))
    values = {'PATH': os.pathsep.join([str(python.parent), *paths]),
              'VLASTUDIO_ACTIVE_PYTHON': str(python), 'VLASTUDIO_ENV': name,
              'PYTHONPATH': pythonpath}
    root = python.parent.parent if python.parent.name in ('bin', 'Scripts') else python.parent
    if (root / 'pyvenv.cfg').is_file():
        values['VIRTUAL_ENV'] = str(root)
    values.setdefault('VIRTUAL_ENV', '')
    values['CONDA_PREFIX'] = str(root) if (root / 'conda-meta').is_dir() else ''
    if shell == 'powershell':
        return '\n'.join(
            "$env:" + key + " = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('"
            + base64.b64encode(value.encode('utf-8')).decode('ascii') + "'))"
            for key, value in values.items())
    return '\n'.join('export ' + key + '=' + shlex.quote(value) for key, value in values.items())


def hook(shell):
    if shell == 'powershell':
        exe = sys.executable.replace("'", "''")
        return """function global:vlastudio {
  if ($args.Count -ge 2 -and $args[0] -eq 'env' -and ($args[1] -eq 'activate' -or $args[1] -eq 'deactivate')) {
    $code = & 'EXE' -m vlastudio @args --shell powershell
    if ($LASTEXITCODE -eq 0) { Invoke-Expression ($code -join "`n") }
  } else {
    & 'EXE' -m vlastudio @args
    if ($LASTEXITCODE -eq 0 -and $args.Count -ge 2 -and $args[0] -eq 'env' -and $args[1] -eq 'create' -and $args -notcontains '--dry-run') {
      $tail = @($args | Select-Object -Skip 2)
      $code = & 'EXE' -m vlastudio env activate @tail --last-created --shell powershell
      if ($LASTEXITCODE -eq 0) { Invoke-Expression ($code -join "`n") }
    }
  }
}""".replace('EXE', exe)
    exe = shlex.quote(sys.executable)
    return '''_vlastudio_pythonpath() {
  local IFS=: result="" part
  for part in ${PYTHONPATH-}; do
    case "$part" in
      */vlastudio/apps/*) ;;
      *) result="${result:+$result:}$part" ;;
    esac
  done
  printf '%s' "$result"
}
vlastudio() {
  if [ "$1" = env ] && { [ "$2" = activate ] || [ "$2" = deactivate ]; }; then
    local code
    code=$(PYTHONPATH="$(_vlastudio_pythonpath)" EXE -m vlastudio "$@" --shell bash) || return $?
    eval "$code"
  else
    PYTHONPATH="$(_vlastudio_pythonpath)" EXE -m vlastudio "$@" || return $?
    if [ "$1" = env ] && [ "$2" = create ]; then
      local code
      case " $* " in *" --dry-run "*) return 0 ;; esac
      code=$(PYTHONPATH="$(_vlastudio_pythonpath)" EXE -m vlastudio env activate "${@:3}" --last-created --shell bash) || return $?
      eval "$code"
    fi
  fi
}'''.replace('EXE', exe)

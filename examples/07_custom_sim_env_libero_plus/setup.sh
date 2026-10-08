#!/usr/bin/env bash
# Install the example into a regular Python virtual environment.
set -euo pipefail

example_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$example_dir/../.." && pwd)"
python_bin="${PYTHON:-python3.10}"
export VLASTUDIO_CACHE="${VLASTUDIO_CACHE:-$HOME/.cache/vlastudio}"
venv_dir="${VENV_DIR:-$VLASTUDIO_CACHE/envs/example07}"
pip_args=()
setup_args=()
install_system=1
download_assets=1
while (($#)); do
    case "$1" in
        -h|--help)
            cat <<'HELP'
Usage: bash examples/07_custom_sim_env_libero_plus/setup.sh [options]
  -i, --index-url URL  Override the Python package index.
  --root PATH         Reuse an existing LIBERO-Plus checkout.
  --assets PATH       Reuse extracted assets instead of downloading them.
  --skip-system       Skip apt packages when native dependencies are installed.
Environment: PYTHON (Python 3.10), VLASTUDIO_CACHE, VENV_DIR.
HELP
            exit 0 ;;
        -i|--index-url|--root|--assets)
            if (($# < 2)); then echo "Missing value for $1" >&2; exit 2; fi
            case "$1" in
                -i|--index-url) pip_args+=(--index-url "$2") ;;
                --root) setup_args+=(--root "$2") ;;
                --assets) setup_args+=(--assets "$2"); download_assets=0 ;;
            esac
            shift 2 ;;
        --skip-system) install_system=0; shift ;;
        *) echo "Unknown option: $1 (see --help)" >&2; exit 2 ;;
    esac
done

if [[ "$(uname -s)" != Linux ]]; then
    echo "This example requires Linux and an NVIDIA GPU." >&2
    exit 1
fi
"$python_bin" -c 'import sys; assert sys.version_info[:2] == (3, 10), "Set PYTHON to a Python 3.10 interpreter"'
if ((install_system)); then
    if ! command -v apt-get >/dev/null; then
        echo "Install the native dependencies listed in README.md, then use --skip-system." >&2
        exit 1
    fi
    apt=(apt-get)
    if ((EUID != 0)); then apt=(sudo apt-get); fi
    "${apt[@]}" update
    "${apt[@]}" install -y build-essential python3.10-dev python3.10-venv git \
        libmagickwand-6.q16-6 libgl1 libegl1 libopengl0
fi
mkdir -p "$VLASTUDIO_CACHE" "$(dirname -- "$venv_dir")"
export VLASTUDIO_CACHE="$(cd -- "$VLASTUDIO_CACHE" && pwd)"
if [[ ! -f "$venv_dir/pyvenv.cfg" ]]; then
    if [[ -e "$venv_dir" ]]; then
        echo "Refusing to overwrite a directory that is not a venv: $venv_dir" >&2
        exit 1
    fi
    "$python_bin" -m venv "$venv_dir"
fi
venv_dir="$(cd -- "$venv_dir" && pwd)"
python="$venv_dir/bin/python"
"$python" -c 'import sys; assert sys.version_info[:2] == (3, 10), "The existing venv must use Python 3.10"'
"$python" -m pip install "${pip_args[@]}" 'PyYAML==6.0.2'
requirements="$(mktemp)"
trap 'rm -f -- "$requirements"' EXIT
"$python" - "$example_dir/config/runtime.yaml" "$requirements" <<'PY'
import sys
from pathlib import Path
import yaml
manifest = yaml.safe_load(Path(sys.argv[1]).read_text())
Path(sys.argv[2]).write_text("\n".join(manifest["requirements"]) + "\n")
PY
# Installing the project also installs shared dependencies such as loguru.
"$python" -m pip install "${pip_args[@]}" -r "$requirements" -e "$repo_root"
if ((download_assets)); then setup_args+=(--download-assets); fi
"$python" "$example_dir/setup.py" "${setup_args[@]}"

# One activation command selects Python and the simulator configuration.
{
    printf 'source %q\n' "$venv_dir/bin/activate"
    printf 'export VLASTUDIO_CACHE=%q\n' "$VLASTUDIO_CACHE"
    printf 'source %q\n' "$VLASTUDIO_CACHE/libero-plus/env.sh"
} > "$venv_dir/activate.sh"
printf '\nSetup complete. Run from the repository root:\n'
printf 'source %q\n' "$venv_dir/activate.sh"
printf 'python examples/07_custom_sim_env_libero_plus/train_and_evaluate.py\n'

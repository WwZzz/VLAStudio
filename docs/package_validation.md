# Packaging validation

Validation date: 2026-09-14. Branch: `codex/package-runtime-isolation`.
Base: `db79c4c12c4858e310a6990c5c4c9f4b4d76f381` (main, PR #39).

## Verified locally

- 24 packaging and extension contract tests passed on Windows, including actual
  multiprocessing spawn with an external file-based component.
- Wheel and source distribution built successfully. The wheel contains the legacy
  implementation inside `vlastudio/_legacy`, without generic top-level packages.
- Installed the wheel into a clean Python 3.11 environment; importing the public
  package does not load torch or create policy environments.
- Managed Python 3.10 ACT/MLP dependencies installed successfully. A real MLP
  training run completed two CPU steps using an external dataset and YAML configs,
  launched from outside the source tree through the installed wheel. Checkpoints
  were saved to a user-selected directory containing spaces.
- A dependency-free custom task ran through its managed Python 3.11 entrypoint,
  including offline reuse. A custom device ran start and close successfully.
- Linux OpenPI (Python 3.11, glibc >= 2.31) and OpenVLA (Python 3.10) dependency
  resolution completed. Their resolved locks are included, along with the Windows
  ACT/MLP lock. OpenPI's transformer overlay is declared in its runtime profile.

## Remaining integration coverage

OpenPI/OpenVLA GPU installation, model execution, hardware SDKs, physical robots,
and simulation backends have not been validated here. Linux dependency resolution
is not a GPU training test. The added Windows/Linux CI workflow has not yet run.
Unsupported policies require an explicit runtime manifest or an existing environment.
A managed Python `Policy.load().predict()` proxy is outside this first version;
use task dispatch or the existing policy-server clients.

This is a development package, not a PyPI release. Build artifacts can be installed
with `pip install /path/to/vlastudio-0.2.0.dev0-py3-none-any.whl`.

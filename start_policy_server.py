"""Compatibility entry point; implementation lives in vlastudio.entrypoints.start_policy_server."""
import runpy

if __name__ == "__main__":
    from vlastudio.compat import enable_legacy_imports
    enable_legacy_imports()
    runpy.run_module("vlastudio.entrypoints.start_policy_server", run_name="__main__")

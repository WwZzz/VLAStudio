"""Compatibility entry point; implementation lives in vlastudio.entrypoints.eval_sim."""
import runpy

if __name__ == "__main__":
    from vlastudio.compat import enable_legacy_imports
    enable_legacy_imports()
    runpy.run_module("vlastudio.entrypoints.eval_sim", run_name="__main__")

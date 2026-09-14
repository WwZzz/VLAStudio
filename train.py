"""Compatibility entry point; implementation lives in vlastudio.entrypoints.train."""
import runpy

if __name__ == "__main__":
    from vlastudio.compat import enable_legacy_imports
    enable_legacy_imports()
    runpy.run_module("vlastudio.entrypoints.train", run_name="__main__")

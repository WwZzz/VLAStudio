"""Compatibility access to the public extension resolver."""
try:
    from vlastudio.extensions import resolve
except ImportError:
    import importlib
    def resolve(reference, kind=None, module=False):
        if module:
            return importlib.import_module(reference)
        name, attribute = reference.rsplit(".", 1)
        return getattr(importlib.import_module(name), attribute)

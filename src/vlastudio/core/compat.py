"""Resolve historical module names for existing configs and serialized checkpoints."""
import importlib
import importlib.abc
import importlib.util
import sys

ROOTS = frozenset({'policy', 'benchmark', 'data_utils', 'deploy', 'utils', 'configs'})


def canonical(reference):
    if reference.split('.', 1)[0].split(':', 1)[0] in ROOTS:
        return 'vlastudio.' + reference
    return reference


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, name):
        self.name = name

    def create_module(self, spec):
        module = importlib.import_module(self.name)
        self.original_spec = module.__spec__
        return module

    def exec_module(self, module):
        # The old name and canonical name must reference exactly the same classes.
        module.__spec__ = self.original_spec


class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        name = canonical(fullname)
        if name == fullname:
            return None
        spec = importlib.util.find_spec(name)
        if spec is None:
            return None
        return importlib.util.spec_from_loader(
            fullname, _AliasLoader(name), is_package=spec.submodule_search_locations is not None)


def enable_legacy_imports():
    """Enable old pickle/import names explicitly inside a task worker only."""
    if not any(isinstance(f, _AliasFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _AliasFinder())

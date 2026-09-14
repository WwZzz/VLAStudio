"""Minimal external components; no VLAStudio source modifications required."""
class Device:
    def __init__(self, name="example", **kwargs):
        self.name = name
        self.is_running = True
    def start(self):
        pass
    def close(self):
        self.is_running = False

class Robot(Device):
    def read(self):
        return {"state": [0.0]}
    def write(self, action):
        self.last_action = action

class ActionManager:
    def __init__(self, **kwargs):
        pass
    def update(self, action):
        return action

class Dataset:
    def __init__(self, size=2):
        self.size = size
    def __len__(self):
        return self.size
    def __getitem__(self, index):
        return {"state": [float(index)], "action": [0.0]}


def run(command, argv):
    """An optional custom task entry point using the managed environment."""
    import json
    import sys
    print(json.dumps({"command": command, "args": argv, "python": sys.executable}))
    return 0

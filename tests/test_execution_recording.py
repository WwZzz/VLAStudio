"""Slow captures must retain their wall-clock duration in a CFR video."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np


class RecordingTimingTests(TestCase):
    def test_capture_gap_repeats_previous_frame_and_preserves_timestamps(self):
        from vlastudio.deploy.execution_session.recording import Recorder
        encoded = []
        writer = SimpleNamespace(append_data=lambda frame:encoded.append(frame.copy()), close=lambda:None)
        with tempfile.TemporaryDirectory() as directory, patch('imageio.v2.get_writer',return_value=writer):
            recorder = Recorder(directory,10)
            recorder.frame(SimpleNamespace(image=np.zeros((1,3,32,32),dtype=np.uint8)), captured_at=10)
            recorder.frame(SimpleNamespace(image=np.full((1,3,32,32),255,dtype=np.uint8)), captured_at=11)
            recorder.close()
            rows = [json.loads(line) for line in (Path(directory)/'traces/frames.jsonl').read_text().splitlines()]
        self.assertEqual(len(encoded),11)
        self.assertTrue(all(not frame.any() for frame in encoded[:10]))
        self.assertTrue((encoded[10]==255).all())
        self.assertEqual([row['elapsed_s'] for row in rows],[0,1])

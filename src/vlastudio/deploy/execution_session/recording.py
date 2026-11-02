"""Bounded asynchronous video recording for native execution sessions."""
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from queue import Queue, Full
from threading import Thread
from types import SimpleNamespace
import time

import numpy as np


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def observation_frame(observation):
    """Tile available normalized camera images without camera-name assumptions."""
    images = getattr(observation, 'image', None)
    if images is None:
        return None
    if isinstance(images, dict):
        parts = list(images.values())
    else:
        array = np.asarray(images)
        parts = list(array) if array.ndim == 4 else [array]
    frames = []
    for part in parts:
        frame = np.asarray(part)
        while frame.ndim > 3:
            frame = frame[0]
        if frame.ndim == 2:
            frame = np.repeat(frame[..., None], 3, axis=-1)
        if frame.ndim != 3:
            continue
        if frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (1, 3, 4):
            frame = frame.transpose(1, 2, 0)
        if frame.shape[-1] == 1:
            frame = np.repeat(frame, 3, axis=-1)
        if frame.dtype != np.uint8:
            frame = np.clip(frame * (255 if frame.size and frame.max() <= 1 else 1), 0, 255).astype(np.uint8)
        frames.append(frame[..., :3])
    if not frames:
        return None
    height = max(frame.shape[0] for frame in frames)
    return np.concatenate([np.pad(frame, ((0, height-frame.shape[0]), (0, 0), (0, 0)))
                           for frame in frames], axis=1)


class Recorder:
    def __init__(self, output, fps):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / 'traces').mkdir(exist_ok=True)
        self.video = self.output / 'rollout.mp4'
        self.events = (self.output / 'events.jsonl').open('a', encoding='utf-8', buffering=1)
        self.actions = (self.output / 'traces/actions.jsonl').open('a', encoding='utf-8', buffering=1)
        self.frames = (self.output / 'traces/frames.jsonl').open('a', encoding='utf-8', buffering=1)
        self.queue = Queue(maxsize=16)
        self.error = None
        self.dropped_frames = 0
        self.fps = fps
        self.thread = Thread(target=self._encode, daemon=True)
        self.thread.start()

    def event(self, event, **fields):
        self.events.write(json.dumps({'at': time.perf_counter(), 'event': event, **fields}, default=json_value) + '\n')

    def action(self, **fields):
        self.actions.write(json.dumps({'at': time.perf_counter(), **fields}, default=json_value) + '\n')

    def frame(self, observation, *, context=None, captured_at=None):
        # The encoder owns image conversion and disk I/O, independently of control.
        images = getattr(observation, 'image', None)
        if images is None or self.error:
            return
        # Environments may reuse their image buffers on the next step.
        images = ({key: np.array(value, copy=True) for key, value in images.items()}
                  if isinstance(images, dict) else np.array(images, copy=True))
        try:
            self.queue.put_nowait((SimpleNamespace(image=images),
                                  time.perf_counter() if captured_at is None else captured_at,
                                  dict(context or {})))
        except Full:
            self.dropped_frames += 1

    def _encode(self):
        writer = None
        previous, origin, encoded = None, None, 0
        pending = False
        try:
            import imageio.v2 as imageio
            while True:
                item = self.queue.get()
                if item is None:
                    break
                observation, stamp, context = item
                frame = observation_frame(observation)
                if frame is None:
                    continue
                if writer is None:
                    writer = imageio.get_writer(str(self.video), fps=self.fps, macro_block_size=2)
                    origin = stamp
                elapsed = max(0, stamp-origin)
                # CFR containers need repeated frames across slow captures.
                # Otherwise a slow simulator is silently shown in fast motion.
                while previous is not None and encoded / self.fps < elapsed:
                    writer.append_data(previous)
                    encoded += 1
                self.frames.write(json.dumps({'at':stamp, 'elapsed_s':elapsed,
                    'video_frame':encoded, **context}, default=json_value)+'\n')
                if context:
                    from PIL import Image, ImageDraw
                    canvas = Image.fromarray(frame)
                    draw = ImageDraw.Draw(canvas)
                    draw.rectangle((0,0,canvas.width,28), fill='black')
                    draw.text((3,2), f"v{context.get('version',0)} {context.get('label','')}"[:70], fill='white')
                    draw.text((3,15), f"{context.get('state','')} | elapsed {elapsed:.2f}s", fill='white')
                    frame = np.asarray(canvas)
                previous = frame
                if encoded / self.fps <= elapsed + 1e-6:
                    writer.append_data(frame)
                    encoded += 1
                    pending = False
                else:
                    pending = True
            if writer is not None and previous is not None and pending:
                writer.append_data(previous)
        except Exception as exc:
            self.error = str(exc)
        finally:
            if writer is not None:
                try:
                    writer.close()
                except Exception as exc:
                    self.error = f'Video finalization failed: {exc}'

    def result(self, status):
        target = self.output / 'result.json'
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(status, indent=2, default=json_value), encoding='utf-8')
        temporary.replace(target)

    def close(self):
        if self.thread.is_alive():
            # Drain bounded pending frames and finalize the container before ACK.
            self.queue.put(None, timeout=10)
            self.thread.join(timeout=15)
        self.events.close()
        self.actions.close()
        if not self.thread.is_alive():
            self.frames.close()
        if self.thread.is_alive():
            raise TimeoutError('Video encoder did not finish')
        if self.error:
            raise RuntimeError(f'Video recording failed: {self.error}')

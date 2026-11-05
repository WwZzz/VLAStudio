"""Exposure sweep through the same camera processes and SHM as VR teleoperation."""
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import struct
import time
import cv2
import numpy as np
from vlastudio.deploy.utils import start_devices, get_shm_name
from vlastudio.deploy.shm_utils import SharedMemoryChannel, cleanup_all_shm

AUTO, EXPOSURE, PRIORITY = 0x9a0901, 0x9a0902, 0x9a0903
GAIN, GAMMA = 0x980913, 0x980910
POWER_LINE = 0x980918
BACKLIGHT = 0x98091c


def control(path, cid, value=None):
    with open(path, 'rb+', buffering=0) as device:
        data = bytearray(struct.pack('Ii', cid, value or 0))
        fcntl.ioctl(device, 0xc008561c if value is not None else 0xc008561b, data)
        return struct.unpack('Ii', data)[1]


def controls(path):
    result = {}
    with open(path, 'rb+', buffering=0) as device:
        cid = 0x80000000
        while True:
            data = bytearray(68)
            struct.pack_into('I', data, 0, cid)
            try:
                fcntl.ioctl(device, 0xc0445624, data)
            except OSError:
                break
            cid, kind = struct.unpack_from('II', data)
            name = bytes(data[8:40]).split(b'\0')[0].decode()
            minimum, maximum, step, default, flags = struct.unpack_from('iiiiI', data, 40)
            try:
                current = control(path, cid)
            except OSError:
                current = None
            result[name] = dict(id=cid, current=current, min=minimum, max=maximum,
                                step=step, default=default, flags=flags)
            cid |= 0x80000000
    return result


def run(config, exposures, seconds, output, restore=True, gain=None, gamma=None, auto_exposure=False, power_line=None, backlight=None):
    # Camera-only configs: robot and teleop classes are never instantiated.
    rows = [r for r in config if 'opencv_camera.OpenCVCamera' in r['type']]
    assert 1 <= len(rows) <= 3
    names = [get_shm_name(r) for r in rows]
    wrists = [r for r in rows if 'wrist' in get_shm_name(r)]
    assert wrists
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    snapshots = {get_shm_name(r): controls(r['args']['index_or_path']) for r in rows}
    (output / 'controls_before.json').write_text(json.dumps(snapshots, indent=2))
    saved = {}
    for r in wrists:
        path = r['args']['index_or_path']
        available = {v['id'] for v in snapshots[get_shm_name(r)].values()}
        optional = ([GAIN] if gain is not None or auto_exposure else []) + ([GAMMA] if gamma is not None or auto_exposure else [])
        if power_line is not None:
            optional.append(POWER_LINE)
        if backlight is not None:
            optional.append(BACKLIGHT)
        saved[path] = {c: control(path, c) for c in (AUTO, EXPOSURE, PRIORITY, *optional) if c in available}
    processes, channels = [], []
    results = []
    try:
        for path, values in saved.items():
            control(path, AUTO, 3 if auto_exposure else 1)
            if PRIORITY in values:
                control(path, PRIORITY, 0)
            if gain is not None:
                control(path, GAIN, gain)
            if gamma is not None:
                control(path, GAMMA, gamma)
            if power_line is not None:
                control(path, POWER_LINE, power_line)
            if backlight is not None:
                control(path, BACKLIGHT, backlight)
        cleanup_all_shm(names)
        processes = start_devices(rows)
        for name in names:
            channels.append(SharedMemoryChannel(name, is_writer=False, timeout=15))
        for ms in ([None] if auto_exposure else exposures):
            actual = {}
            for r in wrists:
                path = r['args']['index_or_path']
                if not auto_exposure:
                    control(path, EXPOSURE, round(ms * 10))
                actual[get_shm_name(r)] = control(path, EXPOSURE) / 10
            # Drain old buffers and wait for sensor settings to settle.
            settle = time.monotonic() + (10 if auto_exposure else 2)
            while time.monotonic() < settle:
                for ch in channels:
                    ch.read(skip_unchanged=True)
                time.sleep(.001)
            begin = time.monotonic()
            end = begin + seconds
            times = {n: [] for n in names}
            stats = {n: [] for n in names}
            images = {}
            hashes = {n: [] for n in names}
            import zlib
            while time.monotonic() < end:
                if not all(p.is_alive() for p in processes):
                    raise RuntimeError('A camera process exited')
                for name, ch in zip(names, channels):
                    data = ch.read(skip_unchanged=True)
                    if data is None:
                        continue
                    ts = data.get('timestamp')
                    if ts is None or ts < begin or (times[name] and ts <= times[name][-1]):
                        continue
                    times[name].append(float(ts))
                    image = data['image']
                    hashes[name].append(zlib.crc32(image))
                    if len(times[name]) % 15 == 1:
                        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                        stats[name].append([float(gray.mean()), float(np.percentile(gray, 90))])
                        images[name] = image.copy()
                time.sleep(.0005)
            summary = {}
            for n in names:
                ts = np.array(times[n])
                gaps = np.diff(ts) * 1000
                summary[n] = dict(frames=len(ts), fps_count=len(ts) / seconds,
                    fps_interval=(len(ts)-1)/(ts[-1]-ts[0]) if len(ts)>1 else 0,
                    gap_ms_p50_p95_max=np.percentile(gaps, [50,95,100]).tolist() if len(gaps) else [],
                    gray_mean_p90=np.mean(stats[n], axis=0).tolist() if stats[n] else [],
                    identical_adjacent=sum(a==b for a,b in zip(hashes[n],hashes[n][1:])))
            record = dict(exposure_ms_requested=ms, exposure_ms_readback=actual,
                          auto_exposure=auto_exposure,
                          power_line_override=power_line,
                          backlight_override=backlight,
                          wrist_gain_override=gain, wrist_gamma_override=gamma,
                          seconds=seconds, cameras=summary)
            results.append(record)
            (output / 'results.json').write_text(json.dumps(results, indent=2))
            tag = 'auto' if auto_exposure else '%gms' % ms
            (output / ('timestamps_%s.json' % tag)).write_text(json.dumps(times))
            tiles = []
            for n in names:
                if n in images:
                    tile = cv2.cvtColor(images[n], cv2.COLOR_RGB2BGR)
                    label = n + ' exposure=' + tag if n in actual else n + ' unchanged'
                    cv2.putText(tile, label, (8,25), cv2.FONT_HERSHEY_SIMPLEX, .55, (0,255,0), 1)
                    tiles.append(tile)
            if tiles:
                cv2.imwrite(str(output / ('exposure_%s.jpg' % tag)), np.concatenate(tiles, axis=1))
            (output / 'controls_end.json').write_text(json.dumps(
                {get_shm_name(r): controls(r['args']['index_or_path']) for r in rows}, indent=2))
            print(json.dumps(record), flush=True)
    finally:
        for p in processes:
            if p.is_alive():
                p.terminate()
        for p in processes:
            p.join(timeout=3)
            if p.is_alive():
                p.kill()
                p.join()
        for ch in channels:
            if ch.shm is not None:
                ch.shm.close()
        cleanup_all_shm(names)
        if restore:
            for path, values in saved.items():
                control(path, AUTO, 1)
                control(path, EXPOSURE, values[EXPOSURE])
                for cid in (GAIN, GAMMA, POWER_LINE, BACKLIGHT):
                    if cid in values:
                        control(path, cid, values[cid])
                if PRIORITY in values:
                    control(path, PRIORITY, values[PRIORITY])
                control(path, AUTO, values[AUTO])
            print('Original exposure controls restored.', flush=True)
    return results

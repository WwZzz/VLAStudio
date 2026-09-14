"""
Tkinter slider GUI for calibrating a parallel gripper that cannot be moved by hand.

Steps:
1) Close gripper with the slider → confirm minimum (closed)
2) Open gripper with the slider → confirm maximum (open)
"""

from __future__ import annotations

import threading
import time
import tkinter as tk
from tkinter import ttk


def _sts_resolution(bus, motor: str) -> int:
    """Return max raw encoder value (resolution - 1) for a motor model."""
    model = bus.motors[motor].model
    table = getattr(bus, "model_resolution_table", None)
    if table is not None and model in table:
        return int(table[model]) - 1
    return 4095


def calibrate_gripper_with_slider(bus, motor: str = "gripper") -> tuple[int, int, int, int]:
    """
    Run interactive gripper calibration: set closed, then open.

    Returns:
        (range_min, range_max, homing_offset, drive_mode)

    Home is the **closed** pose (Present_Position = 0 after homing), not mid/half-turn.
    ``drive_mode`` ensures normalized RANGE_0_100 still maps 0→closed and 100→open
    even when opening decreases the raw encoder.
    """
    # Start from factory-like frame (Homing_Offset=0) so recorded poses are absolute encoder values.
    bus.reset_calibration([motor])
    res_max = _sts_resolution(bus, motor)
    present0 = int(bus.read("Present_Position", motor, normalize=False))

    state = {
        "phase": "closed",  # closed -> open
        "goal": present0,
        "present": present0,
        "closed_v": None,
        "open_v": None,
        "paused": False,
        "finished": False,
        "error": None,
    }
    lock = threading.Lock()
    stop_event = threading.Event()

    bus.write("Torque_Enable", motor, 1)

    def io_loop():
        while not stop_event.is_set():
            with lock:
                paused = state["paused"]
                goal = int(state["goal"])
            if paused:
                time.sleep(0.05)
                continue
            try:
                bus.write("Goal_Position", motor, goal, normalize=False)
                present = int(bus.read("Present_Position", motor, normalize=False))
            except Exception as e:
                with lock:
                    state["error"] = str(e)
                time.sleep(0.05)
                continue

            with lock:
                state["present"] = present
                state["error"] = None
            time.sleep(0.02)

    worker = threading.Thread(target=io_loop, daemon=True)
    worker.start()

    root = tk.Tk()
    root.title(f"SO101-PP Gripper Calibration — {motor}")
    root.geometry("560x280")

    info = tk.StringVar(
        value=(
            "Step 2a: Use the slider to fully CLOSE the parallel gripper, then click "
            "'Set min (closed)'\nor press Enter in this window."
        )
    )
    stats = tk.StringVar(value="")
    status = tk.StringVar(value="")

    ttk.Label(root, textvariable=info, justify=tk.LEFT, wraplength=520).pack(padx=12, pady=(12, 6), anchor="w")

    slider = ttk.Scale(root, from_=0, to=res_max, orient=tk.HORIZONTAL, length=500)
    slider.set(present0)
    slider.pack(padx=12, pady=8)

    ttk.Label(root, textvariable=stats, font=("monospace", 11)).pack(padx=12, pady=4, anchor="w")
    ttk.Label(root, textvariable=status, foreground="#666").pack(padx=12, pady=2, anchor="w")

    btn_frame = ttk.Frame(root)
    btn_frame.pack(pady=10)

    def on_slider(_v=None):
        with lock:
            state["goal"] = int(float(slider.get()))

    slider.configure(command=on_slider)

    def refresh_labels():
        with lock:
            present = state["present"]
            goal = state["goal"]
            closed_v = state["closed_v"]
            open_v = state["open_v"]
            phase = state["phase"]
            err = state["error"]
            finished = state["finished"]
        closed_s = "----" if closed_v is None else f"{closed_v:4d}"
        open_s = "----" if open_v is None else f"{open_v:4d}"
        stats.set(
            f"goal={goal:4d}   present={present:4d}   closed={closed_s}   open={open_s}   step={phase}"
        )
        if err:
            status.set(f"Error: {err}")
        if not finished:
            root.after(50, refresh_labels)

    def confirm_closed():
        with lock:
            if state["phase"] != "closed":
                return
            present = int(state["present"])
            state["closed_v"] = present
            state["phase"] = "open"
        info.set(
            "Step 2b: Use the slider to fully OPEN the parallel gripper, then click "
            "'Set max (open)'\nor press Enter in this window."
        )
        status.set(f"Recorded closed position = {present}")

    def confirm_open_and_finish():
        with lock:
            if state["phase"] != "open":
                status.set("Set min (closed) first")
                return
            present = int(state["present"])
            closed_v = state["closed_v"]
            if closed_v is None:
                status.set("Closed position is not set")
                return
            if present == closed_v:
                status.set("Open equals closed; move the gripper to the other end and confirm again")
                return
            state["open_v"] = present
            state["finished"] = True
            state["paused"] = True
        stop_event.set()
        root.destroy()

    def on_enter(_event=None):
        with lock:
            phase = state["phase"]
        if phase == "closed":
            confirm_closed()
        elif phase == "open":
            confirm_open_and_finish()

    def on_close():
        with lock:
            state["paused"] = True
        stop_event.set()
        root.destroy()

    ttk.Button(btn_frame, text="Set min (closed)", command=confirm_closed).pack(side=tk.LEFT, padx=6)
    ttk.Button(btn_frame, text="Set max (open)", command=confirm_open_and_finish).pack(side=tk.LEFT, padx=6)

    root.bind("<Return>", on_enter)
    root.protocol("WM_DELETE_WINDOW", on_close)
    refresh_labels()

    print(
        "\n[Gripper GUI] Window opened.\n"
        "  1) Fully close → 'Set min (closed)' or Enter\n"
        "  2) Fully open  → 'Set max (open)' or Enter\n"
    )
    root.mainloop()
    stop_event.set()
    worker.join(timeout=1.0)

    with lock:
        closed_v = state["closed_v"]
        open_v = state["open_v"]

    if closed_v is None or open_v is None:
        raise RuntimeError("Gripper calibration aborted before both closed and open were set.")
    if closed_v == open_v:
        raise RuntimeError(f"Gripper range invalid: closed={closed_v} open={open_v}.")

    # Homing_Offset=0. Store raw encoder span as [min, max].
    # drive_mode is used as a so101_pp flag: 1 means closed_raw > open_raw
    # (typical PP gripper: open≈1505, closed≈2700) and normalized values must be
    # flipped in so101_pp so API stays 0=closed, 100=open.
    homing_offset = 0
    range_min = int(min(closed_v, open_v))
    range_max = int(max(closed_v, open_v))
    drive_mode = 1 if closed_v > open_v else 0

    bus.write("Homing_Offset", motor, homing_offset, normalize=False)
    bus.write("Min_Position_Limit", motor, range_min, normalize=False)
    bus.write("Max_Position_Limit", motor, range_max, normalize=False)

    print(
        f"[Gripper] closed_raw={closed_v}, open_raw={open_v} → "
        f"range=[{range_min}, {range_max}], invert_flag(drive_mode)={drive_mode} "
        f"(API: 0=closed, 100=open)"
    )
    return int(range_min), int(range_max), int(homing_offset), int(drive_mode)

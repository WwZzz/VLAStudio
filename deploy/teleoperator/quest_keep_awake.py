"""Keep Meta Quest display awake during VR teleop via adb proximity spoof.

Sends ``com.oculus.vrpowermanager.prox_close`` periodically so removing the
headset does not immediately sleep the device. The wireless adb address is
derived from the WebXR client's source IP. Failures are reported at most once
per helper instance.
"""

from __future__ import annotations

import atexit
import ipaddress
import shutil
import subprocess
import threading
from typing import List, Optional, Sequence, Tuple

from loguru import logger

PROX_CLOSE_ACTION = "com.oculus.vrpowermanager.prox_close"
# Restores normal proximity / auto-sleep after prox_close spoofing.
PROX_RESTORE_ACTION = "com.oculus.vrpowermanager.automation_disable"
PROX_FAR_ACTION = "com.oculus.vrpowermanager.prox_far"
DEFAULT_INTERVAL_S = 5.0
ADB_TIMEOUT_S = 8.0
# Restore on exit must finish before collect_data kills the teleop subprocess (~3–5s join).
ADB_RESTORE_TIMEOUT_S = 2.0
DEFAULT_ADB_PORT = 5555


def find_adb(explicit: Optional[str] = None) -> Optional[str]:
    if explicit:
        import os
        if os.path.isfile(explicit) and os.access(explicit, os.X_OK):
            return explicit
        return shutil.which(explicit)
    return shutil.which("adb")


def _run_adb(adb: str, args: Sequence[str], timeout: float = ADB_TIMEOUT_S) -> Tuple[int, str, str]:
    try:
        proc = subprocess.run(
            [adb, *args],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except FileNotFoundError:
        return 127, "", "adb executable not found"
    except subprocess.TimeoutExpired:
        return 124, "", f"adb timed out after {timeout}s"
    except OSError as e:
        return 1, "", str(e)


def restore_quest_proximity_best_effort(adb_path: Optional[str] = None) -> None:
    """Parent-process fallback: restore Quest sleep after teleop subprocess exits."""
    helper = QuestKeepAwake(enabled=True, adb_path=adb_path)
    helper._restore_proximity()


def list_adb_devices(adb: str) -> List[str]:
    code, out, err = _run_adb(adb, ["devices"])
    if code != 0:
        return []
    devices = []
    for line in out.splitlines():
        line = line.strip()
        if not line or line.startswith("List of devices"):
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "device":
            devices.append(parts[0])
    return devices


def is_quest_device(adb: str, serial: str) -> bool:
    code, out, _err = _run_adb(
        adb,
        ["-s", serial, "shell", "getprop", "ro.product.model"],
    )
    return code == 0 and "quest" in out.strip().lower()


class QuestKeepAwake:
    """Background helper that spoofs headset-on via adb."""

    def __init__(
        self,
        enabled: bool = True,
        interval_s: float = DEFAULT_INTERVAL_S,
        adb_path: Optional[str] = None,
        adb_port: int = DEFAULT_ADB_PORT,
    ):
        self.enabled = enabled
        self.interval_s = max(1.0, float(interval_s))
        self.adb_path = adb_path
        self.adb_port = int(adb_port)
        self.adb_target: Optional[str] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._warned = False
        self._active = False
        self._adb: Optional[str] = None
        self._serial: Optional[str] = None
        self._wireless_connected = False
        self._atexit_registered = False
        self._restore_done = False
        self._stop_lock = threading.Lock()

    @property
    def active(self) -> bool:
        return self._active

    def _warn_once(self, message: str) -> None:
        if self._warned:
            return
        self._warned = True
        logger.warning(f"[QuestKeepAwake] {message}")

    def set_webxr_client_ip(self, client_ip: str) -> None:
        """Select the Quest adb target from the WebXR TCP peer address."""
        if self._active:
            return
        address = ipaddress.ip_address(str(client_ip).strip())
        if address.is_loopback or address.is_unspecified:
            raise ValueError(f"invalid WebXR client IP for wireless adb: {address}")
        host = f"[{address}]" if address.version == 6 else str(address)
        self.adb_target = f"{host}:{self.adb_port}"

    def _run_for_device(
        self,
        args: Sequence[str],
        timeout: float = ADB_TIMEOUT_S,
    ) -> Tuple[int, str, str]:
        adb = self._adb or find_adb(self.adb_path)
        if not adb:
            return 127, "", "adb not found"
        if not self._serial:
            return 1, "", "adb device not selected"
        return _run_adb(adb, ["-s", self._serial, *args], timeout=timeout)

    def _select_device(self, adb: str) -> bool:
        devices = list_adb_devices(adb)
        usb_devices = [serial for serial in devices if ":" not in serial]
        if len(usb_devices) == 1:
            self._serial = usb_devices[0]
            return True

        if self.adb_target:
            _run_adb(adb, ["connect", self.adb_target])
            if self.adb_target in list_adb_devices(adb):
                self._serial = self.adb_target
                self._wireless_connected = True
                return True
            return False

        if len(devices) == 1:
            self._serial = devices[0]
            return True
        wireless = [serial for serial in devices if ":" in serial]
        if len(wireless) == 1:
            self._serial = wireless[0]
            return True
        return False

    def start_connected_if_available(self) -> bool:
        """Prefer an already-connected Quest: USB first, then wireless."""
        if not self.enabled:
            return False
        adb = find_adb(self.adb_path)
        if not adb:
            return False

        quest_devices = [
            serial for serial in list_adb_devices(adb) if is_quest_device(adb, serial)
        ]
        usb_devices = [serial for serial in quest_devices if ":" not in serial]
        wireless_devices = [serial for serial in quest_devices if ":" in serial]
        if len(usb_devices) == 1:
            serial = usb_devices[0]
            transport = "USB"
        elif not usb_devices and len(wireless_devices) == 1:
            serial = wireless_devices[0]
            transport = "wireless"
        else:
            return False

        self._adb = adb
        self._serial = serial
        if transport == "wireless":
            self.adb_target = serial
        logger.info(f"[QuestKeepAwake] Using connected {transport} Quest: {serial}")
        return self.start()

    def _broadcast(self, action: str, timeout: float = ADB_TIMEOUT_S) -> Tuple[int, str, str]:
        return self._run_for_device(
            ["shell", "am", "broadcast", "-a", action],
            timeout=timeout,
        )

    def _disconnect_wireless(self) -> None:
        if not (self._wireless_connected and self.adb_target):
            return
        adb = self._adb or find_adb(self.adb_path)
        if adb:
            _run_adb(adb, ["disconnect", self.adb_target], timeout=ADB_RESTORE_TIMEOUT_S)
        self._wireless_connected = False

    def _restore_proximity(self) -> None:
        """Re-enable normal take-off sleep so the headset can power-save again."""
        try:
            adb = self._adb or find_adb(self.adb_path)
            if not adb:
                return
            self._adb = adb
            if not self._serial and not self._select_device(adb):
                logger.debug("[QuestKeepAwake] Restore skipped: no unambiguous adb device")
                return
            # Order matches common Quest tooling: disable automation spoof, then prox_far.
            for action in (PROX_RESTORE_ACTION, PROX_FAR_ACTION):
                code, _out, err = self._broadcast(action, timeout=ADB_RESTORE_TIMEOUT_S)
                if code != 0:
                    detail = (err or _out).strip() or f"exit={code}"
                    logger.warning(f"[QuestKeepAwake] Restore '{action}' failed ({detail})")
                    return
            logger.info("[QuestKeepAwake] Proximity / auto-sleep restored (headset may sleep again)")
        except BaseException as e:
            # Must not abort exit path on SIGINT during subprocess.communicate.
            logger.warning(f"[QuestKeepAwake] Restore interrupted or failed: {e!r}")
        finally:
            self._disconnect_wireless()

    def _probe_and_pulse(self) -> bool:
        adb = find_adb(self.adb_path)
        if not adb:
            self._warn_once(
                "Cannot keep Quest screen awake: adb not found. "
                "Install with `sudo apt install adb`, enable USB debugging, then reconnect."
            )
            return False

        self._adb = adb
        if not self._serial and not self._select_device(adb):
            target = self.adb_target or "an authorized Quest"
            self._warn_once(
                f"Cannot keep Quest screen awake: wireless adb target {target} is unavailable. "
                "Enable wireless adb on the headset and verify that port 5555 is reachable."
            )
            return False

        code, _out, err = self._broadcast(PROX_CLOSE_ACTION)
        if code != 0:
            detail = (err or _out).strip() or f"exit={code}"
            self._warn_once(
                f"Cannot keep Quest screen awake: prox_close failed ({detail}). "
                "Developer Mode / USB debugging may be required."
            )
            return False

        return True

    def start(self) -> bool:
        """Start keep-awake loop. Returns True if first pulse succeeded."""
        if not self.enabled:
            return False
        if self._thread is not None and self._thread.is_alive():
            return self._active

        if not self._probe_and_pulse():
            return False

        self._stop.clear()
        self._active = True
        self._thread = threading.Thread(
            target=self._loop,
            name="quest-keep-awake",
            daemon=True,
        )
        self._thread.start()
        if not self._atexit_registered:
            atexit.register(self.stop)
            self._atexit_registered = True
        logger.info(
            f"[QuestKeepAwake] Started (interval={self.interval_s:.0f}s). "
            "Will auto-stop and restore proximity on exit."
        )
        return True

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            code, _out, err = self._broadcast(PROX_CLOSE_ACTION)
            if code != 0:
                # Device unplugged mid-session: warn once, then stop pulsing.
                detail = (err or _out).strip() or f"exit={code}"
                self._warn_once(
                    f"Keep-awake pulse failed mid-session ({detail}); stopping keep-awake."
                )
                self._active = False
                break

    def stop(self) -> None:
        """Stop pulsing and restore Quest proximity sleep (safe to call multiple times)."""
        with self._stop_lock:
            was_active = self._active or (self._thread is not None)
            self._stop.set()
            thread = self._thread
            if thread is not None and thread.is_alive():
                thread.join(timeout=2.0)
            self._thread = None
            self._active = False
            if was_active and not self._restore_done:
                self._restore_proximity()
                self._restore_done = True
                logger.info("[QuestKeepAwake] Stopped")

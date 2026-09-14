"""Keep Meta Quest display awake during VR teleop via adb proximity spoof.

Sends ``com.oculus.vrpowermanager.prox_close`` periodically so removing the
headset does not immediately sleep the device. The wireless adb address is
derived from the WebXR client's source IP. Failures are reported at most once
per helper instance.
"""

from __future__ import annotations

import atexit
import ipaddress
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
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
CACHE_FILE_NAME = "quest_adb_target"


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


def _normalize_wireless_target(value: str, default_port: int) -> Optional[str]:
    value = str(value).strip()
    if not value:
        return None
    host, separator, port_text = value.rpartition(":")
    if not separator:
        host, port_text = value, str(default_port)
    try:
        address = ipaddress.ip_address(host.strip("[]"))
        port = int(port_text)
    except (ValueError, TypeError):
        return None
    if address.version != 4 or address.is_loopback or address.is_unspecified:
        return None
    if not 1 <= port <= 65535:
        return None
    return f"{address}:{port}"


def _default_cache_path() -> Path:
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache_root / "ilstudio" / CACHE_FILE_NAME


def _quest_wifi_ip(adb: str, serial: str) -> Optional[str]:
    code, out, _err = _run_adb(
        adb,
        ["-s", serial, "shell", "ip", "route"],
        timeout=2.0,
    )
    if code != 0:
        return None
    fields = out.replace("\n", " ").split()
    for index, field in enumerate(fields[:-1]):
        if field != "src":
            continue
        try:
            address = ipaddress.ip_address(fields[index + 1])
        except ValueError:
            continue
        if address.version == 4 and not address.is_loopback and not address.is_unspecified:
            return str(address)
    return None


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
        self._ever_active = False
        self._stop_lock = threading.Lock()
        self._cache_path = _default_cache_path()

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

    def _load_cached_target(self) -> Optional[str]:
        try:
            return _normalize_wireless_target(
                self._cache_path.read_text(encoding="utf-8"),
                self.adb_port,
            )
        except OSError:
            return None

    def _save_cached_target(self, target: str) -> None:
        target = _normalize_wireless_target(target, self.adb_port)
        if not target:
            return
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(f"{target}\n", encoding="utf-8")
        except OSError as e:
            logger.debug(f"[QuestKeepAwake] Could not cache adb target: {e}")

    def _neighbor_targets(self) -> List[str]:
        """Return IPv4 neighbors as best-effort legacy adb candidates."""
        try:
            proc = subprocess.run(
                ["ip", "-4", "neigh", "show"],
                capture_output=True,
                text=True,
                timeout=2.0,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return []
        if proc.returncode != 0:
            return []

        targets = []
        for line in proc.stdout.splitlines():
            if " FAILED" in f" {line}":
                continue
            ip_text = line.split(maxsplit=1)[0] if line.strip() else ""
            target = _normalize_wireless_target(ip_text, self.adb_port)
            if target:
                targets.append(target)
        return targets

    def _wireless_candidates(self) -> List[str]:
        candidates = [self._load_cached_target(), self.adb_target]
        candidates.extend(self._neighbor_targets())
        unique = []
        for candidate in candidates:
            target = _normalize_wireless_target(candidate or "", self.adb_port)
            if target and target not in unique:
                unique.append(target)
        return unique

    def _connect_wireless_target(self, adb: str, target: str) -> bool:
        _run_adb(adb, ["connect", target], timeout=2.0)
        if target not in list_adb_devices(adb) or not is_quest_device(adb, target):
            return False
        self._serial = target
        self.adb_target = target
        self._wireless_connected = True
        self._save_cached_target(target)
        logger.info(f"[QuestKeepAwake] Connected to wireless Quest: {target}")
        return True

    def _bootstrap_wireless_from_usb(self, adb: str, usb_serial: str) -> bool:
        """Enable legacy wireless adb dynamically while continuing to prefer USB."""
        wifi_ip = _quest_wifi_ip(adb, usb_serial)
        if not wifi_ip:
            logger.warning("[QuestKeepAwake] Could not determine Quest Wi-Fi address over USB")
            return False
        target = f"{wifi_ip}:{self.adb_port}"
        code, out, err = _run_adb(
            adb,
            ["-s", usb_serial, "tcpip", str(self.adb_port)],
        )
        if code != 0:
            detail = (err or out).strip() or f"exit={code}"
            logger.warning(f"[QuestKeepAwake] Wireless adb bootstrap failed ({detail})")
            return False

        self.adb_target = target
        # tcpip restarts adbd: wait for real transports before starting pulses.
        deadline = time.monotonic() + 12.0
        while time.monotonic() < deadline and not self._stop.is_set():
            _run_adb(adb, ["connect", target], timeout=2.0)
            devices = list_adb_devices(adb)
            if target in devices and is_quest_device(adb, target):
                self._wireless_connected = True
                self._save_cached_target(target)
                self._serial = usb_serial if usb_serial in devices else target
                logger.info(
                    f"[QuestKeepAwake] Wireless adb prepared at {target}; USB remains preferred"
                )
                return True
            self._stop.wait(0.5)
        self._serial = None
        logger.warning("[QuestKeepAwake] ADB restart still pending; background retry will continue")
        return False

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
        quest_devices = [
            serial for serial in list_adb_devices(adb) if is_quest_device(adb, serial)
        ]
        usb_devices = [serial for serial in quest_devices if ":" not in serial]
        if len(usb_devices) == 1:
            self._serial = usb_devices[0]
            return True

        wireless = [serial for serial in quest_devices if ":" in serial]
        if not usb_devices and len(wireless) == 1:
            self._serial = wireless[0]
            self.adb_target = wireless[0]
            self._save_cached_target(wireless[0])
            return True

        for target in self._wireless_candidates():
            if self._connect_wireless_target(adb, target):
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
            self._save_cached_target(serial)
        logger.info(f"[QuestKeepAwake] Using connected {transport} Quest: {serial}")
        if transport == "USB":
            self._bootstrap_wireless_from_usb(adb, serial)
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
            self._warn_once(
                "Cannot keep Quest screen awake yet: no reachable authorized Quest adb device. "
                "Will keep retrying while teleop is running."
            )
            return False

        code, _out, err = self._broadcast(PROX_CLOSE_ACTION)
        if code != 0:
            detail = (err or _out).strip() or f"exit={code}"
            self._warn_once(
                f"Cannot keep Quest screen awake: prox_close failed ({detail}). "
                "Developer Mode / USB debugging may be required."
            )
            self._serial = None
            return False

        self._ever_active = True
        return True

    def start(self) -> bool:
        """Start keep-awake loop. Returns True if first pulse succeeded."""
        if not self.enabled:
            return False
        if self._thread is not None and self._thread.is_alive():
            return self._active

        first_pulse_ok = self._probe_and_pulse()

        self._stop.clear()
        self._active = first_pulse_ok
        self._thread = threading.Thread(
            target=self._loop,
            name="quest-keep-awake",
            daemon=True,
        )
        self._thread.start()
        if not self._atexit_registered:
            atexit.register(self.stop)
            self._atexit_registered = True
        if first_pulse_ok:
            logger.info(
                f"[QuestKeepAwake] Started (interval={self.interval_s:.0f}s). "
                "Will auto-stop and restore proximity on exit."
            )
        else:
            logger.info(
                f"[QuestKeepAwake] Waiting for Quest adb; retrying every "
                f"{self.interval_s:.0f}s"
            )
        return first_pulse_ok

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            if not self._active:
                self._active = self._probe_and_pulse()
                continue
            code, _out, err = self._broadcast(PROX_CLOSE_ACTION)
            if code != 0:
                detail = (err or _out).strip() or f"exit={code}"
                self._warn_once(
                    f"Keep-awake pulse failed mid-session ({detail}); reconnecting."
                )
                self._active = False
                self._serial = None

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
            if was_active and self._ever_active and not self._restore_done:
                self._restore_proximity()
                self._restore_done = True
                logger.info("[QuestKeepAwake] Stopped")

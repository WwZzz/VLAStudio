#!/usr/bin/env bash
# Install SocketCAN hotplug setup for a PEAK PCAN-USB on the Rebot Jetson.
set -euo pipefail
export PATH=/usr/sbin:/usr/bin:/sbin:/bin
case "${1:-}" in
    --check) check_only=1 ;;
    '') check_only=0 ;;
    *) echo "Usage: bash $0 [--check]" >&2; exit 2 ;;
esac
if (( !check_only && EUID != 0 )); then
    echo "Run the installer with sudo." >&2
    exit 1
fi

tmp=$(mktemp -d)
trap 'rm -f -- "$tmp/$unit" "$tmp/$rule"; rmdir -- "$tmp"' EXIT
unit=vlastudio-rebot-pcan@.service
rule=90-vlastudio-rebot-pcan.rules
cat >"$tmp/$rule" <<'RULE'
ACTION=="add", SUBSYSTEM=="net", KERNEL=="can[0-9]*", ATTRS{idVendor}=="0c72", ATTRS{idProduct}=="000c", TAG+="systemd", ENV{SYSTEMD_WANTS}+="vlastudio-rebot-pcan@%k.service"
RULE
cat >"$tmp/$unit" <<'UNIT'
[Unit]
Description=Set up Rebot PCAN-USB SocketCAN interface %i
BindsTo=sys-subsystem-net-devices-%i.device
After=sys-subsystem-net-devices-%i.device

[Service]
Type=oneshot
ExecCondition=/usr/bin/grep -qx 0c72 /sys/class/net/%i/device/../idVendor
ExecCondition=/usr/bin/grep -qx 000c /sys/class/net/%i/device/../idProduct
ExecStart=/usr/sbin/ip link set dev %i type can bitrate 1000000 restart-ms 100
ExecStart=/usr/sbin/ip link set dev %i up
UNIT
systemd-analyze verify "$tmp/$unit"
if ((!check_only)); then
    install -m 0644 "$tmp/$unit" "/etc/systemd/system/$unit"
    install -m 0644 "$tmp/$rule" "/etc/udev/rules.d/$rule"
    systemctl daemon-reload
    udevadm control --reload-rules
fi

for dev in /sys/class/net/can*; do
    [[ -r "$dev/device/../idVendor" && -r "$dev/device/../idProduct" ]] || continue
    [[ $(<"$dev/device/../idVendor") == 0c72 && $(<"$dev/device/../idProduct") == 000c ]] || continue
    name=${dev##*/}
    echo "Matched PCAN-USB: $name"
    if ((!check_only)); then
        flags=$(<"$dev/flags")
        if ((flags & 1)); then
            echo "Interface already up; showing its current settings."
        else
            systemctl start "vlastudio-rebot-pcan@$name.service"
        fi
    fi
    ip -details link show dev "$name"
done

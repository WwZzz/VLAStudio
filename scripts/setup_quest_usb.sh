#!/usr/bin/env bash
set -euo pipefail
if [[ "$EUID" != 0 ]]; then
  echo 'Run: sudo bash /home/jetson/Project/VLAStudio/scripts/setup_quest_usb.sh'
  exit 1
fi
# Give this Jetson account access only to Meta/Oculus USB devices.
RULE=/etc/udev/rules.d/70-vlastudio-quest.rules
if [[ -f "$RULE" ]]; then cp -p "$RULE" "$RULE.backup.$(date +%s)"; fi
printf '%s\n' 'SUBSYSTEM=="usb", ATTR{idVendor}=="2833", MODE="0660", GROUP="jetson"' > "$RULE"
udevadm control --reload-rules
udevadm trigger --action=change --subsystem-match=usb --attr-match=idVendor=2833
echo 'Quest USB permission ready. Wear headset and allow USB debugging, with Always allow checked.'

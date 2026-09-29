#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
    echo "Run this installer with sudo." >&2
    exit 1
fi

readonly SOURCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

install -D -m 0644 "$SOURCE_DIR/10-piper-can-left.link" /etc/systemd/network/10-piper-can-left.link
install -D -m 0644 "$SOURCE_DIR/11-piper-can-right.link" /etc/systemd/network/11-piper-can-right.link
install -D -m 0755 "$SOURCE_DIR/lerobot-piper-can-up" /usr/local/sbin/lerobot-piper-can-up
install -D -m 0644 "$SOURCE_DIR/lerobot-piper-can.service" /etc/systemd/system/lerobot-piper-can.service

systemctl daemon-reload
systemctl enable lerobot-piper-can.service

echo "Installed persistent CAN naming/configuration."
echo "It will apply on the next reboot. To apply now, first stop every PiPER CAN owner, then run:"
echo "  sudo systemctl start lerobot-piper-can.service"

#!/usr/bin/env bash
set -euo pipefail

# Refresh or disable Husarnet repo key to avoid EXPKEYSIG during apt update.
# Run on Ubuntu 20.04 (arm64) Jetson.

if [ "${EUID}" -ne 0 ]; then
  echo "Please run as root: sudo $0"
  exit 1
fi

set -x

# Option A: Update key using keyrings (preferred)
mkdir -p /etc/apt/keyrings
curl -fsSL https://install.husarnet.com/repo.key | gpg --dearmor -o /etc/apt/keyrings/husarnet.gpg

list_file=/etc/apt/sources.list.d/husarnet.list
if [ -f "$list_file" ]; then
  # Force signed-by usage if not present
  if ! grep -q "signed-by=/etc/apt/keyrings/husarnet.gpg" "$list_file"; then
    sed -i 's#^deb \(.*\)$#deb [signed-by=/etc/apt/keyrings/husarnet.gpg] \1#g' "$list_file"
  fi
fi

apt-get update || true

# Option B: Disable if still failing
if apt-get update 2>&1 | grep -q "EXPKEYSIG"; then
  echo "Apt key still invalid; disabling Husarnet repo as fallback."
  if [ -f "$list_file" ]; then
    mv "$list_file" "$list_file.disabled"
  fi
  apt-get update
fi

echo "Done."


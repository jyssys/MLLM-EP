#!/usr/bin/env bash
set -euo pipefail

# Called only inside unshare --user --map-root-user --net --mount --pid --fork.
# No host home or network is mounted. The model-generated function runs as
# nobody, with CPU/file limits applied by the runner and a parent timeout.
sandbox_root=$(mktemp -d /tmp/mini-humaneval-root.XXXXXX)
cleanup_sandbox() {
    umount "$sandbox_root/proc" 2>/dev/null || true
    umount "$sandbox_root/usr" 2>/dev/null || true
    umount "$sandbox_root/lib" 2>/dev/null || true
    umount "$sandbox_root/lib64" 2>/dev/null || true
    umount "$sandbox_root" 2>/dev/null || true
    rmdir "$sandbox_root" 2>/dev/null || true
}
trap cleanup_sandbox EXIT
mount -t tmpfs -o size=128m tmpfs "$sandbox_root"
mkdir -p "$sandbox_root/usr" "$sandbox_root/lib" "$sandbox_root/lib64" "$sandbox_root/tmp" "$sandbox_root/dev" "$sandbox_root/proc"
chmod 1777 "$sandbox_root/tmp"
mount --bind /usr "$sandbox_root/usr"
mount -o remount,bind,ro "$sandbox_root/usr"
mount --bind /lib "$sandbox_root/lib"
mount -o remount,bind,ro "$sandbox_root/lib"
mount --bind /lib64 "$sandbox_root/lib64"
mount -o remount,bind,ro "$sandbox_root/lib64"
cp "$1" "$sandbox_root/runner.py"
chmod 644 "$sandbox_root/runner.py"
mount -t proc proc "$sandbox_root/proc"
chroot "$sandbox_root" /usr/bin/setpriv --no-new-privs --bounding-set=-all \
    /usr/bin/python3 -I -B /runner.py

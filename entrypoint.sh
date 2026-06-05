#!/bin/sh
set -e
# Fix ownership of volume-mounted paths so appuser can write to them.
# Runs as root on container start, then drops privileges before exec.
chown -R appuser:appgroup /app/data 2>/dev/null || true
exec runuser -u appuser -- "$@"

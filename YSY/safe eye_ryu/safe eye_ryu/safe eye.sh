#!/usr/bin/env bash
set -e

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_FILE="$APP_DIR/safe eye.py"

PYTHON="/home/ryu/project/env/bin/python"
if [ ! -x "$PYTHON" ]; then
  PYTHON="$APP_DIR/../env/bin/python"
fi
if [ ! -x "$PYTHON" ]; then
  PYTHON="python3"
fi

cd "$APP_DIR"
exec "$PYTHON" "$APP_FILE" --host 0.0.0.0 --port 8000

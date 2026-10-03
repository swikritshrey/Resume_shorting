#!/usr/bin/env bash
cd "$(dirname "$0")"
command -v python3 >/dev/null || { echo "Python 3 is not installed. Get it from https://www.python.org/downloads/"; exit 1; }
[ -d .venv ] || python3 -m venv .venv
. .venv/bin/activate
python -m pip install -q -r requirements.txt || { echo "Install failed. Check your internet connection and try again."; exit 1; }
echo "Starting. Open http://127.0.0.1:5000 in your browser and keep this window open."
python app.py

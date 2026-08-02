#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3.10+ is required." >&2
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "FFmpeg is required. Install it with your system package manager." >&2
  exit 1
fi

if [[ ! -x ".venv/bin/python" ]]; then
  echo "Creating the Python virtual environment..."
  python3 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
fi

if ! .venv/bin/python -c 'import streamlit, yt_dlp, faster_whisper, openai, google.genai, anthropic' >/dev/null 2>&1; then
  echo "Installing project dependencies..."
  .venv/bin/python -m pip install -r requirements.txt
fi

.venv/bin/python -m pip check
exec .venv/bin/python -m streamlit run app.py --server.showEmailPrompt=false --browser.gatherUsageStats=false

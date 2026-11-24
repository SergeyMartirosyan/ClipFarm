#!/usr/bin/env bash
# One-time install on Ubuntu 22.04/24.04 (Oracle Cloud Ampere A1 or any x86/ARM box).
# Run from the repo folder as your normal user:  ./deploy/setup.sh
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RUN_USER="${SUDO_USER:-$(id -un)}"
cd "$APP_DIR"

echo "==> System packages (ffmpeg, Python, fonts)"
sudo apt-get update -y
sudo apt-get install -y ffmpeg python3-venv python3-pip fonts-dejavu-core git curl

echo "==> Python environment"
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

echo "==> Models"
./scripts/fetch_models.sh
WHISPER_MODEL="$(grep -E '^WHISPER_MODEL=' .env 2>/dev/null | cut -d= -f2 || true)"
.venv/bin/python -c "from faster_whisper import WhisperModel; WhisperModel('${WHISPER_MODEL:-small.en}', device='cpu', compute_type='int8')"

echo "==> Config"
if [ ! -f .env ]; then
  cp .env.example .env
  chmod 600 .env
  echo "Created .env from the template. Fill it in before starting the services."
fi

echo "==> systemd services"
for svc in bot worker poster; do
  sed -e "s#__APP_DIR__#${APP_DIR}#g" -e "s#__USER__#${RUN_USER}#g" \
    "deploy/clipfarm-${svc}.service" | sudo tee "/etc/systemd/system/clipfarm-${svc}.service" >/dev/null
done
sudo systemctl daemon-reload
sudo systemctl enable clipfarm-bot clipfarm-worker clipfarm-poster >/dev/null

cat <<MSG

Done. Next:
  1. nano ${APP_DIR}/.env                      # paste your keys
  2. .venv/bin/python -m clipfarm.check        # checks every key and tool
  3. sudo systemctl restart clipfarm-bot clipfarm-worker
  4. Send your bot a video link in Telegram.

Logs:  journalctl -u clipfarm-worker -f      (or clipfarm-bot)
MSG

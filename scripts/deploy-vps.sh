#!/usr/bin/env bash
set -euo pipefail

: "${APP_DIR:?APP_DIR es requerido}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
NODE_VERSION="${NODE_VERSION:-}"
BACKEND_RESTART_COMMAND="${BACKEND_RESTART_COMMAND:-}"
FRONTEND_RESTART_COMMAND="${FRONTEND_RESTART_COMMAND:-}"

echo "==> Actualizando codigo en $APP_DIR"
cd "$APP_DIR"
git fetch --prune origin main
git reset --hard origin/main

echo "==> Preparando backend"
cd "$APP_DIR/backend"
"$PYTHON_BIN" -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m compileall -q src
deactivate

echo "==> Preparando frontend"
cd "$APP_DIR/frontend"

export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
if [ -s "$NVM_DIR/nvm.sh" ]; then
  # shellcheck source=/dev/null
  . "$NVM_DIR/nvm.sh"
fi

if command -v nvm >/dev/null 2>&1; then
  if [ -n "$NODE_VERSION" ]; then
    nvm install "$NODE_VERSION"
    nvm use "$NODE_VERSION"
  elif [ -f .nvmrc ]; then
    nvm install
    nvm use
  fi
fi

npm ci
npm run build

if [ -n "$BACKEND_RESTART_COMMAND" ]; then
  echo "==> Reiniciando backend"
  bash -lc "$BACKEND_RESTART_COMMAND"
else
  echo "==> BACKEND_RESTART_COMMAND no configurado; backend no reiniciado"
fi

if [ -n "$FRONTEND_RESTART_COMMAND" ]; then
  echo "==> Reiniciando frontend/proxy"
  bash -lc "$FRONTEND_RESTART_COMMAND"
else
  echo "==> FRONTEND_RESTART_COMMAND no configurado; frontend/proxy no reiniciado"
fi

echo "==> Despliegue completado"

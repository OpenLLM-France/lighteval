#!/usr/bin/env bash
# One-time setup of the Node/Jest environment used by the WebApp1K lighteval task.
#
# Requires Node.js >= 18 and npm (see ../README.md for how to install them, including offline
# on an HPC cluster). Run once, with internet access:
#
#     bash setup.sh
#
# This installs node_modules into this directory. After that, the WebApp1K task uses it
# automatically (or set WEBAPP1K_JEST_PROJECT to point elsewhere).
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v node >/dev/null 2>&1; then
  echo "ERROR: node not found. Install Node.js >= 18 first (see ../README.md)." >&2
  exit 1
fi
if ! command -v npm >/dev/null 2>&1; then
  echo "ERROR: npm not found. Install Node.js/npm first (see ../README.md)." >&2
  exit 1
fi

echo "Node:  $(node --version)"
echo "npm:   $(npm --version)"
echo "Installing JS dependencies into $(pwd) ..."

# Prefer a reproducible install when a lockfile is present.
if [ -f package-lock.json ]; then
  npm ci
else
  npm install
fi

echo
echo "Done. Verifying Jest is runnable..."
npx jest --version
echo "WebApp1K Node/Jest environment is ready."

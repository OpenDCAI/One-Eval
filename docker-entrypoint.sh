#!/bin/bash
set -e

echo "🚀 starting backend..."
exec uvicorn one_eval.server.app:app --host 0.0.0.0 --port 8000 &

echo "🚀 starting nginx..."
nginx -g "daemon off;"
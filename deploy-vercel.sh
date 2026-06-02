#!/bin/bash

# Temporary hide pyproject.toml and uv.lock so Vercel builds using requirements.txt
echo "Temporarily hiding pyproject.toml and uv.lock to prevent Vercel dependency conflicts..."
mv pyproject.toml pyproject.toml.bak 2>/dev/null
mv uv.lock uv.lock.bak 2>/dev/null

# Run Vercel deployment with passed arguments (e.g. --prod)
echo "Running Vercel CLI..."
vercel "$@"
STATUS=$?

# Restore pyproject.toml and uv.lock
echo "Restoring pyproject.toml and uv.lock..."
mv pyproject.toml.bak pyproject.toml 2>/dev/null
mv uv.lock.bak uv.lock 2>/dev/null

exit $STATUS

#!/bin/bash
set -e
echo "===== MDKeyChunker v2 Setup ====="

python3 --version

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "Virtual environment created"
fi

source .venv/bin/activate
pip install --upgrade pip -q
pip install -e ".[dev]" -q
echo "Package installed"

if [ ! -f ".env" ]; then
    cp .env.sample .env
    echo "Created .env - edit it with your API key"
else
    echo ".env already exists"
fi

python -c "from mdkeychunker import Pipeline, Config; print('Import test passed')"
echo ""
echo "Done! Next: edit .env, then run: mdkeychunker demo.md"

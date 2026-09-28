#!/bin/bash
# Local mirror of the Cerebro Zero 4.0 release gate.

set -euo pipefail

echo "🧪 Cerebro Zero 4.0 local CI"
python -m pytest -q
python cli.py benchmark
python -m build
cerebro-zero --version
python -c "import numpy, rich; from PIL import Image; print('runtime dependencies: OK')"

echo "✅ Release gate passed"

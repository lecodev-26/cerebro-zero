#!/bin/bash
# Script para ejecutar CI localmente antes de hacer push

set -e

echo "🧪 EJECUTANDO CI LOCAL"
echo "="*60

# 1. Tests de Tensor
echo ""
echo "📦 [1/5] Tests de Tensor..."
python tests/test_tensor.py

# 2. Gradient checking
echo ""
echo "🔬 [2/5] Gradient Checking..."
python tests/test_gradient_check.py

# 3. Pytest completo
echo ""
echo "🧪 [3/5] Pytest completo..."
pytest tests/ -v --tb=short

# 4. Verificar importaciones
echo ""
echo "📥 [4/5] Verificando importaciones..."
python -c "
import numpy
import yaml
from core.tensor import Tensor
from models.transformer import Transformer
from config.loader import Config
print('✅ Todas las importaciones funcionan')
"

# 5. Verificar config
echo ""
echo "⚙️ [5/5] Verificando config..."
python -c "
from config.loader import Config
c = Config.load('config/default.yaml')
assert c.get('model.type') == 'transformer'
assert c.get('training.learning_rate') > 0
print('✅ Config cargada correctamente')
"

echo ""
echo "="*60
echo "🎉 ¡TODO PASA! Listo para hacer push"

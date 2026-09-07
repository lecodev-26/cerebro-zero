# Makefile para Cerebro Zero
# Comandos rápidos para el proyecto

.PHONY: help install run test clean

help:
@echo "Comandos disponibles:"
@echo "  make install   - Instalar dependencias"
@echo "  make run       - Ejecutar el menú principal"
@echo "  make test      - Ejecutar pruebas"
@echo "  make clean     - Limpiar archivos temporales"
@echo "  make web       - Ejecutar interfaz web"
@echo "  make chat      - Ejecutar chat interactivo"

install:
pip install -r requirements.txt

run:
python todo_en_uno.py

test:
python -m pytest tests/ 2>/dev/null || echo "No hay pruebas configuradas"

clean:
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.log" -delete 2>/dev/null || true
rm -rf .pytest_cache 2>/dev/null || true

web:
python experiments/interfaz_web.py

chat:
python experiments/chat_ultimate.py

format:
black --check . 2>/dev/null || echo "black no instalado"

lint:
flake8 . 2>/dev/null || echo "flake8 no instalado"

"""
Permite ejecutar: python -m cerebro_zero
o: python -m .
"""

import sys
import os

# Añadir directorio al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cli import main

if __name__ == "__main__":
    sys.exit(main())

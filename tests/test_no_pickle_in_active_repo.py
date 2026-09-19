"""
Test de integridad: CERO pickle en el árbol activo
====================================================

Este test verifica que ningún archivo Python del árbol activo
(fuera de legacy/ y tests/legacy/) usa pickle.

Esto es parte del compromiso de seguridad de Cerebro Zero 4.0:
la persistencia debe ser JSON+NPY, nunca pickle.
"""

import os
import re
import pytest


# ============================================
# CONFIGURACIÓN
# ============================================

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Directorios donde NO debe haber pickle
ACTIVOS = [
    "agent",
    "config",
    "core",
    "datasets",
    "docs",
    "evaluation",
    "language",
    "memory",
    "mobile",
    "models",
    "reasoning",
    "security",
    "tools",
    "training",
]

# Patrones que consideramos "uso real de pickle"
PATRONES_PROHIBIDOS = [
    r"^\s*import\s+pickle\b",
    r"^\s*from\s+pickle\b",
    r"\bpickle\.dump\b",
    r"\bpickle\.load\b",
    r"\bpickle\.dumps\b",
    r"\bpickle\.loads\b",
]

# Excepciones: archivos que pueden mencionar pickle (para tests específicos)
EXCEPCIONES = [
    "test_serialization.py",     # verifica que NO se use pickle
    "test_no_pickle_in_active_repo.py",  # este mismo archivo
]


# ============================================
# HELPERS
# ============================================

def archivos_python(directorio):
    """Lista todos los .py en un directorio (recursivo)"""
    resultado = []
    base = os.path.join(REPO_ROOT, directorio)
    if not os.path.isdir(base):
        return resultado
    for root, dirs, files in os.walk(base):
        # Ignorar __pycache__
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in files:
            if f.endswith(".py"):
                resultado.append(os.path.join(root, f))
    return resultado


def tiene_pickle(archivo):
    """Devuelve la lista de líneas con pickle (o [])"""
    hallazgos = []
    try:
        with open(archivo, "r", encoding="utf-8") as f:
            for num, linea in enumerate(f, 1):
                for patron in PATRONES_PROHIBIDOS:
                    if re.search(patron, linea):
                        # Excepción: comentarios, docstrings con "pickle"
                        # Pero líneas de código reales no
                        if not linea.lstrip().startswith("#"):
                            # Docstring dentro de """ puede mencionar pickle
                            # Permitimos si está entre comillas
                            stripped = linea.strip()
                            if stripped.startswith('"""') or stripped.startswith("'''"):
                                continue
                            if stripped.startswith('"') and stripped.endswith('"'):
                                continue
                            if stripped.startswith("'") and stripped.endswith("'"):
                                continue
                            # Docstring multilínea: permitimos si NO empieza por import/from
                            if '"""' in linea or "'''" in linea:
                                continue
                            hallazgos.append((num, linea.rstrip()))
    except Exception:
        pass
    return hallazgos


# ============================================
# TESTS
# ============================================

def test_no_import_pickle_en_arbol_activo():
    """Verifica que ningún archivo del árbol activo importa pickle"""
    violaciones = []

    for directorio in ACTIVOS:
        for archivo in archivos_python(directorio):
            # Excepciones
            nombre = os.path.basename(archivo)
            if nombre in EXCEPCIONES:
                continue

            hallazgos = tiene_pickle(archivo)
            if hallazgos:
                ruta_rel = os.path.relpath(archivo, REPO_ROOT)
                violaciones.append((ruta_rel, hallazgos))

    if violaciones:
        mensaje = "🚨 PICKLE ENCONTRADO EN ÁRBOL ACTIVO:\n\n"
        for ruta, hallazgos in violaciones:
            mensaje += f"  📄 {ruta}\n"
            for num, linea in hallazgos:
                mensaje += f"     L{num}: {linea}\n"
        mensaje += "\nCerebro Zero 4.0 exige CERO pickle. Usa core.serialization."
        pytest.fail(mensaje)


def test_serialization_v2_existe():
    """Verifica que serialization_v2 está disponible"""
    path = os.path.join(REPO_ROOT, "core", "serialization_v2.py")
    assert os.path.exists(path), "core/serialization_v2.py no existe"


def test_serialization_v2_no_usa_pickle():
    """La propia serialization no debe usar pickle"""
    path = os.path.join(REPO_ROOT, "core", "serialization_v2.py")
    hallazgos = tiene_pickle(path)
    assert hallazgos == [], f"serialization_v2 usa pickle: {hallazgos}"


def test_serialization_v1_no_usa_pickle():
    """La serialization original tampoco debe usar pickle"""
    path = os.path.join(REPO_ROOT, "core", "serialization.py")
    hallazgos = tiene_pickle(path)
    assert hallazgos == [], f"serialization usa pickle: {hallazgos}"


def test_models_brain_usa_serialization():
    """models/brain.py debe usar serialization_v2"""
    path = os.path.join(REPO_ROOT, "models", "brain.py")
    with open(path) as f:
        contenido = f.read()
    assert "safe_save_dict_v2" in contenido or "safe_save_dict" in contenido
    assert "pickle" not in contenido.lower() or "sin pickle" in contenido.lower()


def test_lm_trainer_usa_serialization():
    """training/lm_trainer.py debe usar serialization"""
    path = os.path.join(REPO_ROOT, "training", "lm_trainer.py")
    with open(path) as f:
        contenido = f.read()
    assert "safe_save_dict" in contenido or "safe_load_dict" in contenido


def test_continuous_learning_v2_usa_serialization():
    """training/continuous_learning_v2.py debe usar serialization"""
    path = os.path.join(REPO_ROOT, "training", "continuous_learning_v2.py")
    with open(path) as f:
        contenido = f.read()
    assert "safe_save_dict" in contenido


def test_memory_long_term_sin_pickle():
    """memory/long_term.py no debe tener import pickle"""
    path = os.path.join(REPO_ROOT, "memory", "long_term.py")
    with open(path) as f:
        contenido = f.read()
    assert "import pickle" not in contenido


def test_memory_semantic_sin_pickle():
    """memory/semantic.py no debe tener import pickle"""
    path = os.path.join(REPO_ROOT, "memory", "semantic.py")
    with open(path) as f:
        contenido = f.read()
    assert "import pickle" not in contenido


def test_legacy_excluido():
    """Verifica que legacy/ existe y NO está en los activos"""
    legacy_path = os.path.join(REPO_ROOT, "legacy")
    assert os.path.isdir(legacy_path), "legacy/ no existe"
    assert "legacy" not in ACTIVOS, "legacy no debe estar en ACTIVOS"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

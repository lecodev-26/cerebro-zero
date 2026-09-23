"""
Tests de la CLI de Cerebro Zero
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import subprocess

# Ruta al cli.py
CLI = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "cli.py",
)


# ============================================
# HELPERS
# ============================================

def run_cli(*args, timeout=30):
    """Ejecuta cli.py con los args dados"""
    cmd = [sys.executable, CLI] + list(args)
    r = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout,
    )
    return r


# ============================================
# TESTS DE VERSIÓN
# ============================================

def test_cli_version():
    r = run_cli("--version")
    assert r.returncode == 0
    assert "3.0.0" in r.stdout


def test_cli_help():
    r = run_cli("--help")
    assert r.returncode == 0
    assert "cerebro-zero" in r.stdout.lower()


def test_cli_sin_argumentos():
    r = run_cli()
    # Sin args, muestra help
    assert "info" in r.stdout or "chat" in r.stdout


# ============================================
# TESTS DE COMANDO INFO
# ============================================

def test_cli_info():
    r = run_cli("info")
    assert r.returncode == 0
    assert "CEREBRO ZERO" in r.stdout
    assert "3.0.0" in r.stdout


def test_cli_info_componentes():
    r = run_cli("info")
    assert "Working Memory" in r.stdout
    assert "World Model" in r.stdout
    assert "Planner 3.0" in r.stdout
    assert "Tokenizer 3.0" in r.stdout


def test_cli_info_estado():
    r = run_cli("info")
    # La salida visual usa "Estado del sistema" (rich table)
    assert "Estado del sistema" in r.stdout or "ESTADO DEL SISTEMA" in r.stdout
    assert "Tools" in r.stdout


# ============================================
# TESTS DE COMANDO CHAT
# ============================================

def test_cli_chat_suma():
    r = run_cli("chat", "5 + 3")
    assert r.returncode == 0
    assert "8" in r.stdout


def test_cli_chat_multiplica():
    r = run_cli("chat", "4 * 5")
    assert "20" in r.stdout


def test_cli_chat_aprender():
    r = run_cli("chat", "aprende color = rojo")
    assert "rojo" in r.stdout


def test_cli_chat_desconocido():
    r = run_cli("chat", "xyzzy foobar")
    assert r.returncode == 0
    assert r.stdout != ""


def test_cli_chat_con_tool():
    """El chat debe indicar la tool usada"""
    r = run_cli("chat", "5 + 3")
    assert "tool=suma" in r.stdout or "suma" in r.stdout


# ============================================
# TESTS DE COMANDO PLAN
# ============================================

def test_cli_plan_entrenar():
    r = run_cli("plan", "Entrenar el modelo")
    assert r.returncode == 0
    assert "cargar_dataset" in r.stdout
    assert "entrenar" in r.stdout


def test_cli_plan_subgoals():
    r = run_cli("plan", "Entrenar el modelo")
    assert "8" in r.stdout  # 8 subgoals
    # Acepta "subgoals" o "Subgoals"
    assert "subgoals" in r.stdout.lower()


def test_cli_plan_evaluar():
    r = run_cli("plan", "evaluar el modelo")
    assert r.returncode == 0


# ============================================
# TESTS DE INTEGRIDAD
# ============================================

def test_cli_archivo_existe():
    assert os.path.exists(CLI)


def test_cli_no_depende_de_cwd():
    """La CLI debe funcionar desde cualquier directorio"""
    r = subprocess.run(
        [sys.executable, CLI, "--version"],
        cwd="/",  # desde la raíz
        capture_output=True, text=True, timeout=10,
    )
    assert r.returncode == 0
    assert "3.0.0" in r.stdout


# ============================================
# TESTS DE ARCHIVOS DE PACKAGING
# ============================================

def test_pyproject_existe():
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "pyproject.toml",
    )
    assert os.path.exists(path)


def test_pyproject_tiene_nombre():
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "pyproject.toml",
    )
    with open(path) as f:
        contenido = f.read()
    assert 'name = "cerebro-zero"' in contenido
    assert "3.0.0" in contenido


def test_pyproject_tiene_script():
    """Debe declarar el comando cerebro-zero"""
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "pyproject.toml",
    )
    with open(path) as f:
        contenido = f.read()
    assert "cerebro-zero = " in contenido


def test_setup_py_existe():
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "setup.py",
    )
    assert os.path.exists(path)


def test_main_modulo_existe():
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "__main__.py",
    )
    assert os.path.exists(path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

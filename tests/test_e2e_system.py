"""System-level end-to-end checks for the 4.0 release."""

import subprocess
import sys


ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
CLI = ROOT / "cli.py"


def run_cli(*args):
    return subprocess.run(
        [sys.executable, str(CLI), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_cli_info_e2e():
    result = run_cli("info")
    assert result.returncode == 0
    assert "CEREBRO ZERO 4.0.0" in result.stdout
    assert "CerebroV3" not in result.stderr


def test_cli_chat_e2e():
    result = run_cli("chat", "5 + 3")
    assert result.returncode == 0
    assert "8.0" in result.stdout
    assert "tool=suma" in result.stdout or "suma" in result.stdout


def test_cli_plan_e2e():
    result = run_cli("plan", "Entrenar el modelo")
    assert result.returncode == 0
    assert "cargar_dataset" in result.stdout
    assert "entrenar" in result.stdout


def test_cli_benchmark_e2e():
    result = run_cli("benchmark")
    assert result.returncode == 0
    assert "CerebroV3 E2E" in result.stdout
    assert "Accuracy" in result.stdout


def test_module_entrypoint_version():
    result = subprocess.run(
        [sys.executable, "__main__.py", "--version"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0

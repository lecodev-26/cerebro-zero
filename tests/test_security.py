"""
Tests del sistema de seguridad
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from security.permissions import Permission, PermissionSet, is_dangerous, PERMISSIONS_SAFE
from security.sandbox import Sandbox, SandboxResult
from security.parser import Parser
from tools.registry import ToolRegistry


# ============================================
# TESTS DE PERMISOS
# ============================================

def test_permission_set_basic():
    ps = PermissionSet({Permission.EXECUTE_MATH})
    assert ps.has(Permission.EXECUTE_MATH)
    assert not ps.has(Permission.READ_NETWORK)


def test_permission_set_add_remove():
    ps = PermissionSet()
    ps.add(Permission.EXECUTE_MATH)
    assert ps.has(Permission.EXECUTE_MATH)
    ps.remove(Permission.EXECUTE_MATH)
    assert not ps.has(Permission.EXECUTE_MATH)


def test_dangerous_permissions():
    assert is_dangerous(Permission.EXECUTE_SUBPROCESS)
    assert is_dangerous(Permission.DELETE_FILES)
    assert not is_dangerous(Permission.EXECUTE_MATH)
    assert not is_dangerous(Permission.READ_MEMORY)


# ============================================
# TESTS DE SANDBOX
# ============================================

def test_sandbox_allows_safe():
    sandbox = Sandbox(strict=True)
    sandbox.permissions.add(Permission.EXECUTE_MATH)
    
    def suma(a, b):
        return a + b
    
    r = sandbox.execute(suma, Permission.EXECUTE_MATH, 5, 3)
    assert r.success
    assert r.result == 8


def test_sandbox_blocks_without_permission():
    sandbox = Sandbox(strict=True)
    
    def suma(a, b):
        return a + b
    
    r = sandbox.execute(suma, Permission.EXECUTE_MATH, 5, 3)
    assert not r.success
    assert r.blocked


def test_sandbox_blocks_dangerous_without_approval():
    sandbox = Sandbox(strict=True)
    sandbox.permissions.add(Permission.EXECUTE_SUBPROCESS)
    
    def peligroso(cmd):
        return f"Ejecutando {cmd}"
    
    r = sandbox.execute(peligroso, Permission.EXECUTE_SUBPROCESS, "ls")
    assert not r.success
    assert r.blocked


def test_sandbox_allows_dangerous_with_approval():
    sandbox = Sandbox(strict=True)
    sandbox.approve_dangerous(Permission.EXECUTE_SUBPROCESS)
    
    def peligroso(cmd):
        return f"Ejecutando {cmd}"
    
    r = sandbox.execute(peligroso, Permission.EXECUTE_SUBPROCESS, "ls")
    assert r.success


def test_sandbox_audit_log():
    sandbox = Sandbox()
    sandbox.permissions.add(Permission.EXECUTE_MATH)
    
    def suma(a, b):
        return a + b
    
    sandbox.execute(suma, Permission.EXECUTE_MATH, 1, 2)
    assert len(sandbox.audit_log) > 0


# ============================================
# TESTS DE PARSER
# ============================================

def test_parser_math():
    parser = Parser()
    p = parser.parse("¿Cuánto es 5 + 3?")
    assert p.intent == 'math'
    assert 5.0 in p.entities['numbers']
    assert 3.0 in p.entities['numbers']


def test_parser_dangerous_rm():
    parser = Parser()
    p = parser.parse("rm -rf /")
    assert p.dangerous
    assert 'borrar' in p.danger_reason.lower() or 'recursiv' in p.danger_reason.lower()


def test_parser_dangerous_eval():
    parser = Parser()
    p = parser.parse("eval('print(1)')")
    assert p.dangerous


def test_parser_dangerous_subprocess():
    parser = Parser()
    p = parser.parse("os.system('ls')")
    assert p.dangerous


def test_parser_safe_input():
    parser = Parser()
    p = parser.parse("hola, ¿cómo estás?")
    assert not p.dangerous


# ============================================
# TESTS DE TOOL REGISTRY
# ============================================

def test_registry_register_and_execute():
    sandbox = Sandbox()
    sandbox.permissions.add(Permission.EXECUTE_MATH)
    registry = ToolRegistry(sandbox=sandbox)
    
    @registry.register("suma", "Suma dos números", {Permission.EXECUTE_MATH})
    def suma(a, b):
        return a + b
    
    r = registry.execute('suma', 5, 3)
    assert r['success']
    assert r['result'] == 8


def test_registry_blocks_without_permission():
    sandbox = Sandbox()
    registry = ToolRegistry(sandbox=sandbox)
    
    @registry.register("suma", "Suma", {Permission.EXECUTE_MATH})
    def suma(a, b):
        return a + b
    
    r = registry.execute('suma', 5, 3)
    assert not r['success']


def test_registry_tool_not_found():
    registry = ToolRegistry()
    r = registry.execute('no_existe')
    assert not r['success']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

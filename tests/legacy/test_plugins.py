"""
Tests del sistema de plugins
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from tools.tool_base import Tool
from tools.loader import discover_plugins, list_plugins
from tools.registry_v2 import ToolRegistryV2
from security.sandbox import Sandbox
from security.permissions import Permission, PERMISSIONS_STANDARD


# ============================================
# TESTS DEL TOOL BASE
# ============================================

def test_tool_base_is_abstract():
    # Tool ahora usa atributos de clase, no constructor
    tool = Tool()
    assert tool.name == "tool"
    assert tool.description == "Herramienta genérica"
    assert tool.permissions == set()
    with pytest.raises(NotImplementedError):
        tool.can_handle("test")
    with pytest.raises(NotImplementedError):
        tool.execute("test")


# ============================================
# TESTS DEL LOADER
# ============================================

def test_discover_plugins():
    plugins = discover_plugins()
    assert len(plugins) >= 4, f"Esperados >=4 plugins, encontrados {len(plugins)}"
    assert 'calculadora' in plugins
    assert 'reloj' in plugins
    assert 'saludo' in plugins
    assert 'contador' in plugins


def test_list_plugins():
    names = list_plugins()
    assert 'calculadora' in names
    assert 'reloj' in names
    assert 'saludo' in names


# ============================================
# TESTS DEL PLUGIN CALCULADORA
# ============================================

def test_calculator_can_handle():
    plugins = discover_plugins()
    calc = plugins['calculadora']
    assert calc.can_handle("¿Cuánto es 5 + 3?")
    assert calc.can_handle("10 * 7")
    assert not calc.can_handle("hola")


def test_calculator_execute_sum():
    plugins = discover_plugins()
    calc = plugins['calculadora']
    result = calc.execute("5 + 3")
    assert result['success']
    assert result['value'] == 8


def test_calculator_execute_mul():
    plugins = discover_plugins()
    calc = plugins['calculadora']
    result = calc.execute("10 * 7")
    assert result['success']
    assert result['value'] == 70


def test_calculator_execute_div_zero():
    plugins = discover_plugins()
    calc = plugins['calculadora']
    result = calc.execute("10 / 0")
    assert not result['success']


# ============================================
# TESTS DEL PLUGIN RELOJ
# ============================================

def test_clock_can_handle():
    plugins = discover_plugins()
    clock = plugins['reloj']
    assert clock.can_handle("¿Qué hora es?")
    assert clock.can_handle("¿Qué fecha es hoy?")
    assert not clock.can_handle("5 + 3")


def test_clock_execute_hora():
    plugins = discover_plugins()
    clock = plugins['reloj']
    result = clock.execute("¿Qué hora es?")
    assert result['success']
    assert result['type'] == 'hora'
    assert ':' in result['result']


def test_clock_execute_fecha():
    plugins = discover_plugins()
    clock = plugins['reloj']
    result = clock.execute("¿Qué fecha es hoy?")
    assert result['success']
    assert result['type'] == 'fecha'


# ============================================
# TESTS DEL PLUGIN SALUDO
# ============================================

def test_greeting_can_handle():
    plugins = discover_plugins()
    greet = plugins['saludo']
    assert greet.can_handle("hola")
    assert greet.can_handle("adiós")
    assert not greet.can_handle("5 + 3")


def test_greeting_execute():
    plugins = discover_plugins()
    greet = plugins['saludo']
    result = greet.execute("hola")
    assert result['success']
    assert result['type'] == 'saludo'


# ============================================
# TESTS DEL REGISTRY V2
# ============================================

def test_registry_v2_loads_plugins():
    registry = ToolRegistryV2()
    assert len(registry.plugins) >= 4


def test_registry_v2_find_handler():
    registry = ToolRegistryV2()
    assert registry.find_handler("5 + 3") is not None
    assert registry.find_handler("hola") is not None
    assert registry.find_handler("xyz random") is None


def test_registry_v2_execute_by_input():
    sandbox = Sandbox(permissions=PERMISSIONS_STANDARD.copy())
    registry = ToolRegistryV2(sandbox=sandbox)
    
    result = registry.execute_by_input("5 + 3")
    assert result['success']


def test_registry_v2_blocks_without_permission():
    sandbox = Sandbox(permissions=set())  # Sin permisos
    registry = ToolRegistryV2(sandbox=sandbox)
    
    result = registry.execute_by_input("5 + 3")
    assert not result['success']
    assert result.get('blocked') == True


def test_registry_v2_list_tools():
    registry = ToolRegistryV2()
    tools = registry.list_tools()
    assert len(tools) >= 4
    names = [t['name'] for t in tools]
    assert 'calculadora' in names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

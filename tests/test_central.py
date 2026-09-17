"""
Tests del Cerebro Central
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from agent.central import CerebroCentral
from agent.pipeline import (
    Context,
    step_parse,
    step_sandbox_check,
)


# ============================================
# TESTS DEL CONTEXTO
# ============================================

def test_context_creation():
    ctx = Context(input="hola")
    assert ctx.input == "hola"
    assert ctx.parsed is None
    assert ctx.allowed == False


def test_context_to_dict():
    ctx = Context(input="test")
    d = ctx.to_dict()
    assert d['input'] == "test"
    assert 'intent' in d


# ============================================
# TESTS DE PASOS INDIVIDUALES
# ============================================

def test_step_parse():
    from security.parser import Parser
    parser = Parser()
    ctx = Context(input="¿Cuánto es 5 + 3?")
    ctx = step_parse(ctx, parser)
    assert ctx.parsed is not None
    assert ctx.parsed.intent == 'math'


def test_step_sandbox_allows_safe():
    from security.parser import Parser
    from security.sandbox import Sandbox
    
    parser = Parser()
    sandbox = Sandbox(strict=True)
    
    ctx = Context(input="hola")
    ctx = step_parse(ctx, parser)
    ctx = step_sandbox_check(ctx, sandbox)
    
    assert ctx.allowed


def test_step_sandbox_blocks_dangerous():
    from security.parser import Parser
    from security.sandbox import Sandbox
    
    parser = Parser()
    sandbox = Sandbox(strict=True)
    
    ctx = Context(input="rm -rf /")
    ctx = step_parse(ctx, parser)
    ctx = step_sandbox_check(ctx, sandbox)
    
    assert not ctx.allowed
    assert ctx.block_reason is not None


# ============================================
# TESTS DEL CEREBRO CENTRAL
# ============================================

def test_central_creation():
    cerebro = CerebroCentral()
    assert cerebro.name == "Cerebro Central"
    assert cerebro.version == "2.0.0"
    assert len(cerebro.registry.tools) >= 3  # calculadora, fecha, hora


def test_central_blocks_dangerous():
    cerebro = CerebroCentral()
    result = cerebro.procesar("rm -rf /")
    assert "🔒" in result
    assert "Bloqueado" in result or "seguridad" in result


def test_central_math():
    cerebro = CerebroCentral()
    result = cerebro.procesar("¿Cuánto es 5 + 3?")
    assert "8" in result


def test_central_multiplication():
    cerebro = CerebroCentral()
    result = cerebro.procesar("¿Cuánto es 10 * 7?")
    assert "70" in result


def test_central_time():
    cerebro = CerebroCentral()
    result = cerebro.procesar("¿Qué hora es?")
    # El resultado debe tener el formato HH:MM:SS o similar
    assert ":" in result or "🕐" in result


def test_central_teach_and_remember():
    cerebro = CerebroCentral()
    cerebro.enseñar("test_entrada", "test_respuesta")
    result = cerebro.procesar("test_entrada")
    assert "test_respuesta" in result


def test_central_default_response():
    cerebro = CerebroCentral()
    result = cerebro.procesar("xyz_entrada_que_no_existe")
    # No debería crashear
    assert isinstance(result, str)


def test_central_history():
    cerebro = CerebroCentral()
    cerebro.procesar("hola")
    cerebro.procesar("¿Cuánto es 2 + 2?")
    assert len(cerebro.history) == 2


def test_central_registry_has_tools():
    cerebro = CerebroCentral()
    tools = cerebro.registry.list_tools()
    tool_names = [t['name'] for t in tools]
    assert 'calculadora' in tool_names
    assert 'fecha' in tool_names
    assert 'hora' in tool_names


def test_central_pipeline_order():
    """El pipeline ejecuta pasos en orden"""
    cerebro = CerebroCentral()
    cerebro.procesar("hola")
    ctx = cerebro.history[-1]
    
    # Verificar que el parser se ejecutó
    assert ctx.parsed is not None
    # Verificar que el sandbox verificó
    assert ctx.allowed == True
    # Verificar que se intentó aprender
    assert ctx.learned == True or ctx.output is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Tests del Cerebro Central V3.0
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from agent.central import CerebroCentral
from agent.pipeline import Context, step_parse, step_sandbox_check


def test_context_creation():
    ctx = Context(input="hola")
    assert ctx.input == "hola"


def test_context_to_dict():
    ctx = Context(input="test")
    d = ctx.to_dict()
    assert d['input'] == "test"


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


def test_central_creation():
    cerebro = CerebroCentral()
    assert cerebro.version == "3.0.0"


def test_central_blocks_dangerous():
    cerebro = CerebroCentral()
    result = cerebro.procesar("rm -rf /")
    assert "🔒" in result


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
    import re
    assert re.search(r'\d{1,2}:\d{2}', result)


def test_central_teach_and_remember():
    cerebro = CerebroCentral()
    cerebro.procesar("recuerda color_test = azul")
    result = cerebro.procesar("recordar color_test")
    assert "azul" in result


def test_central_default_response():
    cerebro = CerebroCentral()
    result = cerebro.procesar("xyz_entrada_que_no_existe_123")
    assert isinstance(result, str)


def test_central_history():
    cerebro = CerebroCentral()
    cerebro.procesar("hola")
    cerebro.procesar("¿Cuánto es 2 + 2?")
    assert len(cerebro.history) == 2


def test_central_pipeline_order():
    cerebro = CerebroCentral()
    cerebro.procesar("hola")
    ctx = cerebro.history[-1]
    assert ctx.parsed is not None
    assert ctx.allowed == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

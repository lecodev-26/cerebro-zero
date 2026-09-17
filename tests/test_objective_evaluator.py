"""
Tests de autoevaluación objetiva
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from evaluation.objective_evaluator import (
    ObjectiveEvaluator, EvaluationResult, Evidence,
)


# ============================================
# FIXTURES
# ============================================

@pytest.fixture
def evaluator():
    return ObjectiveEvaluator()


# ============================================
# TESTS BÁSICOS
# ============================================

def test_evaluator_creation(evaluator):
    assert evaluator is not None


def test_empty_answer_fails(evaluator):
    result = evaluator.evaluate("hola", "")
    assert not result.verified
    assert result.confidence == 0.0
    assert len(result.errors) > 0


def test_non_empty_answer_passes_base(evaluator):
    result = evaluator.evaluate("hola", "Hola, ¿qué tal?")
    # Debe haber al menos una evidencia positiva
    assert any(e.passed for e in result.evidence)


# ============================================
# TESTS DE MATEMÁTICAS
# ============================================

def test_math_correct_addition(evaluator):
    result = evaluator.evaluate("¿Cuánto es 5 + 3?", "5 + 3 = 8", {'intent': 'math'})
    assert result.verified
    assert result.confidence > 0.8


def test_math_incorrect_addition(evaluator):
    result = evaluator.evaluate("¿Cuánto es 5 + 3?", "5 + 3 = 9", {'intent': 'math'})
    assert not result.verified
    assert any('incorrecto' in err.lower() for err in result.errors)


def test_math_correct_multiplication(evaluator):
    result = evaluator.evaluate("¿Cuánto es 10 * 7?", "10 * 7 = 70", {'intent': 'math'})
    assert result.verified


def test_math_correct_division(evaluator):
    result = evaluator.evaluate("¿Cuánto es 15 / 3?", "15 / 3 = 5", {'intent': 'math'})
    assert result.verified


def test_math_bad_format(evaluator):
    result = evaluator.evaluate("¿Cuánto es 5 + 3?", "ocho", {'intent': 'math'})
    assert not result.verified


def test_math_division_by_zero(evaluator):
    result = evaluator.evaluate("¿Cuánto es 5 / 0?", "5 / 0 = error", {'intent': 'math'})
    # No debe crashear
    assert isinstance(result, EvaluationResult)


# ============================================
# TESTS DE HORA
# ============================================

def test_time_correct(evaluator):
    result = evaluator.evaluate("¿Qué hora es?", "14:30:25", {'intent': 'time'})
    assert result.verified


def test_time_correct_short(evaluator):
    result = evaluator.evaluate("¿Qué hora es?", "14:30", {'intent': 'time'})
    assert result.verified


def test_time_incorrect(evaluator):
    result = evaluator.evaluate("¿Qué hora es?", "no sé", {'intent': 'time'})
    assert not result.verified


# ============================================
# TESTS DE FECHA
# ============================================

def test_date_correct(evaluator):
    result = evaluator.evaluate("¿Qué fecha es hoy?", "17/09/2026", {'intent': 'date'})
    assert result.verified


def test_date_incorrect(evaluator):
    result = evaluator.evaluate("¿Qué fecha es hoy?", "hoy es un buen día", {'intent': 'date'})
    assert not result.verified


# ============================================
# TESTS DE DEFINICIÓN
# ============================================

def test_definition_good(evaluator):
    result = evaluator.evaluate(
        "¿Qué es la IA?",
        "La inteligencia artificial es un campo de la computación",
        {'intent': 'definition'}
    )
    assert result.verified


def test_definition_excuse(evaluator):
    result = evaluator.evaluate("¿Qué es la IA?", "no sé", {'intent': 'definition'})
    assert not result.verified


def test_definition_too_short(evaluator):
    result = evaluator.evaluate("¿Qué es la IA?", "cosa", {'intent': 'definition'})
    assert not result.verified


# ============================================
# TESTS DE RESPUESTAS NEGATIVAS
# ============================================

def test_negative_answer_detected(evaluator):
    result = evaluator.evaluate("¿Cuánto es 5 + 3?", "No sé cuánto es", {'intent': 'math'})
    assert not result.verified
    assert any('negativa' in err.lower() for err in result.errors)


def test_negative_no_tengo(evaluator):
    result = evaluator.evaluate("test", "No tengo información sobre eso")
    assert not result.verified


# ============================================
# TESTS DE CONFIANZA
# ============================================

def test_confidence_high_for_verified_math(evaluator):
    result = evaluator.evaluate("5 + 3", "5 + 3 = 8", {'intent': 'math'})
    assert result.confidence > 0.7


def test_confidence_low_for_empty(evaluator):
    result = evaluator.evaluate("test", "")
    assert result.confidence == 0.0


def test_confidence_in_range(evaluator):
    result = evaluator.evaluate("test", "respuesta válida con contenido")
    assert 0.0 <= result.confidence <= 1.0


# ============================================
# TESTS DE ESTRUCTURA
# ============================================

def test_evidence_structure(evaluator):
    result = evaluator.evaluate("5 + 3", "5 + 3 = 8", {'intent': 'math'})
    assert len(result.evidence) > 0
    for e in result.evidence:
        assert isinstance(e, Evidence)
        assert e.source
        assert e.check
        assert isinstance(e.passed, bool)


def test_evaluation_result_to_dict(evaluator):
    result = evaluator.evaluate("5 + 3", "5 + 3 = 8", {'intent': 'math'})
    d = result.to_dict()
    assert 'confidence' in d
    assert 'verified' in d
    assert 'evidence' in d
    assert 'errors' in d


def test_evaluation_result_repr(evaluator):
    result = evaluator.evaluate("5 + 3", "5 + 3 = 8", {'intent': 'math'})
    repr_str = repr(result)
    assert 'EvaluationResult' in repr_str


# ============================================
# TESTS DE DETECCIÓN AUTOMÁTICA
# ============================================

def test_auto_detect_math(evaluator):
    result = evaluator.evaluate("¿Cuánto es 5 + 3?", "5 + 3 = 8")
    # Sin context, debe detectar automáticamente
    assert result.verified


def test_auto_detect_time(evaluator):
    result = evaluator.evaluate("¿Qué hora es?", "14:30")
    # Sin context, debe detectar automáticamente
    assert result.verified


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

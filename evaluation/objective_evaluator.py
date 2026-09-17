"""
Autoevaluación objetiva basada en evidencia
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Any, Callable


@dataclass
class Evidence:
    """Una pieza de evidencia"""
    source: str          # 'math', 'memory', 'tool', 'language'
    check: str           # descripción de la comprobación
    passed: bool         # si pasó
    detail: str = ""     # detalle adicional
    
    def __repr__(self):
        status = "✅" if self.passed else "❌"
        return f"{status} [{self.source}] {self.check}"


@dataclass
class EvaluationResult:
    """Resultado de la evaluación objetiva"""
    confidence: float                    # 0-1 basado en evidencia
    verified: bool                       # True si toda la evidencia pasa
    evidence: List[Evidence] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def __repr__(self):
        n_passed = sum(1 for e in self.evidence if e.passed)
        n_total = len(self.evidence)
        status = "✅" if self.verified else "❌"
        return f"{status} EvaluationResult(conf={self.confidence:.2f}, evidence={n_passed}/{n_total})"
    
    def to_dict(self) -> dict:
        return {
            'confidence': self.confidence,
            'verified': self.verified,
            'evidence': [
                {
                    'source': e.source,
                    'check': e.check,
                    'passed': e.passed,
                    'detail': e.detail,
                }
                for e in self.evidence
            ],
            'errors': self.errors,
        }


class ObjectiveEvaluator:
    """
    Evaluador objetivo que verifica respuestas con evidencia.
    
    Uso:
        evaluator = ObjectiveEvaluator()
        result = evaluator.evaluate(
            question="¿Cuánto es 5 + 3?",
            answer="5 + 3 = 8",
            context={'intent': 'math'}
        )
    """
    
    def __init__(self):
        self.math_pattern = re.compile(r'(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)')
        self.time_pattern = re.compile(r'^\d{1,2}:\d{2}(:\d{2})?$')
        self.date_pattern = re.compile(r'^\d{1,2}/\d{1,2}/\d{2,4}$')
    
    def evaluate(self, question: str, answer: str,
                 context: dict = None) -> EvaluationResult:
        """
        Evalúa una respuesta con evidencia objetiva.
        
        Args:
            question: pregunta original
            answer: respuesta a evaluar
            context: info adicional (intent, etc.)
        
        Returns:
            EvaluationResult con evidencia
        """
        context = context or {}
        evidence = []
        errors = []
        
        # 1. Verificación básica: respuesta vacía
        if not answer or len(answer.strip()) < 1:
            evidence.append(Evidence(
                source='base',
                check='Respuesta no vacía',
                passed=False,
                detail='Respuesta vacía'
            ))
            errors.append("Respuesta vacía")
            return EvaluationResult(
                confidence=0.0,
                verified=False,
                evidence=evidence,
                errors=errors,
            )
        
        evidence.append(Evidence(
            source='base',
            check='Respuesta no vacía',
            passed=True,
        ))
        
        # 2. Verificación según el tipo de pregunta
        intent = context.get('intent', self._detect_intent(question))
        
        if intent == 'math':
            self._verify_math(question, answer, evidence, errors)
        elif intent == 'time':
            self._verify_time(answer, evidence, errors)
        elif intent == 'date':
            self._verify_date(answer, evidence, errors)
        elif intent == 'definition':
            self._verify_definition(answer, evidence, errors)
        else:
            self._verify_generic(answer, evidence, errors)
        
        # 3. Verificación adicional: respuesta no contiene negaciones
        self._verify_no_negative(answer, evidence, errors)
        
        # 4. Calcular confianza
        confidence = self._compute_confidence(evidence)
        verified = all(e.passed for e in evidence)
        
        return EvaluationResult(
            confidence=confidence,
            verified=verified,
            evidence=evidence,
            errors=errors,
        )
    
    # ============================================
    # VERIFICADORES POR TIPO
    # ============================================
    
    def _detect_intent(self, question: str) -> str:
        """Detecta la intención de la pregunta"""
        q = question.lower()
        if re.search(r'\d+\s*[+\-*/]\s*\d+', question):
            return 'math'
        if 'hora' in q:
            return 'time'
        if 'fecha' in q or 'día' in q:
            return 'date'
        if 'qué es' in q or 'que es' in q or 'defin' in q:
            return 'definition'
        return 'generic'
    
    def _verify_math(self, question: str, answer: str,
                     evidence: List[Evidence], errors: List[str]):
        """Verifica una respuesta matemática recalculándola"""
        # Buscar patrón en la respuesta
        match = self.math_pattern.search(answer)
        
        if not match:
            evidence.append(Evidence(
                source='math',
                check='Formato de resultado matemático',
                passed=False,
                detail='No se encontró patrón "a op b = c"'
            ))
            errors.append("Formato matemático incorrecto")
            return
        
        a = float(match.group(1))
        op = match.group(2)
        b = float(match.group(3))
        result = float(match.group(4))
        
        # Recalcular
        if op == '+':
            expected = a + b
        elif op == '-':
            expected = a - b
        elif op == '*':
            expected = a * b
        elif op == '/':
            expected = a / b if b != 0 else None
        else:
            evidence.append(Evidence(
                source='math',
                check='Operador válido',
                passed=False,
                detail=f'Operador desconocido: {op}'
            ))
            errors.append(f"Operador desconocido: {op}")
            return
        
        if expected is None:
            evidence.append(Evidence(
                source='math',
                check='División válida',
                passed=False,
                detail='División por cero'
            ))
            errors.append("División por cero")
            return
        
        # Comparar con tolerancia
        tolerance = 1e-6
        diff = abs(expected - result)
        passed = diff < tolerance
        
        evidence.append(Evidence(
            source='math',
            check=f'Recalcular {a} {op} {b} = {expected}',
            passed=passed,
            detail=f"Resultado dado: {result}, esperado: {expected}"
        ))
        
        if not passed:
            errors.append(f"Cálculo incorrecto: {result} ≠ {expected}")
    
    def _verify_time(self, answer: str, evidence: List[Evidence], errors: List[str]):
        """Verifica que la respuesta tenga formato de hora"""
        answer_clean = answer.strip()
        # Buscar patrón HH:MM:SS o HH:MM dentro del texto
        if re.search(r'\d{1,2}:\d{2}(:\d{2})?', answer_clean):
            evidence.append(Evidence(
                source='time',
                check='Formato de hora válido',
                passed=True,
                detail=answer_clean
            ))
        else:
            evidence.append(Evidence(
                source='time',
                check='Formato de hora válido',
                passed=False,
                detail=f"No parece una hora: {answer_clean}"
            ))
            errors.append("Formato de hora incorrecto")
    
    def _verify_date(self, answer: str, evidence: List[Evidence], errors: List[str]):
        """Verifica que la respuesta tenga formato de fecha"""
        answer_clean = answer.strip()
        if re.search(r'\d{1,2}/\d{1,2}/\d{2,4}', answer_clean):
            evidence.append(Evidence(
                source='date',
                check='Formato de fecha válido',
                passed=True,
                detail=answer_clean
            ))
        else:
            evidence.append(Evidence(
                source='date',
                check='Formato de fecha válido',
                passed=False,
                detail=f"No parece una fecha: {answer_clean}"
            ))
            errors.append("Formato de fecha incorrecto")
    
    def _verify_definition(self, answer: str,
                           evidence: List[Evidence], errors: List[str]):
        """Verifica que una definición sea sustancial"""
        # Debe tener al menos 3 palabras
        palabras = answer.split()
        has_length = len(palabras) >= 3
        
        evidence.append(Evidence(
            source='definition',
            check='Definición sustancial',
            passed=has_length,
            detail=f"{len(palabras)} palabras"
        ))
        
        if not has_length:
            errors.append("Definición demasiado corta")
        
        # No debe ser una excusa
        excusas = ['no sé', 'no se', 'no tengo', 'no puedo', 'sin información']
        answer_lower = answer.lower()
        for excusa in excusas:
            if excusa in answer_lower:
                evidence.append(Evidence(
                    source='definition',
                    check='No es una excusa',
                    passed=False,
                    detail=f"Contiene '{excusa}'"
                ))
                errors.append(f"Excusa detectada: {excusa}")
                break
        else:
            evidence.append(Evidence(
                source='definition',
                check='No es una excusa',
                passed=True,
            ))
    
    def _verify_generic(self, answer: str,
                        evidence: List[Evidence], errors: List[str]):
        """Verificación genérica: respuesta con longitud mínima"""
        palabras = answer.split()
        has_length = len(palabras) >= 2
        
        evidence.append(Evidence(
            source='generic',
            check='Respuesta con contenido',
            passed=has_length,
            detail=f"{len(palabras)} palabras"
        ))
        
        if not has_length:
            errors.append("Respuesta demasiado corta")
    
    def _verify_no_negative(self, answer: str,
                            evidence: List[Evidence], errors: List[str]):
        """Verifica que no sea una respuesta negativa"""
        negativas = [
            'no sé', 'no se', 'no tengo', 'no entiendo',
            'no puedo', 'sin información', 'no hay',
        ]
        answer_lower = answer.lower()
        for neg in negativas:
            if neg in answer_lower:
                evidence.append(Evidence(
                    source='negative',
                    check='Respuesta afirmativa',
                    passed=False,
                    detail=f"Contiene '{neg}'"
                ))
                errors.append(f"Respuesta negativa: {neg}")
                return
        
        evidence.append(Evidence(
            source='negative',
            check='Respuesta afirmativa',
            passed=True,
        ))
    
    def _compute_confidence(self, evidence: List[Evidence]) -> float:
        """
        Calcula la confianza basada en la evidencia.
        
        Cada pieza de evidencia pesa según su fuente:
        - math: 1.0 (verificación directa)
        - tool: 0.9
        - memory: 0.8
        - definition: 0.7
        - generic: 0.5
        - negative: 0.3
        """
        weights = {
            'math': 1.0,
            'tool': 0.9,
            'memory': 0.8,
            'definition': 0.7,
            'time': 0.9,
            'date': 0.9,
            'generic': 0.5,
            'negative': 0.3,
            'base': 0.5,
        }
        
        if not evidence:
            return 0.0
        
        total_weight = 0.0
        passed_weight = 0.0
        
        for e in evidence:
            w = weights.get(e.source, 0.5)
            total_weight += w
            if e.passed:
                passed_weight += w
        
        if total_weight == 0:
            return 0.0
        
        return passed_weight / total_weight


if __name__ == "__main__":
    print("🧪 PROBANDO AUTOEVALUACIÓN OBJETIVA")
    print("="*60)
    
    evaluator = ObjectiveEvaluator()
    
    casos = [
        # (question, answer, context, esperado_verified)
        ("¿Cuánto es 5 + 3?", "5 + 3 = 8", {'intent': 'math'}, True),
        ("¿Cuánto es 5 + 3?", "5 + 3 = 9", {'intent': 'math'}, False),
        ("¿Cuánto es 10 * 7?", "10 * 7 = 70", {'intent': 'math'}, True),
        ("¿Qué hora es?", "14:30:25", {'intent': 'time'}, True),
        ("¿Qué hora es?", "no sé", {'intent': 'time'}, False),
        ("¿Qué fecha es hoy?", "17/09/2026", {'intent': 'date'}, True),
        ("¿Qué es la IA?", "La inteligencia artificial es un campo de la computación", {'intent': 'definition'}, True),
        ("¿Qué es la IA?", "no sé", {'intent': 'definition'}, False),
        ("hola", "Hola, ¿qué tal?", {'intent': 'generic'}, True),
        ("hola", "", {'intent': 'generic'}, False),
    ]
    
    passed = 0
    for question, answer, ctx, expected in casos:
        result = evaluator.evaluate(question, answer, ctx)
        status = "✅" if result.verified == expected else "❌"
        if result.verified == expected:
            passed += 1
        
        print(f"\n{status} Pregunta: {question}")
        print(f"   Respuesta: {answer}")
        print(f"   Verified: {result.verified} (esperado: {expected})")
        print(f"   Confianza: {result.confidence:.2f}")
        print(f"   Evidencia:")
        for e in result.evidence:
            print(f"      {e}")
        if result.errors:
            print(f"   Errores: {result.errors}")
    
    print(f"\n{'='*60}")
    print(f"📊 {passed}/{len(casos)} casos correctos")
    
    if passed == len(casos):
        print("🎉 ¡TODOS LOS CASOS PASARON!")
    else:
        print(f"❌ {len(casos) - passed} casos fallaron")
    
    print("\n✅ AUTOEVALUACIÓN OBJETIVA FUNCIONANDO")

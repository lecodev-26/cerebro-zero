"""
Sistema de autoevaluación
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
from datetime import datetime

class SelfEvaluation:
    def __init__(self):
        self.historial = []
        self.errores_detectados = []
        self.metricas = {
            'total_evaluaciones': 0,
            'respuestas_correctas': 0,
            'respuestas_incorrectas': 0,
            'tasa_confianza': 0.0
        }
    
    def evaluar(self, pregunta, respuesta):
        """
        Evalúa una respuesta antes de darla
        """
        evaluacion = {
            'pregunta': pregunta,
            'respuesta': respuesta,
            'fecha': datetime.now().isoformat(),
            'confianza': 0.0,
            'problemas': [],
            'sugerencias': []
        }
        
        # 1. Verificar que la respuesta no esté vacía
        if not respuesta or len(respuesta.strip()) < 2:
            evaluacion['problemas'].append("Respuesta vacía o muy corta")
            evaluacion['confianza'] = 0.1
            evaluacion['sugerencias'].append("Proporcionar una respuesta más elaborada")
        
        # 2. Verificar coherencia
        elif self._es_coherente(pregunta, respuesta):
            evaluacion['confianza'] = 0.9
        else:
            evaluacion['problemas'].append("Posible falta de coherencia")
            evaluacion['confianza'] = 0.5
            evaluacion['sugerencias'].append("Revisar la relación pregunta-respuesta")
        
        # 3. Verificar si es una respuesta por defecto
        if "No tengo información" in respuesta:
            evaluacion['problemas'].append("Respuesta por defecto (sin información)")
            evaluacion['confianza'] = 0.3
            evaluacion['sugerencias'].append("Buscar en memoria o usar herramientas")
        
        # 4. Verificar si es un error
        if "Error" in respuesta or "❌" in respuesta:
            evaluacion['problemas'].append("Respuesta con error")
            evaluacion['confianza'] = 0.0
            evaluacion['sugerencias'].append("Corregir el error")
        
        # Actualizar métricas
        self.metricas['total_evaluaciones'] += 1
        if evaluacion['confianza'] >= 0.7:
            self.metricas['respuestas_correctas'] += 1
        else:
            self.metricas['respuestas_incorrectas'] += 1
        
        self.metricas['tasa_confianza'] = (
            self.metricas['respuestas_correctas'] / self.metricas['total_evaluaciones']
            if self.metricas['total_evaluaciones'] > 0 else 0
        )
        
        # Guardar en historial
        self.historial.append(evaluacion)
        
        return evaluacion
    
    def _es_coherente(self, pregunta, respuesta):
        """
        Verifica si la respuesta es coherente con la pregunta
        """
        pregunta_lower = pregunta.lower()
        respuesta_lower = respuesta.lower()
        
        # Preguntas de hora
        if 'hora' in pregunta_lower and ('🕐' in respuesta or ':' in respuesta):
            return True
        
        # Preguntas de fecha
        if 'fecha' in pregunta_lower and ('📅' in respuesta or '/' in respuesta):
            return True
        
        # Operaciones matemáticas
        if any(op in pregunta for op in ['+', '-', '*', '/', 'suma', 'resta', 'multiplica', 'divide']):
            if '=' in respuesta and any(c.isdigit() for c in respuesta):
                return True
        
        # Saludos
        if any(s in pregunta_lower for s in ['hola', 'buenos', 'buenas']):
            if any(s in respuesta_lower for s in ['hola', 'buenos', 'buenas']):
                return True
        
        # Despedidas
        if any(s in pregunta_lower for s in ['adiós', 'adios', 'hasta']):
            if any(s in respuesta_lower for s in ['adiós', 'adios', 'hasta', 'luego']):
                return True
        
        # Por defecto, considerar coherente si tiene longitud razonable
        return len(respuesta) > 5
    
    def obtener_errores(self):
        """Obtiene las evaluaciones con problemas"""
        return [e for e in self.historial if e['confianza'] < 0.7]
    
    def estadisticas(self):
        """Muestra estadísticas de autoevaluación"""
        print("\n📊 ESTADÍSTICAS DE AUTOEVALUACIÓN")
        print("="*40)
        print(f"   Total evaluaciones: {self.metricas['total_evaluaciones']}")
        print(f"   Respuestas correctas: {self.metricas['respuestas_correctas']}")
        print(f"   Respuestas incorrectas: {self.metricas['respuestas_incorrectas']}")
        print(f"   Tasa de confianza: {self.metricas['tasa_confianza']*100:.2f}%")
        
        errores = self.obtener_errores()
        if errores:
            print(f"\n   Últimos problemas detectados:")
            for e in errores[-3:]:
                print(f"      - {e['pregunta']}: {', '.join(e['problemas'])}")
    
    def __repr__(self):
        return f"SelfEvaluation(total={self.metricas['total_evaluaciones']}, confianza={self.metricas['tasa_confianza']*100:.1f}%)"

def prueba_self_evaluation():
    print("🧠 PROBANDO AUTOEVALUACIÓN")
    print("="*40)
    
    evaluador = SelfEvaluation()
    
    # Casos de prueba
    casos = [
        ("¿Qué hora es?", "🕐 13:30:00"),
        ("¿Cuánto es 5 + 3?", "5 + 3 = 8"),
        ("hola", "Hola, soy Cerebro Zero"),
        ("¿Qué es la IA?", "No tengo información sobre eso"),
        ("capital de Francia", "Error: no encontrado"),
        ("", "Respuesta muy corta"),
        ("adiós", "Hasta luego, ha sido un placer"),
    ]
    
    print("\n📝 Evaluando respuestas:")
    for pregunta, respuesta in casos:
        evaluacion = evaluador.evaluar(pregunta, respuesta)
        estado = "✅" if evaluacion['confianza'] >= 0.7 else "⚠️"
        print(f"\n   {estado} Pregunta: {pregunta}")
        print(f"      Respuesta: {respuesta}")
        print(f"      Confianza: {evaluacion['confianza']:.2f}")
        if evaluacion['problemas']:
            print(f"      Problemas: {evaluacion['problemas']}")
    
    evaluador.estadisticas()
    
    print("\n✅ AUTOEVALUACIÓN FUNCIONANDO!")

if __name__ == "__main__":
    prueba_self_evaluation()

"""
Benchmark propio para Cerebro Zero
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from datetime import datetime

class Benchmark:
    def __init__(self):
        self.pruebas = {
            'memoria': self._test_memoria,
            'matematicas': self._test_matematicas,
            'lenguaje': self._test_lenguaje,
            'razonamiento': self._test_razonamiento,
            'herramientas': self._test_herramientas,
            'seguridad': self._test_seguridad,
            'aprendizaje': self._test_aprendizaje,
        }
        self.resultados = {}
    
    def ejecutar_todo(self):
        """Ejecuta todas las pruebas"""
        print("🧠 EJECUTANDO BENCHMARK COMPLETO")
        print("="*50)
        
        inicio = time.time()
        
        for nombre, test in self.pruebas.items():
            print(f"\n📝 Probando: {nombre.upper()}")
            print("-"*40)
            try:
                resultado = test()
                self.resultados[nombre] = resultado
                print(f"   ✅ Completado: {resultado['puntuacion']}/{resultado['total']} ({resultado['porcentaje']:.1f}%)")
            except Exception as e:
                print(f"   ❌ Error: {e}")
                self.resultados[nombre] = {'puntuacion': 0, 'total': 0, 'porcentaje': 0}
        
        tiempo_total = time.time() - inicio
        
        print("\n" + "="*50)
        print("📊 RESULTADOS FINALES")
        print("="*50)
        
        for nombre, resultado in self.resultados.items():
            barra = "█" * int(resultado['porcentaje'] / 10)
            print(f"   {nombre:15} {barra:10} {resultado['porcentaje']:5.1f}%")
        
        # Calcular puntuación total
        puntuacion_total = sum(r['puntuacion'] for r in self.resultados.values())
        total = sum(r['total'] for r in self.resultados.values())
        porcentaje_total = (puntuacion_total / total * 100) if total > 0 else 0
        
        print("\n" + "="*50)
        print(f"🎯 PUNTUACIÓN TOTAL: {puntuacion_total}/{total} ({porcentaje_total:.1f}%)")
        print(f"⏱️ Tiempo total: {tiempo_total:.2f}s")
        print("="*50)
        
        return self.resultados
    
    def _test_memoria(self):
        """Prueba de memoria"""
        import sys
        sys.path.insert(0, '..')
        from memory.retrieval import MemoryRetrieval
        
        memoria = MemoryRetrieval()
        pruebas = [
            ("proyecto", "Cerebro Zero"),
            ("lenguaje", "Python"),
            ("framework", "NumPy"),
        ]
        
        puntuacion = 0
        for clave, valor_esperado in pruebas:
            memoria.add_knowledge(clave, valor_esperado)
            resultados = memoria.remember(clave)
            if resultados.get('long_term'):
                puntuacion += 1
        
        return {'puntuacion': puntuacion, 'total': len(pruebas), 'porcentaje': puntuacion/len(pruebas)*100}
    
    def _test_matematicas(self):
        """Prueba de matemáticas"""
        from tools.tool_system import ToolSystem
        tools = ToolSystem()
        
        pruebas = [
            ("5 + 3", "8"),
            ("10 - 4", "6"),
            ("6 * 7", "42"),
            ("15 / 3", "5"),
        ]
        
        puntuacion = 0
        for expr, esperado in pruebas:
            resultado = tools.ejecutar('calculadora', expr)
            if esperado in resultado:
                puntuacion += 1
        
        return {'puntuacion': puntuacion, 'total': len(pruebas), 'porcentaje': puntuacion/len(pruebas)*100}
    
    def _test_lenguaje(self):
        """Prueba de lenguaje"""
        from language.tokenizer import Tokenizer
        
        tokenizer = Tokenizer()
        tokenizer.build_vocab(["hola mundo", "cerebro zero", "inteligencia artificial"], vocab_size=20)
        
        pruebas = [
            ("hola", 1),
            ("cerebro", 1),
            ("mundo", 1),
        ]
        
        puntuacion = 0
        for texto, min_tokens in pruebas:
            ids = tokenizer.encode(texto)
            if len(ids) >= min_tokens:
                puntuacion += 1
        
        return {'puntuacion': puntuacion, 'total': len(pruebas), 'porcentaje': puntuacion/len(pruebas)*100}
    
    def _test_razonamiento(self):
        """Prueba de razonamiento"""
        from reasoning.planner import Razonamiento
        razon = Razonamiento()
        
        pruebas = [
            "¿Cuánto es 5 + 3?",
            "¿Qué es la IA?",
            "¿Qué hora es?",
        ]
        
        puntuacion = 0
        for pregunta in pruebas:
            resultados = razon.razonar(pregunta)
            if resultados and resultados[0]:
                puntuacion += 1
        
        return {'puntuacion': puntuacion, 'total': len(pruebas), 'porcentaje': puntuacion/len(pruebas)*100}
    
    def _test_herramientas(self):
        """Prueba de herramientas"""
        from tools.tool_system import ToolSystem
        tools = ToolSystem()
        
        pruebas = [
            ('calculadora', '5 + 3'),
            ('fecha', None),
            ('hora', None),
            ('contar', 'hola mundo'),
        ]
        
        puntuacion = 0
        for comando, arg in pruebas:
            if arg:
                resultado = tools.ejecutar(comando, arg)
            else:
                resultado = tools.ejecutar(comando)
            if resultado and '❌' not in str(resultado):
                puntuacion += 1
        
        return {'puntuacion': puntuacion, 'total': len(pruebas), 'porcentaje': puntuacion/len(pruebas)*100}
    
    def _test_seguridad(self):
        """Prueba de seguridad"""
        from evaluation.security import SecuritySystem
        seguridad = SecuritySystem()
        
        pruebas = [
            ("hola", True),
            ("rm -rf /", False),
            ("eval('print(1)')", False),
            ("¿Qué hora es?", True),
        ]
        
        puntuacion = 0
        for entrada, esperado in pruebas:
            seguro, _ = seguridad.verificar(entrada)
            if seguro == esperado:
                puntuacion += 1
        
        return {'puntuacion': puntuacion, 'total': len(pruebas), 'porcentaje': puntuacion/len(pruebas)*100}
    
    def _test_aprendizaje(self):
        """Prueba de aprendizaje"""
        from training.continuous_learning import ContinuousLearning
        aprendizaje = ContinuousLearning(archivo="test_benchmark.json")
        
        # Aprender
        aprendizaje.aprender("test1", "respuesta1", correcto=True)
        aprendizaje.aprender("test2", "respuesta2", correcto=True)
        aprendizaje.aprender("test3", "respuesta3", correcto=False)
        
        # Verificar
        puntuacion = 0
        if len(aprendizaje.experiencias) == 3:
            puntuacion += 1
        if aprendizaje.metricas['total_experiencias'] == 3:
            puntuacion += 1
        if aprendizaje.metricas['errores'] == 1:
            puntuacion += 1
        
        return {'puntuacion': puntuacion, 'total': 3, 'porcentaje': puntuacion/3*100}

def prueba_benchmark():
    print("🧠 INICIANDO BENCHMARK")
    print("="*50)
    
    benchmark = Benchmark()
    resultados = benchmark.ejecutar_todo()
    
    print("\n✅ BENCHMARK COMPLETADO!")

if __name__ == "__main__":
    prueba_benchmark()

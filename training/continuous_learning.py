"""
Sistema de aprendizaje continuo
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pickle
from datetime import datetime

class ContinuousLearning:
    def __init__(self, archivo="aprendizaje_continuo.json"):
        self.archivo = archivo
        self.experiencias = []
        self.errores = []
        self.correcciones = []
        self.metricas = {
            'total_experiencias': 0,
            'aciertos': 0,
            'errores': 0,
            'tasa_acierto': 0.0
        }
        self.cargar()
    
    def aprender(self, entrada, salida, correcto=True, feedback=None):
        """
        Aprende de una experiencia
        """
        experiencia = {
            'entrada': entrada,
            'salida': salida,
            'correcto': correcto,
            'feedback': feedback,
            'fecha': datetime.now().isoformat()
        }
        self.experiencias.append(experiencia)
        
        # Actualizar métricas
        self.metricas['total_experiencias'] += 1
        if correcto:
            self.metricas['aciertos'] += 1
        else:
            self.metricas['errores'] += 1
            self.errores.append(experiencia)
        
        self.metricas['tasa_acierto'] = self.metricas['aciertos'] / self.metricas['total_experiencias']
        
        self.guardar()
        return experiencia
    
    def corregir(self, entrada_original, salida_correcta):
        """
        Corrige una respuesta incorrecta
        """
        correccion = {
            'entrada': entrada_original,
            'salida_correcta': salida_correcta,
            'fecha': datetime.now().isoformat()
        }
        self.correcciones.append(correccion)
        
        # Buscar la experiencia original y marcarla como corregida
        for exp in self.experiencias:
            if exp['entrada'] == entrada_original and not exp['correcto']:
                exp['correcto'] = True
                exp['corregido'] = True
                break
        
        self.guardar()
        print(f"🔧 Corregido: {entrada_original} → {salida_correcta}")
        return correccion
    
    def obtener_experiencias(self, limite=10):
        """Obtiene las últimas experiencias"""
        return self.experiencias[-limite:]
    
    def obtener_errores(self, limite=10):
        """Obtiene los últimos errores"""
        return self.errores[-limite:]
    
    def obtener_correcciones(self, limite=10):
        """Obtiene las últimas correcciones"""
        return self.correcciones[-limite:]
    
    def estadisticas(self):
        """Muestra estadísticas de aprendizaje"""
        print("\n📊 ESTADÍSTICAS DE APRENDIZAJE")
        print("="*40)
        print(f"   Total experiencias: {self.metricas['total_experiencias']}")
        print(f"   Aciertos: {self.metricas['aciertos']}")
        print(f"   Errores: {self.metricas['errores']}")
        print(f"   Tasa de acierto: {self.metricas['tasa_acierto']*100:.2f}%")
        print(f"   Correcciones: {len(self.correcciones)}")
    
    def guardar(self):
        """Guarda las experiencias en disco"""
        data = {
            'experiencias': self.experiencias,
            'errores': self.errores,
            'correcciones': self.correcciones,
            'metricas': self.metricas,
            'fecha_actualizacion': datetime.now().isoformat()
        }
        with open(self.archivo, 'w') as f:
            json.dump(data, f, indent=2)
    
    def cargar(self):
        """Carga las experiencias desde disco"""
        try:
            with open(self.archivo, 'r') as f:
                data = json.load(f)
            self.experiencias = data.get('experiencias', [])
            self.errores = data.get('errores', [])
            self.correcciones = data.get('correcciones', [])
            self.metricas = data.get('metricas', self.metricas)
        except:
            pass
    
    def __repr__(self):
        return f"ContinuousLearning(experiencias={len(self.experiencias)}, tasa={self.metricas['tasa_acierto']*100:.1f}%)"

def prueba_continuous_learning():
    print("🧠 PROBANDO APRENDIZAJE CONTINUO")
    print("="*40)
    
    aprendizaje = ContinuousLearning(archivo="test_aprendizaje.json")
    
    # Aprender experiencias
    print("\n📝 Aprendiendo...")
    aprendizaje.aprender("hola", "Hola, soy Cerebro Zero", correcto=True)
    aprendizaje.aprender("2+2", "4", correcto=True)
    aprendizaje.aprender("capital de Francia", "París", correcto=True)
    aprendizaje.aprender("capital de España", "Barcelona", correcto=False)  # Error intencional
    
    # Corregir error
    aprendizaje.corregir("capital de España", "Madrid")
    
    # Mostrar estadísticas
    aprendizaje.estadisticas()
    
    print("\n📋 Últimas experiencias:")
    for exp in aprendizaje.obtener_experiencias(3):
        print(f"   {exp['entrada']} → {exp['salida']} ({'✅' if exp['correcto'] else '❌'})")
    
    print("\n✅ APRENDIZAJE CONTINUO FUNCIONANDO!")

if __name__ == "__main__":
    prueba_continuous_learning()

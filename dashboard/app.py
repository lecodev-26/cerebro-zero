"""
Dashboard web para Cerebro Zero
Servidor Flask con chat interactivo y métricas en tiempo real.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify

# Importar componentes
from agent.central import CerebroCentral
from mobile.metrics import ResourceMonitor
from mobile.optimizer import MobileOptimizer


# ============================================
# ESTADO GLOBAL
# ============================================

class DashboardState:
    """Estado global del dashboard"""
    def __init__(self):
        self.agent = CerebroCentral(debug=False)
        self.monitor = ResourceMonitor()
        self.optimizer = MobileOptimizer()
        self.start_time = time.time()
        self.messages = []
        
        # Enseñar algunos conocimientos iniciales
        self.agent.enseñar("hola", "¡Hola! Soy Cerebro Zero 2.0. ¿En qué puedo ayudarte?")
        self.agent.enseñar("adiós", "¡Hasta luego! Ha sido un placer.")
        self.agent.enseñar("qué es la ia", "La inteligencia artificial es un campo de la computación que busca crear sistemas capaces de aprender.")
    
    def uptime(self) -> float:
        """Segundos desde que arrancó el dashboard"""
        return time.time() - self.start_time
    
    def get_stats(self) -> dict:
        """Obtiene estadísticas del agente"""
        return {
            'uptime_seconds': self.uptime(),
            'uptime_human': self._format_uptime(),
            'messages_count': len(self.messages),
            'memory': {
                'short_term': self.agent.memory.short_term.size(),
                'long_term': self.agent.memory.long_term.size(),
                'episodic': self.agent.memory.episodic.size(),
                'semantic': self.agent.memory.semantic.size(),
            },
            'tools': 4,  # calculadora, reloj, saludo, contador
            'tools_list': ['calculadora', 'reloj', 'saludo', 'contador'],
            'experiences': len(self.agent.learning.experiencias),
            'history_length': len(self.agent.history),
            'ram_mb': self.monitor.get_ram_usage_mb(),
        }
    
    def _format_uptime(self) -> str:
        seconds = int(self.uptime())
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        if h > 0:
            return f"{h}h {m}m {s}s"
        elif m > 0:
            return f"{m}m {s}s"
        else:
            return f"{s}s"
    
    def process(self, message: str) -> dict:
        """Procesa un mensaje del usuario"""
        # Guardar mensaje del usuario
        self.messages.append({
            'role': 'user',
            'content': message,
            'timestamp': datetime.now().isoformat(),
        })
        
        # Medir tiempo de respuesta
        with self.monitor.measure("process"):
            try:
                response = self.agent.procesar(message)
            except Exception as e:
                response = f"❌ Error: {e}"
        
        # Guardar respuesta
        self.messages.append({
            'role': 'assistant',
            'content': response,
            'timestamp': datetime.now().isoformat(),
        })
        
        return {
            'response': response,
            'stats': self.get_stats(),
        }


state = DashboardState()


# ============================================
# SERVIDOR FLASK
# ============================================

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['SECRET_KEY'] = 'cerebro-zero-2.0'


@app.route('/')
def index():
    """Página principal"""
    stats = state.get_stats()
    return render_template('index.html', stats=stats)


@app.route('/api/chat', methods=['POST'])
def api_chat():
    """API para enviar un mensaje al agente"""
    data = request.get_json()
    message = data.get('message', '').strip()
    
    if not message:
        return jsonify({'error': 'Mensaje vacío'}), 400
    
    result = state.process(message)
    return jsonify(result)


@app.route('/api/stats')
def api_stats():
    """API para obtener estadísticas del agente"""
    return jsonify(state.get_stats())


@app.route('/api/history')
def api_history():
    """API para obtener el historial de mensajes"""
    return jsonify({
        'messages': state.messages,
        'count': len(state.messages),
    })


@app.route('/api/reset', methods=['POST'])
def api_reset():
    """API para limpiar el historial"""
    state.messages.clear()
    return jsonify({'status': 'ok', 'message': 'Historial limpiado'})


@app.route('/api/monitor')
def api_monitor():
    """API para obtener métricas detalladas"""
    return jsonify({
        'stats': state.monitor.get_stats(),
        'ram_mb': state.monitor.get_ram_usage_mb(),
    })


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404


def run_dashboard(host='0.0.0.0', port=5000, debug=False):
    """Ejecuta el dashboard"""
    print(f"\n🚀 CEREBRO ZERO 2.0 DASHBOARD")
    print("="*60)
    print(f"🌐 Abre en tu navegador: http://localhost:{port}")
    print(f"📱 Desde otro dispositivo: http://TU_IP:{port}")
    print("="*60)
    print("Ctrl+C para detener")
    print("="*60 + "\n")
    
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_dashboard(debug=True)

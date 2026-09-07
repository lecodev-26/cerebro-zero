import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("🌐 INTERFAZ WEB PARA CEREBRO ZERO")
print("="*40)
print("⚠️ Para usar esta opción necesitas instalar Flask:")
print("   pip install flask")
print("")
print("O puedes usar la interfaz de chat mejorada (opción 2)")
print("   python experiments/chat_mejorado.py")
print("="*40)

# Código para interfaz web (requiere Flask)
def crear_interfaz_web():
    try:
        from flask import Flask, request, jsonify, render_template_string
        from experiments.chat_mejorado import ChatMejorado
        
        app = Flask(__name__)
        chat = ChatMejorado()
        
        HTML_TEMPLATE = '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Cerebro Zero - Chat</title>
            <style>
                body { font-family: Arial; max-width: 600px; margin: auto; padding: 20px; background: #1a1a2e; color: #eee; }
                .chat-box { border: 1px solid #444; height: 400px; overflow-y: auto; padding: 10px; background: #16213e; border-radius: 10px; }
                .user-msg { color: #4fc3f7; margin: 5px 0; }
                .bot-msg { color: #81c784; margin: 5px 0; }
                input { width: 80%; padding: 10px; margin: 10px 0; border-radius: 5px; border: none; }
                button { padding: 10px 20px; background: #4fc3f7; border: none; border-radius: 5px; color: #1a1a2e; cursor: pointer; }
                button:hover { background: #81d4fa; }
            </style>
        </head>
        <body>
            <h1>🧠 Cerebro Zero</h1>
            <div class="chat-box" id="chat">
                <div class="bot-msg">¡Hola! Soy Cerebro Zero. ¿Qué quieres saber?</div>
            </div>
            <input type="text" id="mensaje" placeholder="Escribe tu mensaje..." onkeypress="if(event.key==='Enter') enviar()">
            <button onclick="enviar()">Enviar</button>
            <script>
                function enviar() {
                    const input = document.getElementById('mensaje');
                    const chat = document.getElementById('chat');
                    const mensaje = input.value;
                    if (!mensaje) return;
                    
                    chat.innerHTML += `<div class="user-msg">🧑 Tú: ${mensaje}</div>`;
                    input.value = '';
                    
                    fetch('/chat', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({mensaje: mensaje})
                    })
                    .then(res => res.json())
                    .then(data => {
                        chat.innerHTML += `<div class="bot-msg">🤖 Cerebro: ${data.respuesta}</div>`;
                        chat.scrollTop = chat.scrollHeight;
                    });
                }
            </script>
        </body>
        </html>
        '''
        
        @app.route('/')
        def index():
            return render_template_string(HTML_TEMPLATE)
        
        @app.route('/chat', methods=['POST'])
        def chat_endpoint():
            data = request.get_json()
            mensaje = data.get('mensaje', '')
            respuesta = chat.procesar(mensaje)
            return jsonify({'respuesta': respuesta})
        
        print("\n🚀 Servidor web iniciado!")
        print("🌐 Abre en tu navegador: http://localhost:5000")
        print("📱 También puedes usar la IP de tu móvil en otro dispositivo")
        print("="*40)
        print("⚠️ Presiona CTRL+C para detener el servidor")
        
        app.run(host='0.0.0.0', port=5000, debug=False)
        
    except ImportError:
        print("❌ Flask no está instalado.")
        print("   Instálalo con: pip install flask")
        print("   Luego usa: python experiments/chat_mejorado.py")

if __name__ == "__main__":
    crear_interfaz_web()

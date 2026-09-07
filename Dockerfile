# Dockerfile para Cerebro Zero
# Ejecuta: docker build -t cerebro-zero . && docker run -it cerebro-zero

FROM python:3.14-slim

WORKDIR /app

# Instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY . .

# Puerto para la interfaz web
EXPOSE 5000

# Comando por defecto
CMD ["python", "todo_en_uno.py"]

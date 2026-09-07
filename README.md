# 🧠 CEREBRO ZERO

## Sistema de Inteligencia Artificial desde cero en Termux

[![Python](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-2.4.4-green.svg)](https://numpy.org/)
[![Termux](https://img.shields.io/badge/Termux-Android-orange.svg)](https://termux.com/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?logo=github)](https://github.com/lecodev-26/cerebro-zero)

---

## 📖 Descripción

**Cerebro Zero** es un sistema de inteligencia artificial completo construido **desde cero** en un móvil Android usando Termux. Sin frameworks, sin TensorFlow, sin PyTorch. Solo **Python y NumPy**.

El proyecto incluye:

- ✅ Redes neuronales desde cero (MLP, CNN)
- ✅ Autograd manual
- ✅ Optimizadores (SGD, Adam)
- ✅ Memoria persistente
- ✅ Reconocimiento de voz
- ✅ APIs de clima y noticias
- ✅ Interfaz web
- ✅ Chat interactivo

---

## 🚀 LOGROS

| Logro | Detalle |
|-------|---------|
| ✅ **12 opciones funcionales** | Menú unificado con diferentes cerebros |
| ✅ **Cerebro de 1024 neuronas** | Entrenado con MNIST |
| ✅ **78.95% de precisión** | Reconocimiento de dígitos |
| ✅ **814,090 parámetros** | Entrenados en un móvil |
| ✅ **Redes desde cero** | Sin frameworks externos |
| ✅ **APIs reales** | Clima + Noticias |

---

## 🛠️ TECNOLOGÍAS

| Tecnología | Versión |
|------------|---------|
| Python | 3.14 |
| NumPy | 2.4.4 |
| Flask | 3.1.3 |
| Termux | Última |
| Git | 2.55.0 |

---

## 📁 ESTRUCTURA DEL PROYECTO

```

cerebro-zero/
├── 📁 cerebros/              # 7 implementaciones de redes neuronales
├── 📁 core/                  # Núcleo: tensores, capas, activaciones, optimizadores
├── 📁 datos/                 # Generadores y gestores de datasets
├── 📁 datasets/              # Datasets (MNIST, etc.)
├── 📁 entrenadores/          # Lógica de entrenamiento
├── 📁 evaluation/            # Métricas y evaluación
├── 📁 experiments/           # Scripts de experimentos y pruebas
├── 📁 memory/                # Sistema de memoria persistente
├── 📁 modelos_guardados/     # Modelos entrenados (.pkl)
├── 📁 models/                # Modelos unificados (MLP, CNN, Transformer)
├── 📁 pruebas/               # Pruebas unitarias
├── 📁 scripts/               # Scripts ejecutables
├── 📁 tools/                 # Herramientas integradas (clima, noticias, voz...)
├── 📁 training/              # Lógica de entrenamiento avanzada
├── 📁 utils/                 # Utilidades (visualización, autoencoder...)
├── 📄 README.md              # Este archivo
├── 📄 todo_en_uno.py         # Menú principal
└── 📄 .gitignore             # Archivos ignorados

```

---

## 🧠 CEREBROS IMPLEMENTADOS

| # | Cerebro | Capas | Parámetros | Estado |
|---|---------|-------|------------|--------|
| 1 | **XOR (básico)** | [2, 8, 1] | 33 | ✅ |
| 2 | **SENO (polinómico)** | N/A | N/A | ✅ |
| 3 | **Profundo** | [2, 64, 32, 16, 1] | 2,817 | ⚠️ |
| 4 | **Memoria** | Variable | Variable | ✅ |
| 5 | **Premium** | [2, 16, 2] | 82 | ✅ |
| 6 | **Autoencoder** | PCA | N/A | ✅ |
| 7 | **Visión** | [9216, 32, 10] | 295,232 | ✅ |
| 8 | **Reforzamiento** | Q-Learning | N/A | ✅ |
| 9 | **Gigante** | [10, 64, 64, 1] | 4,929 | ✅ |
| 10 | **Máximo** | [10, 512, 1] | 5,632 | ✅ |
| 11 | **DIOS** | [10, 1000, 1] | 12,001 | ✅ |
| 12 | **MNIST** | [784, 1024, 10] | 814,090 | ✅ |

---

## 🛠️ HERRAMIENTAS INTEGRADAS

### Avanzadas
- Fecha y hora
- Búsqueda simulada
- Traducción
- Memorizar y recordar

### Extra
- Clima (simulado y real con OpenWeatherMap)
- Web simulada
- Cálculo de edad
- Conversor de moneda
- Dado y moneda
- Número aleatorio

### API (Reales)
- **Clima**: OpenWeatherMap (configurable)
- **Noticias**: NewsAPI (configurable)
- **Chistes**: Simulados
- **Citas**: Simuladas
- **Definiciones**: Diccionario interno

### Voz
- Reconocimiento de voz (Termux-API)
- Modo simulado para pruebas

---

## 🚀 INSTALACIÓN Y USO

### 1. Instalar Termux
```bash
pkg update && pkg upgrade -y
pkg install python python-pip git nano -y
pip install numpy flask
```

2. Clonar el repositorio

```bash
git clone https://github.com/lecodev-26/cerebro-zero.git
cd cerebro-zero
```

3. Ejecutar el menú principal

```bash
python todo_en_uno.py
```

4. Ejecutar el chat interactivo

```bash
python experiments/chat_ultimate.py
```

5. Ejecutar la interfaz web

```bash
python experiments/interfaz_web.py
```

Abrir en navegador: http://localhost:5000

---

🌤️ CONFIGURAR APIs REALES

Clima (OpenWeatherMap)

1. Regístrate en: https://openweathermap.org/api
2. Obtén tu API key
3. Edita tools/clima_real.py:

```python
self.api_key = "TU_API_KEY"
```

Noticias (NewsAPI)

1. Regístrate en: https://newsapi.org/register
2. Obtén tu API key
3. Edita tools/noticias_real.py:

```python
self.api_key = "TU_API_KEY"
```

---

📊 EJEMPLOS DE USO

Menú principal

```
🧠 CEREBRO ZERO - SISTEMA COMPLETO
====================================
1. XOR (básico) - ✅
2. SENO - ✅
3. Cargar cerebro guardado - ✅
4. Memoria - ✅
5. CEREBRO PREMIUM - ✅
...
```

Chat interactivo

```
🧑 Tú: clima Madrid
🧠 Cerebro Zero: 🌡️ Clima en Madrid: 28°C, ☀️ soleado

🧑 Tú: noticias tecnologia
🧠 Cerebro Zero: 📰 Internet 6G en desarrollo (TechNews)

🧑 Tú: memorizar color = azul
🧠 Cerebro Zero: 🧠 Memorizado: color = azul

🧑 Tú: recordar color
🧠 Cerebro Zero: azul
```

---

📤 SUBIR A GITHUB

```bash
git add .
git commit -m "🧠 CEREBRO ZERO - SISTEMA COMPLETO"
git push
```

---

📄 LICENCIA

Este proyecto es de código abierto. Puedes usarlo, modificarlo y distribuirlo libremente.

---

👨‍💻 AUTOR

Manuel (lecodev-26)

· GitHub: lecodev-26
· Email: axiomsystemsechepares@gmail.com

---

🙏 AGRADECIMIENTOS

· NumPy por la computación numérica
· Termux por hacer posible Python en Android
· OpenWeatherMap y NewsAPI por las APIs
· La comunidad de código abierto

---

🧠 ESTADO DEL PROYECTO

COMPLETO ✅

Fase Estado
Red neuronal desde cero ✅
MNIST ✅
Memoria persistente ✅
Autograd ✅
Transformer ✅
Lenguaje ✅
Herramientas ✅
APIs reales ✅
Voz ✅
Interfaz web ✅

---

⭐ ¡Si te gusta este proyecto, dale una estrella en GitHub! ⭐

---

Construido con 🧠 y ☕ desde un Samsung A16
EOF

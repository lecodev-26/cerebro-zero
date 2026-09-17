# 🧠 CEREBRO ZERO 3.0

## Autonomous Learning Cognitive Agent

[![Python](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-2.4.4-green.svg)](https://numpy.org/)
[![Tests](https://img.shields.io/badge/Tests-792%2F792-brightgreen.svg)](tests/)
[![CI](https://github.com/lecodev-26/cerebro-zero/actions/workflows/tests.yml/badge.svg)](https://github.com/lecodev-26/cerebro-zero/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📖 Descripción

**Cerebro Zero 3.0** es un **agente cognitivo integrado** construido **desde cero** en Python + NumPy. Sin frameworks de IA. Sin `pickle`. Todo verificado con gradient checking y tests.

**Filosofía 3.0 — Autonomous Learning:**
> 2.1 = sistema verificable.
> 3.0 = sistema que aprende, planifica y actúa de forma verificable.

---

## 🏆 Logros 3.0

| Área | Estado |
|------|--------|
| **Tests** | ✅ 792/792 pasan |
| **Autograd** | ✅ Verificado con gradient checking |
| **Transformer** | ✅ Backprop REAL (17/17 gradientes correctos) |
| **Pipeline 3.0** | ✅ CerebroV3 integrado (10 componentes) |
| **Planner 3.0** | ✅ Goal → Subgoals → Dependencies → Actions |
| **Learning Engine** | ✅ Accept/Reject con verificación objetiva |
| **RL básico** | ✅ Q-Learning + GridWorld (99% éxito) |
| **Tokenizer 3.0** | ✅ Byte-level + BPE desde cero |
| **Tool System 3.0** | ✅ Contratos + permisos + schemas |
| **Reproducibilidad** | ✅ Fingerprints de entorno + config |
| **Sandbox** | ✅ Subprocess aislado + timeout |
| **Persistencia** | ✅ Formato JSON+NPY (sin pickle) |
| **CI/CD** | ✅ 5 workflows en GitHub Actions |
| **CLI** | ✅ `cerebro-zero info/chat/plan/benchmark` |

---

## 🎯 Lo que funciona en 3.0

### **Motor matemático**
- Tensor con autograd completo
- Broadcasting correcto en backward
- Gradient checking exhaustivo
- Float32/float64 configurable

### **Transformer con backprop REAL**
- Multi-Head Attention con máscara causal
- LayerNorm con gradientes correctos
- Residuales y FFN
- **Verificado: 17/17 parámetros con gradient correcto**

### **Memorias cognitivas**
- **Working Memory**: capacidad limitada, TTL, prioridad, refuerzo
- **Procedural Memory**: aprender cómo hacer cosas
- **Semantic Memory**: búsqueda por similitud
- **World Model**: estado persistente del mundo con rutas anidadas

### **Planner 3.0**
- Descomposición: Goal → Subgoals → Dependencies → Actions
- Orden topológico (Kahn's algorithm)
- Detección de ciclos y dependencias rotas
- Plantillas de dominio: entrenar, evaluar, recordar
- Ejecución paso a paso

### **Learning Engine**
- Registro de experiencias (state, action, result, reward, error)
- Experience replay
- **Regla de oro:** el modelo NUNCA se actualiza sin mejora objetiva
- Accept/Reject con umbral configurable

### **RL básico**
- GridWorld NxN con obstáculos
- Q-Learning tabular con epsilon-greedy
- Entrenamiento y evaluación sin exploración
- ~99% de éxito tras entrenamiento

### **Tokenizer 3.0**
- **ByteTokenizer**: 256 bytes + 4 especiales, maneja cualquier texto
- **BPETokenizer**: Byte Pair Encoding desde cero
- Compresión real (hasta 6.5x en palabras vistas)
- Maneja emojis, acentos, chino, símbolos raros

### **Tool System 3.0**
- `ToolSpec`: contrato declarativo (input, output, permisos, capacidades)
- Validación de schemas (tipos, min/max, requerido, default)
- Búsqueda por capacidad y texto
- Historial y stats de ejecuciones

### **Benchmark 4.0**
- 7 categorías: math, memory, language, tools, security, planning, rl, learning
- Métricas científicas: accuracy, exact_match, error_rate, latency_p50/p95
- Splits: KNOWN, UNSEEN, ADVERSARIAL
- Reproducible con seed

### **Reproducibilidad**
- `hash_estructura`: determinista, orden ignorado
- `hash_archivo`: SHA256 chunked
- `EntornoCaptura`: Python, NumPy, plataforma, hardware, git commit
- `RegistroExperimento`: fingerprint único por experimento
- Verificación de reproducibilidad con tolerancia

### **Sandbox real**
- Subprocess aislado
- Timeout real (loop infinito se corta)
- Límites de recursos (SAFE/STANDARD/GENEROUS)
- Audit log persistente

### **Persistencia segura**
- Formato JSON + NPY (sin pickle)
- Checksums SHA256
- Detección de corrupción
- Inspeccionable con `cat`

---

## 🚀 Instalación

```bash
git clone https://github.com/lecodev-26/cerebro-zero.git
cd cerebro-zero
pip install -r requirements.txt
pytest tests/ -v
```

---

🖥️ CLI

```bash
# Ver versión
python cli.py --version

# Estado del sistema
python cli.py info

# Chat one-shot
python cli.py chat "5 + 3"

# Chat interactivo
python cli.py chat

# Generar plan
python cli.py plan "Entrenar el modelo"

# Ejecutar benchmark
python cli.py benchmark

# Ejecutar tests
python cli.py test
```

---

🧠 Ejemplo de uso

```python
from agent.cerebro_v3 import CerebroV3

cerebro = CerebroV3(nombre="Zero")

# Matemáticas (tool)
cerebro.procesar("5 + 3")           # → "5.0 + 3.0 = 8.0"

# Aprender / recordar (world model)
cerebro.procesar("aprende color = azul")
cerebro.procesar("recordar color")   # → "💭 Recuerdo: color = azul"

# Planificar (planner 3.0)
r = cerebro.procesar("Entrenar el modelo")
r.plan                                # → Plan con 8 subgoals

# Stats
cerebro.stats()
```

---

📁 Estructura

```
cerebro-zero/
├── agent/            # CerebroV3 (integración)
├── config/           # Configuración YAML
├── core/             # Tensor, autograd, capas, transformer
├── dashboard/        # Flask web
├── datasets/         # Datasets
├── docs/             # Documentación
├── evaluation/       # Benchmark, reproducibilidad
├── experiments/      # Scripts de prueba
├── language/         # Tokenizer 3.0, vocabulario
├── memory/           # Working, Procedural, World, Semantic
├── mobile/           # Optimizer, métricas, quantizer
├── models/           # Brain, MLP, Transformer
├── reasoning/        # Planner 3.0
├── security/         # Permissions, sandbox, process
├── tests/            # 792 tests
├── tools/            # Tool System 3.0
├── training/         # LM Trainer, Learning Engine, RL
├── cli.py            # CLI
└── pyproject.toml    # Packaging
```

---

📊 Benchmark 4.0

Ejecutar:

```bash
python cli.py benchmark
```

Métricas:

· Global accuracy
· Por categoría (math, memory, planning, rl, learning, ...)
· Por split (KNOWN, UNSEEN, ADVERSARIAL)
· Latencia p50, p95
· Error rate

---

🗺️ Roadmap

Ver ROADMAP.md para el detalle completo.

Completado

· ✅ Fases 1-20: Núcleo neuronal
· ✅ Fases 21-40: P0-P3 (corrección, cerebro central, inteligencia, ingeniería)
· ✅ Fases 2.1.1-2.1.8: Technical Integrity Release
· ✅ Fases 3.1-3.13: Autonomous Learning Cognitive Agent

Estado

· Total: 61/61 fases
· Tests: 792/792
· Versión: 3.0.0

---

🤝 Contribuir

Ver CONTRIBUTING.md.

---

📄 Licencia

MIT — ver LICENSE.

---

👨‍💻 Autor

Manuel (lecodev-26)

· GitHub: @lecodev-26
· Email: axiomsystemsechepares@gmail.com

---

Construido con 🧠 y ☕ desde un Samsung A16

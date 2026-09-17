# 🧠 CEREBRO ZERO 2.1

## Autonomous Learning Cognitive Agent

[![Python](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-2.4.4-green.svg)](https://numpy.org/)
[![Tests](https://img.shields.io/badge/Tests-308%2F308-brightgreen.svg)](tests/)
[![CI](https://github.com/lecodev-26/cerebro-zero/actions/workflows/tests.yml/badge.svg)](https://github.com/lecodev-26/cerebro-zero/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📖 Descripción

**Cerebro Zero 2.1** es un **agente híbrido neuronal** construido **desde cero** en Python + NumPy. Sin frameworks de IA. Sin `pickle`. Todo verificado con gradient checking.

**Filosofía 2.1 — Technical Integrity:**
> No más código. Mejor código. Que cada pieza sea demostrable.

---

## 🏆 Logros 2.1

| Área | Estado |
|------|--------|
| **Tests** | ✅ 308/308 pasan |
| **Autograd** | ✅ Verificado con gradient checking |
| **Transformer** | ✅ Backprop REAL (17/17 gradientes correctos) |
| **Pipeline** | ✅ 97.74% accuracy (benchmark honesto) |
| **Sandbox** | ✅ Subprocess aislado + timeout |
| **Persistencia** | ✅ Formato JSON+NPY (sin pickle) |
| **CI/CD** | ✅ 4 workflows en GitHub Actions |

---

## 🎯 Lo que funciona en 2.1

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

### **Benchmark honesto** (265 casos)
| Categoría | Accuracy |
|-----------|----------|
| **math** | 100.0% ✅ |
| **learning** | 100.0% ✅ |
| **memory** | 97.8% ✅ |
| **security** | 97.8% ✅ |
| **language** | 95.6% ✅ |
| **tools** | 95.6% ✅ |
| **TOTAL** | **97.74%** 🚀 |

**Splits:**
- KNOWN: 100.0%
- UNSEEN: 100.0%
- ADVERSARIAL: 76.0%

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

📁 Estructura

```
cerebro-zero/
├── core/           # Tensor, layers, serialization
├── models/         # Brain, MLP, Transformer
├── memory/         # Short/Long/Episodic/Semantic
├── reasoning/      # Planner, verifier
├── security/       # Permissions, sandbox, process, limits
├── tools/          # Plugins + registry
├── training/       # LM trainer, continuous learning
├── evaluation/     # Benchmark, objective evaluator
├── agent/          # Cerebro Central
├── dashboard/      # Flask web
├── mobile/         # Optimizer, metrics, quantizer
├── tests/          # 308 tests
└── docs/           # Documentación
```

---

🧠 Ejemplo de uso

```python
from agent.central import CerebroCentral

cerebro = CerebroCentral()

# Matemáticas
cerebro.procesar("¿Cuánto es 5 + 3?")  # → "5 + 3 = 8"

# Memoria
cerebro.procesar("recuerda color = azul")
cerebro.procesar("recordar color")     # → "💭 Recuerdo: color = azul"

# Aprendizaje
cerebro.procesar("aprende python")     # → "🎓 Aprendido: python"

# Seguridad
cerebro.procesar("rm -rf /")           # → "🔒 Bloqueado por seguridad"
```

---

🗺️ Roadmap completado

P0 — CORRECCIÓN ✅

· FASE 21: Autograd verificado
· FASE 22: Tests automáticos
· FASE 23: Broadcasting + numérica
· FASE 24: Seguridad real
· FASE 25: Documentación

P1 — CEREBRO CENTRAL ✅

· FASE 26-30: Brain, pipeline, memoria semántica, planner, plugins

P2 — INTELIGENCIA ✅

· FASE 31-35: Transformer, dataset, entrenamiento, aprendizaje continuo, autoevaluación

P3 — INGENIERÍA ✅

· FASE 36-40: Benchmark, config YAML, CI/CD, mobile, dashboard

2.1 — TECHNICAL INTEGRITY ✅

· FASE 2.1.1-8: Autograd 3.0, backprop real, LMTrainer limpio, benchmark honesto, pipeline arreglado, sandbox real, persistencia segura, documentación

---

👨‍💻 Autor

Manuel (lecodev-26)

· GitHub: @lecodev-26
· Email: axiomsystemsechepares@gmail.com

---

📄 Licencia

MIT — ver LICENSE

---

Construido con 🧠 y ☕ desde un Samsung A16

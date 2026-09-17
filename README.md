# 🧠 CEREBRO ZERO 2.0

## Sistema de Inteligencia Artificial desde cero en Termux

[![Python](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![Tests](https://github.com/lecodev-26/cerebro-zero/actions/workflows/tests.yml/badge.svg)](https://github.com/lecodev-26/cerebro-zero/actions/workflows/tests.yml)
[![Lint](https://github.com/lecodev-26/cerebro-zero/actions/workflows/lint.yml/badge.svg)](https://github.com/lecodev-26/cerebro-zero/actions/workflows/lint.yml)
[![Benchmark](https://github.com/lecodev-26/cerebro-zero/actions/workflows/benchmark.yml/badge.svg)](https://github.com/lecodev-26/cerebro-zero/actions/workflows/benchmark.yml)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-265%2F265-brightgreen.svg)](tests/)


[![Python](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-2.4.4-green.svg)](https://numpy.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1.3-red.svg)](https://flask.palletsprojects.com/)
[![Tests](https://img.shields.io/badge/Tests-69%2F69-brightgreen.svg)](tests/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📖 Descripción

**Cerebro Zero** es un **agente híbrido neuronal** construido **desde cero** en Python + NumPy. No usa frameworks como PyTorch o TensorFlow.

La versión **2.0** se centra en la **coherencia, corrección y verificabilidad** del sistema, no en añadir más código.

---

## 🎯 Filosofía 2.0

> **"No más código. Mejor código. Que cada pieza sea demostrable."**

- **Desde cero** — Sin frameworks de IA
- **Verificado** — Cada operación tiene tests
- **Modular** — Cada módulo tiene una responsabilidad
- **Seguro** — Sandbox + permisos
- **Coherente** — Documentación alineada con código

---

## 🏆 Estado Actual

| Área | Estado |
|------|--------|
| **Autograd** | ✅ Verificado (25/25 tests) |
| **Tests** | ✅ 69/69 pasando |
| **Broadcasting** | ✅ Completo |
| **Numérica** | ✅ Estable (float32/float64) |
| **Seguridad** | ✅ Sandbox + permisos |
| **CI/CD** | ✅ GitHub Actions |

---


## 🚦 Estado del CI/CD

| Workflow | Estado |
|----------|--------|
| Tests | [![Tests](https://github.com/lecodev-26/cerebro-zero/actions/workflows/tests.yml/badge.svg)](https://github.com/lecodev-26/cerebro-zero/actions/workflows/tests.yml) |
| Lint | [![Lint](https://github.com/lecodev-26/cerebro-zero/actions/workflows/lint.yml/badge.svg)](https://github.com/lecodev-26/cerebro-zero/actions/workflows/lint.yml) |
| Benchmark | [![Benchmark](https://github.com/lecodev-26/cerebro-zero/actions/workflows/benchmark.yml/badge.svg)](https://github.com/lecodev-26/cerebro-zero/actions/workflows/benchmark.yml) |
| Package | [![Package](https://github.com/lecodev-26/cerebro-zero/actions/workflows/package.yml/badge.svg)](https://github.com/lecodev-26/cerebro-zero/actions/workflows/package.yml) |

## 🗺️ Roadmap 2.0 — 20 fases

### 🔴 P0 — CORRECCIÓN ✅
- [x] **FASE 21:** Autograd verificado (25/25 tests)
- [x] **FASE 22:** Tests automáticos (28/28 tests)
- [x] **FASE 23:** Broadcasting + Numérica (53/53 tests)
- [x] **FASE 24:** Seguridad real (sandbox) (69/69 tests)
- [x] **FASE 25:** Documentación coherente ← *actual*

### 🟠 P1 — CEREBRO CENTRAL
- [ ] **FASE 26:** Interfaz unificada `Brain`
- [ ] **FASE 27:** Cerebro Central (pipeline limpio)
- [ ] **FASE 28:** Memoria semántica
- [ ] **FASE 29:** Planner + Verificador serios
- [ ] **FASE 30:** Tool Registry con plugins

### 🟡 P2 — INTELIGENCIA
- [ ] **FASE 31:** Transformer mejorado
- [ ] **FASE 32:** Dataset real
- [ ] **FASE 33:** Entrenamiento LM
- [ ] **FASE 34:** Aprendizaje continuo real
- [ ] **FASE 35:** Autoevaluación objetiva

### 🔵 P3 — INGENIERÍA
- [ ] **FASE 36:** Benchmark 2.0 (800 casos)
- [ ] **FASE 37:** Configuración YAML
- [ ] **FASE 38:** CI/CD real
- [ ] **FASE 39:** Cerebro Zero Mobile
- [ ] **FASE 40:** Dashboard web

**Progreso:** 🔄 **5/20 fases (25%)**

---

## 🚀 Instalación

```bash
# Clonar
git clone https://github.com/lecodev-26/cerebro-zero.git
cd cerebro-zero

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar tests
pytest tests/ -v
```

---

🧪 Tests

```bash
# Todos los tests
pytest tests/ -v

# Gradient checking (autograd)
python tests/test_gradient_check.py

# Seguridad
pytest tests/test_security.py -v
```

Estado: ✅ 69/69 tests pasan

---

📁 Estructura

```
cerebro-zero/
├── core/           # Tensor, capas, losses, optimizers
├── security/       # Permisos, sandbox, parser
├── tools/          # Herramientas + registry
├── memory/         # Memoria (corto/largo/episódico)
├── reasoning/      # Planner + verifier
├── training/       # Aprendizaje continuo
├── evaluation/     # Evaluación
├── agent/          # Agente completo
├── language/       # Tokenizador
├── models/         # Modelos (MLP, CNN, Transformer)
├── experiments/    # Experimentos
├── tests/          # 69 tests
└── docs/           # Documentación técnica
```

---

📚 Documentación

· ARCHITECTURE.md — Arquitectura completa
· TESTING.md — Guía de testing
· API.md — Referencia de API
· CHANGELOG.md — Historial de cambios
· ROADMAP.md — Roadmap completo

---

🧠 Ejemplo de uso

```python
from agent.cerebro_zero_v1 import CerebroZero

cerebro = CerebroZero()

# Enseñar
cerebro.enseñar("hola", "Hola, soy Cerebro Zero 2.0")

# Procesar
respuesta = cerebro.procesar("hola")
print(respuesta)  # "Hola, soy Cerebro Zero 2.0"

# Matemáticas (usa herramientas)
respuesta = cerebro.procesar("¿Cuánto es 5 + 3?")
print(respuesta)  # "5.0 + 3.0 = 8.0"

# Seguridad (bloquea entradas peligrosas)
respuesta = cerebro.procesar("rm -rf /")
print(respuesta)  # "🔒 Entrada bloqueada por seguridad"
```

---

👨‍💻 Autor

Manuel (lecodev-26)

· GitHub: @lecodev-26
· Email: axiomsystemsechepares@gmail.com

---

📄 Licencia

MIT — ver LICENSE

---

⭐ Contribuir

Ver CONTRIBUTING.md

---

Construido con 🧠 y ☕ desde un Samsung A16

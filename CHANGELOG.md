# 📝 CHANGELOG

Todos los cambios importantes en Cerebro Zero.

## [2.0.0-alpha] - En desarrollo

### 🔴 P0 — CORRECCIÓN

#### Añadido — FASE 25 (Documentación)
- **`docs/ARCHITECTURE.md`** — Arquitectura completa
- **`docs/TESTING.md`** — Guía de testing
- **`docs/API.md`** — Referencia de API
- README, ROADMAP y CHANGELOG alineados

#### Añadido — FASE 24 (Seguridad)
- **`security/permissions.py`** — 15 permisos atómicos
- **`security/sandbox.py`** — Sandbox con audit log
- **`security/parser.py`** — Detección de 15 patrones peligrosos
- **`tools/registry.py`** — Tool Registry con permisos
- **`tests/test_security.py`** — 15 tests

#### Corregido — FASE 23 (Broadcasting + Numérica)
- Broadcasting completo en backward
- Softmax, sigmoid, exp numéricamente estables
- `mean()` lanza `ValueError` en tensor vacío
- Soporte float32/float64 configurable
- **`tests/test_broadcasting.py`** — 10 tests
- **`tests/test_numerics.py`** — 14 tests

#### Corregido — FASE 22 (Tests automáticos)
- **`tests/test_layers.py`** — 7 tests
- **`tests/test_losses.py`** — 6 tests
- **`tests/test_optimizers.py`** — 4 tests
- **`tests/test_models.py`** — 3 tests
- **`pytest.ini`** configurado
- **`.github/workflows/tests.yml`** — CI/CD

#### Corregido — FASE 21 (Autograd)
- `log()` — dominio correcto (x > 0)
- `sqrt()` — dominio correcto (x ≥ 0)
- Broadcasting en backward
- `sum()` con axis y keepdims
- `mean()` con axis y keepdims
- **`tests/test_gradient_check.py`** — 25 tests

---

## [1.0.0] - 2026-09-08

### Añadido
- Redes neuronales desde cero (MLP, CNN, Transformer)
- 12 cerebros funcionales
- Cerebro de 1024 neuronas (MNIST)
- 78.95% de precisión
- Interfaz web con Flask
- Chat interactivo
- APIs reales (clima, noticias)
- Reconocimiento de voz

### Documentación inicial
- LICENSE (MIT)
- README
- CONTRIBUTING
- CODE_OF_CONDUCT
- SECURITY
- AUTHORS
- CITATION
- Makefile, setup.py, pyproject.toml, Dockerfile

---

**Formato basado en [Keep a Changelog](https://keepachangelog.com/)**

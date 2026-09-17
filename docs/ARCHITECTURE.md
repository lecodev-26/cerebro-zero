# 🏗️ Arquitectura de Cerebro Zero 2.0

## Visión General

Cerebro Zero 2.0 es un **agente híbrido neuronal** construido desde cero en Python + NumPy. No usa frameworks como PyTorch o TensorFlow.

## Pipeline de procesamiento

```

ENTRADA
↓
[ PARSER ] → detecta intención y peligros
↓
[ SANDBOX ] → verifica permisos
↓
[ TOOL REGISTRY ] → ejecuta o deriva
↓
[ MEMORIA ] → recupera contexto
↓
[ RAZONAMIENTO ] → planifica pasos
↓
[ VERIFICADOR ] → comprueba resultado
↓
[ APRENDIZAJE ] → guarda experiencia
↓
SALIDA

```

## Módulos principales

### `core/` — Núcleo matemático

- **`tensor.py`** — Tensor con autograd verificado (float32/float64)
- **`layers.py`** — Linear, Sequential, Flatten
- **`activations.py`** — ReLU, Sigmoid, Tanh, Softmax
- **`losses.py`** — MSE, BCE, CrossEntropy
- **`optimizers.py`** — SGD con momentum, Adam
- **`module.py`** — Clase base Module

### `security/` — Seguridad (FASE 24)

- **`permissions.py`** — 15 permisos atómicos
- **`sandbox.py`** — Sandbox con audit log
- **`parser.py`** — Detección de 15 patrones peligrosos

### `tools/` — Herramientas

- **`registry.py`** — Registro con permisos
- **`tool_system.py`** — Herramientas básicas (calculadora, fecha)
- **`herramientas_*.py`** — Herramientas avanzadas

### `memory/` — Sistema de memoria

- **`short_term.py`** — Memoria de trabajo (deque)
- **`long_term.py`** — Memoria persistente (JSON)
- **`episodic.py`** — Eventos con contexto
- **`retrieval.py`** — Recuperación unificada

### `reasoning/` — Razonamiento

- **`planner.py`** — Planificación paso a paso
- **`verifier.py`** — Verificación de resultados

### `training/` — Entrenamiento

- **`continuous_learning.py`** — Aprendizaje continuo

### `evaluation/` — Evaluación

- **`self_evaluation.py`** — Autoevaluación
- **`security.py`** — (legacy) verificación de entrada
- **`optimization.py`** — Optimización
- **`benchmark.py`** — Benchmark

### `agent/` — Agente completo

- **`brain.py`** — Agente v1
- **`brain_v2.py`** — Agente con aprendizaje
- **`cerebro_zero_v1.py`** — Agente final 1.0

## Flujo de datos

```

Usuario
↓
CerebroZero.procesar(input)
↓

1. Parser.parse(input) → ParsedCommand
   ↓
2. Sandbox: ¿permisos OK?
   ↓
3. ToolRegistry.execute() → resultado
   ↓
4. MemoryRetrieval.remember() → contexto
   ↓
5. Razonamiento.razonar() → plan
   ↓
6. SelfEvaluation.evaluar() → confianza
   ↓
7. ContinuousLearning.aprender() → experiencia
   ↓
   Respuesta

```

## Testing

```

tests/
├── test_tensor.py           # 8 tests
├── test_gradient_check.py   # 25 tests
├── test_layers.py           # 7 tests
├── test_losses.py           # 6 tests
├── test_optimizers.py       # 4 tests
├── test_models.py           # 3 tests
├── test_broadcasting.py     # 10 tests
├── test_numerics.py         # 14 tests
└── test_security.py         # 15 tests

```

**Total: 69 tests (100% passing)**

## Dependencias

- `numpy>=2.4.0`
- `flask>=3.0.0`
- `pytest>=9.0.0` (dev)

## Filosofía

1. **Desde cero** — Sin frameworks de IA
2. **Verificado** — Cada operación tiene tests
3. **Modular** — Cada módulo tiene una responsabilidad
4. **Seguro** — Sandbox + permisos
5. **Coherente** — Documentación alineada con código

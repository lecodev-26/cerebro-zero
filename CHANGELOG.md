# 📝 CHANGELOG

## [3.0.0] - 2026-09-17

### 🧠 AUTONOMOUS LEARNING COGNITIVE AGENT

#### FASE 3.13 — Documentación + Release 3.0
- README actualizado al estado real (792 tests)
- ROADMAP con las 13 fases del bloque 3.x
- CHANGELOG con todas las fases 3.x
- Documentación específica en docs/CEREBRO_V3.md

#### FASE 3.12 — Packaging
- CLI completa: `info`, `chat`, `plan`, `benchmark`, `test`
- `pyproject.toml` con entry point `cerebro-zero`
- `python cli.py --version` → Cerebro Zero 3.0.0
- `__main__.py` para `python -m cerebro_zero`

#### FASE 3.11 — Integración final
- `CerebroV3`: agente cognitivo unificado
- Pipeline completo: input → tokenizer → world → planner → tool → learning
- Respuesta tipada con traza (plan, tool, confianza, latencia)
- 5 tools básicas integradas (suma, resta, multiplica, recordar, aprender)

#### FASE 3.10 — Reproducibilidad
- `hash_estructura` determinista (orden de claves ignorado)
- `hash_archivo` SHA256 chunked
- `EntornoCaptura`: Python, NumPy, plataforma, hardware, git
- `RegistroExperimento` con fingerprint único
- Comparación y verificación de reproducibilidad con tolerancia

#### FASE 3.9 — Tool System 3.0
- `Schema` simplificado (tipos, min/max, requerido, default)
- `ToolSpec`: contrato declarativo
- `ToolRegistryV3`: búsqueda por capacidad y texto
- `ToolResult` tipado (ok/error/latencia)
- Validación de input y permisos con callback

#### FASE 3.8 — Tokenizer 3.0
- `ByteTokenizer`: 256 bytes + 4 especiales
- `BPETokenizer`: Byte Pair Encoding desde cero
- Entrenamiento sobre corpus con aprendizaje de fusiones
- Maneja cualquier texto: emojis, acentos, chino, zero-width
- Compresión real (hasta 6.5x en palabras vistas)

#### FASE 3.7 — Benchmark 4.0
- Categorías nuevas: planning, rl, learning
- Métricas científicas: accuracy, exact_match, error_rate, latency_p50/p95
- Splits KNOWN/UNSEEN/ADVERSARIAL
- Reproducibilidad con `random.Random` local

#### FASE 3.6 — RL básico
- `GridWorld` NxN con obstáculos, meta, timeout
- `QLearning` tabular con epsilon-greedy
- `EntrenadorRL` con evaluación sin exploración
- Aprendizaje real verificado (tasa éxito sube con entrenamiento)

#### FASE 3.5 — Learning Engine
- `Experiencia`: state, action, result, reward, error, verified
- `ExperienceBuffer` con límite y muestreo
- `EvaluadorMejora`: accept/reject con umbral
- Regla de oro: el modelo NUNCA se actualiza sin mejora objetiva

#### FASE 3.4 — Planner 3.0
- `Plan` con orden topológico (Kahn's algorithm)
- Detección de ciclos y dependencias rotas
- 3 plantillas: entrenar, evaluar, recordar
- Plan genérico para goals desconocidos
- Persistencia JSON de planes

#### FASE 3.3 — World Model
- `WorldModel` con rutas anidadas (`user.nombre`)
- set/get/update/delete/existe
- `incrementar`, `append`
- Historial de cambios (límite 200)
- Suscripciones con notificación a padres
- Persistencia JSON

#### FASE 3.2 — Procedural Memory
- `Procedimiento` con pasos (`Step`)
- Aprender de éxito y de instrucciones
- Búsqueda por trigger y tasa de éxito
- Ejecución con ejecutores externos

#### FASE 3.1 — Working Memory
- `WorkingItem` con prioridad, TTL, refuerzo
- Capacidad limitada (`max_items`)
- Expulsión por menor relevancia
- Relevancia calculada (prioridad + decay + refuerzo)

---

## [2.1.0] - 2026-09-17

### 🔴 TECHNICAL INTEGRITY RELEASE

#### FASE 2.1.8 — Documentación coherente
- README actualizado al estado real (308 tests)
- CHANGELOG con todas las fases 2.1
- Limpieza de archivos muertos

#### FASE 2.1.7 — Persistencia segura
- Formato JSON + NPY (persistencia nueva de 2.1; legacy conserva pickle)
- Checksums SHA256
- Detección de corrupción
- `safe_save_model` / `safe_load_model`

#### FASE 2.1.6 — Sandbox real
- Subprocess aislado con timeout
- Límites de recursos (SAFE/STANDARD/GENEROUS)
- Audit log persistente

#### FASE 2.1.5 — Pipeline arreglado
- Parser con 9 intenciones
- Steps: memory_store, memory_recall, learning
- Accuracy: 52.45% → 97.74%

#### FASE 2.1.4 — Benchmark 3.0 honesto
- 265 casos con splits KNOWN/UNSEEN/ADVERSARIAL
- Métricas p50/p95 de latencia

#### FASE 2.1.3 — LMTrainer limpio
- **BUGFIX CRÍTICO:** `__neg__` rompía el grafo de autograd
- Backprop REAL sin gradientes aleatorios
- Loss: 4.95 → 2.25 (-54.5%)

#### FASE 2.1.2 — Transformer con backprop real
- Multi-Head Attention con autograd completo
- LayerNorm, FFN, residuales
- **Gradient checking 17/17**

#### FASE 2.1.1 — Autograd 3.0
- `gather`, `embedding`, `batch_matmul`
- `stack`, `concat`, `where`, `mask`, `max`
- **Gradient checking 10/10**

---

## [2.0.0] - 2026-09-17

### 🟢 RELEASE 2.0

#### P3 — Ingeniería
- FASE 36: Benchmark 2.0
- FASE 37: Configuración YAML
- FASE 38: CI/CD real
- FASE 39: Cerebro Zero Mobile
- FASE 40: Dashboard Flask

#### P2 — Inteligencia
- FASE 31-35

#### P1 — Cerebro Central
- FASE 26-30

#### P0 — Corrección
- FASE 21-25

---

## [1.0.0] - 2026-09-08

### Añadido
- Redes neuronales desde cero
- 12 cerebros funcionales
- Interfaz web
- APIs reales

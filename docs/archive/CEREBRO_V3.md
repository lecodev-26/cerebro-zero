# 🧠 Cerebro Zero 3.0 — Arquitectura

## Autonomous Learning Cognitive Agent

Documentación técnica de la versión 3.0 de Cerebro Zero.

---

## 🎯 Filosofía

**2.0 → 3.0**

| Versión | Enfoque |
|---------|---------|
| 2.1 | Sistema verificable. Cada pieza demostrable. |
| **3.0** | **Sistema que aprende, planifica y actúa de forma verificable.** |

La 3.0 añade **cognición**: memoria de trabajo, memoria procedimental, world model, planner, learning engine y RL.

---

## 🏗️ Arquitectura

```

┌─────▼─────┐    ┌──────▼──────┐   ┌─────▼─────┐
│ PROCEDURAL│    │ WORLD MODEL │   │ SEMANTIC  │
└─────┬─────┘    └──────┬──────┘   └─────┬─────┘
└─────────────────┼─────────────────┘
↓
┌──────────────┐
│ PLANNER 3.0  │
└──────┬───────┘
↓
┌──────────────┐
│ TOOL SYSTEM  │
└──────┬───────┘
↓
┌──────────────┐
│  OBSERVER    │
└──────┬───────┘
↓
┌──────────────┐
│LEARNING ENG. │
└──────┬───────┘
↓
┌──────────────┐
│  RESPONSE    │
└──────────────┘

```

---

## 🧩 Componentes

### 3.1 — Working Memory (`memory/working_memory.py`)

Memoria de trabajo con capacidad limitada.

**Características:**
- `WorkingItem`: prioridad, TTL, veces accedido
- Expulsión automática por menor relevancia
- Relevancia = prioridad * 0.5 + decay * 0.3 + refuerzo * 0.2

**API:**
```python
wm = WorkingMemory(max_items=10, default_ttl=300.0)
wm.add(key="x", value=42, prioridad=0.8)
wm.get("x")
wm.get_top_relevantes(k=5)
```

---

3.2 — Procedural Memory (memory/procedural_memory.py)

Aprender cómo hacer cosas.

Características:

· Procedimiento con pasos (Step)
· Tasa de éxito calculada
· Búsqueda por trigger (regex) y tasa
· Ejecución con ejecutores externos

API:

```python
pm = ProceduralMemory(max_procedimientos=50)
pm.aprender(nombre="sumar_lista", trigger="sumar", pasos=[...])
pm.buscar("sumar lista")
pm.ejecutar(proc, contexto={"lista": [1,2,3]})
```

---

3.3 — World Model (memory/world_model.py)

Estado persistente del mundo con rutas anidadas.

Características:

· Rutas anidadas: world.set("user.nombre", "Manuel")
· Historial de cambios (límite 200)
· Suscripciones con notificación a padres
· Persistencia JSON

API:

```python
world = WorldModel()
world.set("user.nombre", "Manuel", razon="init")
world.get("user.nombre")             # → "Manuel"
world.incrementar("contador", 5)
world.append("logs", "mensaje")
world.suscribir("user", callback)
```

---

3.4 — Planner 3.0 (reasoning/planner_v3.py)

Descomposición jerárquica de objetivos.

Características:

· Plan con Subgoal y Action
· Orden topológico real (Kahn's algorithm)
· Detección de ciclos y dependencias rotas
· Plantillas de dominio: entrenar, evaluar, recordar
· Plan genérico para goals desconocidos
· Persistencia JSON

API:

```python
planner = Planner3()
plan = planner.planificar("Entrenar el modelo")
plan.orden_topologico()              # → ["cargar_dataset", "entrenar", ...]
plan.siguiente()                     # → próximo subgoal ejecutable
planner.ejecutar_plan(plan)
```

---

3.5 — Learning Engine (training/learning_engine.py)

Aprendizaje continuo con verificación.

Características:

· Experiencia: estado, acción, resultado, recompensa, error, verificado
· ExperienceBuffer: límite y muestreo
· EvaluadorMejora: accept/reject con umbral
· Regla de oro: el modelo NUNCA se actualiza sin mejora objetiva

API:

```python
engine = LearningEngine(umbral_mejora=0.0)
engine.registrar_exito("estado", "acción", "resultado")
engine.intentar_actualizar(score=0.9)  # → acepta si mejora
```

---

3.6 — RL básico (training/reinforcement.py)

Q-Learning + GridWorld.

Características:

· GridWorld NxN con obstáculos, meta, timeout
· QLearning tabular con epsilon-greedy
· EntrenadorRL con evaluación sin exploración
· ~99% de éxito tras entrenamiento

API:

```python
env = GridWorld(filas=5, columnas=5)
agente = QLearning(num_estados=25, num_acciones=4)
entrenador = EntrenadorRL(env, agente, epsilon_decay=0.99)
entrenador.entrenar(num_episodios=200)
entrenador.evaluar(num_episodios=20)  # → 100% éxito
```

---

3.7 — Benchmark 4.0 (evaluation/benchmark_v4.py)

Métricas científicas con splits.

Características:

· Categorías: math, memory, planning, rl, learning
· Métricas: accuracy, exact_match, error_rate, latency_p50/p95
· Splits: KNOWN, UNSEEN, ADVERSARIAL
· Reproducible con seed
· 145 casos generados automáticamente

API:

```python
bench = Benchmark(seed=42)
bench.registrar_handler("math", handler_math)
bench.cargar_casos()
reporte = bench.ejecutar()
reporte.por_categoria()              # → {"math": Metricas(...), ...}
reporte.por_split()                  # → {"KNOWN": Metricas(...), ...}
```

---

3.8 — Tokenizer 3.0 (language/tokenizer_v3.py)

Byte-level + BPE desde cero.

Características:

· ByteTokenizer: 256 bytes + 4 especiales (PAD/BOS/EOS/UNK)
· BPETokenizer: Byte Pair Encoding entrenado desde corpus
· Compresión real (hasta 6.5x en palabras vistas)
· Maneja emojis, acentos, chino, zero-width

API:

```python
bpe = BPETokenizer(vocab_size=300)
bpe.entrenar(corpus)
bpe.encode("programación")            # → [289, 110]
bpe.decode([289, 110])                # → "programación"
bpe.compresion_media(corpus)          # → 3.12x
```

---

3.9 — Tool System 3.0 (tools/tool_system_v3.py)

Contratos + permisos + schemas.

Características:

· Schema: tipos, min/max, requerido, default
· ToolSpec: contrato declarativo (input, output, permisos, capacidades)
· ToolRegistryV3: búsqueda por capacidad y texto
· ToolResult tipado (ok/error/latencia)

API:

```python
reg = ToolRegistryV3()
reg.registrar(ToolSpec(
    nombre="suma",
    descripcion="Suma dos números",
    funcion=suma,
    input_schema={"a": Schema("float"), "b": Schema("float")},
    capacidades=["matemáticas"],
))
r = reg.ejecutar("suma", {"a": 5, "b": 3})
# → ToolResult(ok=True, output=8)
```

---

3.10 — Reproducibilidad (evaluation/reproducibility.py)

Snapshots verificables de experimentos.

Características:

· hash_estructura: determinista, orden ignorado
· hash_archivo: SHA256 chunked
· EntornoCaptura: Python, NumPy, plataforma, hardware, git commit
· RegistroExperimento: fingerprint único
· Comparación y verificación con tolerancia

API:

```python
repro = Reproducibilidad()
reg = repro.nuevo_experimento("exp1", config={"lr": 0.01}, seed=42)
reg.resultado = {"accuracy": 0.95}
repro.guardar(reg)
repro.verificar_reproducible(reg1, reg2)
```

---

3.11 — Integración final (agent/cerebro_v3.py)

Agente cognitivo unificado.

Pipeline:

```
input → tokenizer → world → planner → tool → observer → learning → response
```

API:

```python
cerebro = CerebroV3(nombre="Zero")
r = cerebro.procesar("5 + 3")          # → Respuesta con output, tool, confianza
r = cerebro.procesar("Entrenar")       # → Respuesta con plan
cerebro.planificar("Evaluar")
cerebro.aprender_procedimiento("p1", ["paso1", "paso2"])
cerebro.stats()                        # → estado completo
```

---

3.12 — Packaging (cli.py, pyproject.toml)

CLI completa e instalable.

API:

```bash
cerebro-zero --version
cerebro-zero info
cerebro-zero chat "5 + 3"
cerebro-zero plan "Entrenar"
cerebro-zero benchmark
cerebro-zero test
```

---

3.13 — Documentación + Release

Esta misma documentación + README + CHANGELOG + ROADMAP.

---

📊 Estado final

Métrica Valor
Tests 792/792 ✅
Fases 3.x 13/13 ✅
Fases totales 61/61 ✅
Versión 3.0.0
Tooling CLI + pip installable

---

🚀 Próximos pasos (4.0)

Posibles líneas para 4.0:

· Autograd 4.0: einsum, más operaciones
· Transformer 4.0: KV cache, RMSNorm, beam search
· Continual Learning real: EWC, distillation
· Vision 3.0: CNN, datasets reales (MNIST, CIFAR)
· Voice 3.0: TTS, STT
· Multimodal: texto + imagen

---

Construido con 🧠 y ☕ desde un Samsung A16

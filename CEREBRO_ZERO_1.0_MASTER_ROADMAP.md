# Cerebro Zero 1.0 — Roadmap Maestro Definitivo
## Documento rector de arquitectura, producto, IA, entrenamiento y lanzamiento

**Estado:** planificación maestra  
**Objetivo:** Cerebro Zero 1.0.0  
**Regla:** V2–V5 son etapas históricas; 1.0 será la primera arquitectura pública unificada.

## 1. Misión

Cerebro Zero será un sistema de IA completo: runtime cognitivo + memoria + herramientas + seguridad + evaluación + entrenamiento + **su propio modelo neuronal Cerebro**.

Los modelos externos no serán la identidad final del producto. Se utilizarán como profesores, fuentes de comparación, generadores de datos sintéticos, evaluadores o proveedores auxiliares durante el desarrollo.

La meta de 1.0 es publicar una primera generación de **Cerebro Model 1.0**, entrenada de forma reproducible e integrada directamente en el runtime.

La promesa pública mínima será:

```bash
pip install cerebro-zero
cerebro
```

Y la API Python estable:

```python
from cerebro_zero import Cerebro
ai = Cerebro()
result = ai.run("Hola")
print(result.text)
```

## 2. Principios

1. Una arquitectura, un namespace público.
2. V2–V5 no deben dictar la arquitectura futura.
3. La GPU será opcional, nunca un requisito.
4. Los modelos serán intercambiables.
5. La memoria será persistente y controlable.
6. Las herramientas estarán gobernadas por políticas.
7. Todo comportamiento importante será observable.
8. El entrenamiento será reproducible.
9. Las APIs públicas serán estables.
10. Seguridad y límites forman parte del runtime, no son un añadido.## 3. Resultado final de 1.0

La versión 1.0 deberá poder:

- instalarse limpiamente desde PyPI;
- arrancar mediante CLI;
- usarse mediante Python;
- utilizar un modelo local;
- utilizar un proveedor remoto;
- usar OpenRouter como proveedor opcional;
- usar modelos gratuitos cuando estén disponibles;
- mantener memoria persistente;
- recuperar contexto relevante;
- ejecutar herramientas con permisos;
- pedir confirmación para acciones sensibles;
- registrar acciones y resultados;
- evaluar respuestas;
- ejecutar benchmarks reproducibles;
- ejecutar **Cerebro Model 1.0 propio**;
- entrenar y evaluar la línea de modelos Cerebro de forma reproducible;
- utilizar modelos externos como teachers cuando proceda;
- entrenar/adaptar modelos pequeños durante desarrollo y evolución;
- delegar trabajos pesados a compute externo;
- producir artefactos reproducibles;
- funcionar en CPU/Termux para capacidades ligeras;
- funcionar con GPU cuando exista;
- exponer API HTTP opcional;
- diagnosticar instalaciones mediante `cerebro doctor`.

## 4. Arquitectura de alto nivel

```
CLI / Python SDK / REST
          |
          v
   Cerebro Orchestrator
          |
          +--> Context Engine
          +--> Memory Engine
          +--> Model Engine
          +--> Reasoning Engine
          +--> Planner
          +--> Tool Engine
          +--> Security/Policy
          +--> Evaluation
          +--> Learning
          +--> Observability
          |
          v
 Local / OpenRouter / Other API / Own Model
```

La orquestación debe ser independiente del proveedor de modelo.## 5. Namespace definitivo

Destino:

```
cerebro_zero/
  __init__.py
  api.py
  cli/
  config/
  core/
  models/
    providers/
    local/
    tokenization/
    artifacts/
  cognition/
    runtime.py
    context.py
    reasoning/
    planning/
    skills.py
    learning.py
  memory/
    working.py
    episodic.py
    semantic.py
    procedural.py
    retrieval.py
    consolidation.py
  tools/
    registry.py
    permissions.py
    execution.py
  security/
    keys.py
    policy.py
    audit.py
    secrets.py
  evaluation/
  training/
  compute/
    local.py
    github.py
    gpu.py
    external.py
  observability/
  server/
```

La migración será progresiva. No se eliminarán módulos V5 hasta que exista equivalencia funcional y pruebas de instalación limpia.

## 6. Fases 1–10: fundación

### Fase 1 — Namespace
- Crear `cerebro_zero/`.
- Definir exports públicos.
- Definir versionado de API.
- Añadir pruebas de importación.

### Fase 2 — Facade Cerebro
- Crear `Cerebro`.
- Definir `run()`, `chat()`, `close()`.
- Crear resultado estable.
- Ocultar detalles internos.

### Fase 3 — Configuración
- Config central.
- Variables de entorno.
- Archivo de configuración.
- Perfiles local/remote/training.

### Fase 4 — Core contracts
- ModelProvider.
- MemoryProvider.
- ToolProvider.
- Evaluator.
- ComputeBackend.

### Fase 5 — Dependency boundaries
- Evitar imports circulares.
- Separar interfaces de implementaciones.
- Definir errores públicos.

### Fase 6 — Compatibility layer
- Adaptar V5.
- Mantener tests existentes.
- Añadir tests sobre la fachada 1.0.

### Fase 7 — Logging
- Logger estructurado.
- IDs de ejecución.
- Niveles configurables.

### Fase 8 — Persistence
- Directorios de datos.
- SQLite como base local inicial.
- Política de artefactos.

### Fase 9 — Doctor
- `cerebro doctor`.
- Python/dependencias.
- Memoria.
- Modelos.
- credenciales.
- compute.

### Fase 10 — Foundation gate
- Imports.
- configuración.
- persistencia.
- tests.
- documentación mínima.## 7. Fases 11–20: runtime cognitivo

### Fase 11 — Observe
Capturar entrada, contexto, estado y eventos.

### Fase 12 — Understand
Normalizar la petición y construir una representación interna.

### Fase 13 — Retrieve
Consultar memoria y contexto externo permitido.

### Fase 14 — Reason
Generar hipótesis, evidencias y alternativas.

### Fase 15 — Plan
Convertir intención en pasos verificables.

### Fase 16 — Guard
Evaluar riesgo, permisos, límites y necesidad de confirmación.

### Fase 17 — Act
Ejecutar herramientas o generar una respuesta.

### Fase 18 — Verify
Comprobar resultados, errores y condiciones esperadas.

### Fase 19 — Learn
Registrar experiencia, recompensa/evaluación y fallos.

### Fase 20 — Consolidate
Actualizar memoria y conocimiento sin crecimiento incontrolado.

## 8. Runtime control

Cada ciclo tendrá:

- execution_id;
- parent_execution_id;
- cycle_id;
- deadline;
- token budget;
- action budget;
- tool budget;
- memory budget;
- retry budget;
- risk level;
- cancellation;
- trace.

El runtime nunca dependerá de un bucle infinito.

## 9. Reparación

El planificador podrá:

1. detectar fallo;
2. clasificarlo;
3. recuperar evidencia;
4. modificar el plan;
5. reintentar dentro del presupuesto;
6. detenerse si supera límites.

Los fallos serán datos de evaluación, no excepciones silenciosas.## 10. Modelo Engine

El motor de modelos será una capa de abstracción:

```text
ModelProvider
 ├── LocalProvider
 ├── OpenRouterProvider
 ├── CompatibleAPIProvider
 └── CerebroModelProvider
```

Interfaz conceptual:

```python
response = provider.generate(
    messages=messages,
    tools=tools,
    temperature=...,
    max_tokens=...,
)
```

El runtime no conocerá detalles de cada proveedor.

## 11. OpenRouter

OpenRouter encaja como proveedor remoto opcional. Su API ofrece un endpoint compatible con chat completions y un catálogo consultable de modelos. citeturn0search2turn0search9

También existe un router `openrouter/free` que selecciona modelos gratuitos disponibles. La disponibilidad y los límites de esos modelos pueden cambiar, por lo que 1.0 no dependerá de un modelo gratuito concreto. citeturn0search8turn0search10

Diseño:

```bash
cerebro providers list
cerebro providers add openrouter
cerebro models list
cerebro models use openrouter/free
```

API key:

```bash
export OPENROUTER_API_KEY="..."
```

Nunca se almacenará una clave en código fuente, logs o datasets.

## 12. Modelos locales

Cerebro podrá cargar modelos locales compatibles con Transformers u otros backends.

Objetivos:

- CPU para modelos pequeños;
- GPU opcional;
- cuantización cuando proceda;
- descarga/cache;
- tokenizer;
- generación;
- streaming;
- adapters;
- checkpoints;
- metadata de procedencia.

La primera integración local priorizará modelos pequeños que puedan ejecutarse en hardware limitado.## 13. ¿Cómo construiremos nuestra propia IA?

La decisión estratégica de 1.0 es inequívoca: **Cerebro debe llegar a 1.0 con un modelo propio entrenado e integrado**.

No necesitamos que el primer modelo tenga miles de millones de parámetros para que sea propio. Debemos elegir un tamaño inicial que podamos entrenar, evaluar, reproducir y mejorar con el compute disponible.

La estrategia será teacher → student:

```text
LLMs grandes ya entrenados
        ↓
Teachers / evaluadores
        ↓
Dataset sintético + datos curados
        ↓
Filtrado / deduplicación / validación
        ↓
SFT / distillation / PEFT
        ↓
Cerebro Model 1.0
        ↓
Benchmark + safety + regresión
```

Los parámetros del teacher **no se transfieren automáticamente** al student. El student aprende mediante ejemplos, preferencias, trazas permitidas o logits cuando estén disponibles.

El pretraining desde cero sigue siendo una línea posible, pero no será un requisito de que el modelo 1.0 tenga un tamaño enorme. 1.0 exige un modelo propio útil y reproducible; 1.x ampliará capacidad, datos y arquitectura.

## 14. Stack de entrenamiento

Propuesta:

- PyTorch como framework principal;
- Transformers para modelos compatibles;
- PEFT/LoRA para adaptación eficiente;
- TRL para entrenamiento de lenguaje/instrucción cuando encaje;
- datasets versionados;
- safetensors para pesos;
- Accelerate/compute abstraction;
- evaluación antes de publicar artefactos.

PEFT permite entrenar una pequeña cantidad de parámetros adicionales manteniendo congelado el modelo base, reduciendo costes de memoria y almacenamiento. citeturn0search7turn0search1

## 15. Fases 21–30: modelos y Cerebro Model

### 21 — ModelProvider
Contrato único para cualquier backend.

### 22 — Local inference
Primer modelo pequeño local y loader reproducible.

### 23 — Tokenization
Tokenizer propio o seleccionado para la primera generación, con metadata versionada.

### 24 — OpenRouter
Provider remoto para desarrollo, comparación y teachers.

### 25 — Streaming
Respuestas incrementales.

### 26 — Tool calling
Normalización de llamadas a herramientas.

### 27 — Teacher registry
Registro de teachers, capacidades, versiones, licencias, términos de uso y procedencia.

### 28 — Model registry
Registro de Cerebro checkpoints, adapters, tokenizers, configs y hashes.

### 29 — Cerebro Model 1.0
Primer checkpoint propio entrenado, evaluado e integrado en el runtime.

### 30 — Model gate
Pruebas de generación, reproducibilidad, seguridad, regresión y calidad mínima antes de aceptar el checkpoint.

## 16. Memoria unificada

Capas:

```
Working
   ↓
Episodic
   ↓
Semantic
   ↓
Procedural
   ↓
Long-term
```

Componentes:

- almacenamiento;
- embeddings;
- índice;
- búsqueda lexical;
- búsqueda vectorial;
- recuperación híbrida;
- reranking;
- deduplicación;
- consolidación;
- expiración;
- importancia;
- procedencia.

## 17. Fases 31–40: memoria

31. Memory API.
32. Working memory.
33. Episodic memory.
34. Semantic memory.
35. Procedural memory.
36. Retrieval engine.
37. Embeddings.
38. Hybrid retrieval.
39. Consolidation/forgetting.
40. Memory benchmark.

Cada memoria deberá poder indicar:

- origen;
- timestamp;
- confianza;
- relevancia;
- scope;
- propietario;
- sensibilidad;
- motivo de conservación.

## 18. Tools

Arquitectura:

```
Model proposes
      ↓
Policy evaluates
      ↓
Permission check
      ↓
Confirmation if needed
      ↓
Execution
      ↓
Observation
      ↓
Audit
```

Tipos iniciales:

- filesystem;
- shell controlado;
- HTTP;
- Python limitado;
- memoria;
- búsqueda;
- procesos;
- extensiones.

No se concederá acceso ilimitado al modelo.

## 19. Fases 41–50: agentes/tools

41. Tool protocol.
42. Tool registry.
43. Schemas.
44. Permission engine.
45. Confirmation flow.
46. Execution sandbox.
47. Timeout/retry.
48. Audit events.
49. Skill registry.
50. Agent/tool benchmark.## 20. Seguridad e identidad

Sistema de claves:

```
cerebro keys create
cerebro keys list
cerebro keys revoke
cerebro keys rotate
```

Scopes previstos:

- models:read
- models:run
- memory:read
- memory:write
- tools:read
- tools:execute
- training:read
- training:run
- admin:*

Cada clave podrá tener:

- scopes;
- expiración;
- rate limit;
- presupuesto;
- IP/origen opcional;
- auditoría.

## 21. Fases 51–60: seguridad

51. Identity model.
52. API key store.
53. Hashing/secure storage.
54. Scope enforcement.
55. Revocation.
56. Rotation.
57. Rate limits.
58. Budgets.
59. Secret handling.
60. Security audit.

## 22. API HTTP

Servidor opcional:

```
POST /v1/chat
POST /v1/generate
POST /v1/tools/execute
GET  /v1/models
GET  /v1/memory
POST /v1/train/jobs
GET  /v1/jobs/{id}
GET  /health
```

Requisitos:

- authentication;
- request IDs;
- validation;
- timeouts;
- streaming;
- structured errors;
- audit;
- health/readiness.

## 23. CLI

```
cerebro
cerebro chat
cerebro run
cerebro doctor
cerebro config
cerebro models
cerebro providers
cerebro memory
cerebro tools
cerebro keys
cerebro train
cerebro jobs
cerebro evaluate
cerebro benchmark
```

El CLI será una interfaz sobre la API pública, no una segunda arquitectura.## 24. Fases 61–70: API/CLI

61. Python API.
62. CLI foundation.
63. Interactive chat.
64. Model commands.
65. Provider commands.
66. Memory commands.
67. Tool commands.
68. Key commands.
69. Training/job commands.
70. REST server.

## 25. Configuración

Prioridad:

1. argumentos explícitos;
2. variables de entorno;
3. archivo de configuración;
4. defaults seguros.

Configuración por perfiles:

- local;
- termux;
- development;
- ci;
- gpu;
- remote;
- production.

No se deberán incluir secretos en el repositorio.

## 26. Entrenamiento y sistema Teacher → Student

El entrenamiento propio será una parte central del producto, no una funcionalidad experimental posterior.

Pipeline principal:

```text
Teachers
  ↓
Data Factory
  ↓
Validate
  ↓
Clean / Deduplicate
  ↓
Safety / License / Provenance filter
  ↓
Tokenize
  ↓
Split train / validation / holdout
  ↓
SFT / Distillation / PEFT
  ↓
Checkpoint
  ↓
Evaluate
  ↓
Regression against previous checkpoint
  ↓
Select
  ↓
Package
  ↓
Register
```

La Data Factory podrá generar ejemplos de instrucción, código, matemáticas, razonamiento de tareas, uso de herramientas y correcciones. No se asumirán permisos de entrenamiento sobre outputs de terceros: cada teacher/dataset deberá registrar licencia, términos aplicables y procedencia.

Se evitará el deterioro por datos sintéticos repetitivos mediante deduplicación, datos frescos, conjuntos humanos/curados cuando estén disponibles y benchmarks holdout que nunca entren en el modelo.

Todo job debe registrar:

- commit;
- dataset version;
- model base;
- tokenizer;
- hiperparámetros;
- hardware;
- seed;
- dependencia versions;
- métricas;
- checkpoint hash.

## 27. Fases 71–80: training y modelo propio

71. Dataset contracts.
72. Teacher registry + provenance.
73. Synthetic Data Factory.
74. Dataset validator, license filter y deduplicación.
75. Tokenization pipeline.
76. SFT + LoRA/PEFT baseline.
77. Distillation/evaluation pipeline.
78. Checkpoint/resume + model lineage.
79. Model artifact packaging y registry.
80. **Cerebro Model 1.0 Training Gate**.## 28. Compute

Tres niveles:

### Local
Termux/CPU para desarrollo, pruebas, inferencia ligera y entrenamientos pequeños.

### CI
GitHub Actions para tests, builds, benchmarks, validación y pequeños jobs reproducibles.

### GPU/Cloud
Runner GPU o servicio externo para entrenamiento pesado.

PyTorch dispone de APIs de entrenamiento distribuido como DDP y FSDP2 para cargas que necesitan múltiples dispositivos o máquinas. citeturn0search0turn0search4

Importante: GitHub Actions no implica automáticamente GPU. La arquitectura deberá separar el scheduler del backend físico.

## 29. Training orchestrator

```
cerebro train submit
        ↓
Job spec
        ↓
Compute scheduler
        ↓
local / CI / GPU / external
        ↓
artifact
        ↓
evaluation
        ↓
registry
```

Comandos:

```bash
cerebro train run config.toml
cerebro train submit --backend github
cerebro train status
cerebro train logs
cerebro train cancel
cerebro models register
```

## 30. Fases 81–90: compute/CI

81. ComputeBackend.
82. Local backend.
83. GitHub backend.
84. GPU backend contract.
85. External backend.
86. Job queue/state.
87. Artifact upload/download.
88. Training status.
89. CI training smoke test.
90. Distributed-training readiness.

## 31. Observabilidad

Eventos mínimos:

- request.started;
- model.called;
- memory.retrieved;
- tool.requested;
- tool.allowed;
- tool.denied;
- tool.executed;
- response.generated;
- evaluation.completed;
- training.started;
- training.completed;
- error.

Cada evento tendrá timestamp y execution ID.## 32. Evaluación

No mediremos únicamente “responde bien”.

Suites:

- generation;
- instruction following;
- reasoning;
- planning;
- memory;
- retrieval;
- tool use;
- safety;
- latency;
- resource usage;
- regression.

Métricas:

- task success;
- exact match cuando aplique;
- retrieval recall;
- tool success;
- failure rate;
- latency;
- token usage;
- memory footprint;
- cost.

## 33. Fases 91–95: evaluación y release

91. Unified benchmark.
92. Regression benchmark.
93. Safety benchmark.
94. Performance benchmark.
95. Release qualification.

## 34. Migración V2–V5

Orden obligatorio:

1. namespace;
2. public facade;
3. provider interface;
4. current model adapter;
5. V5 runtime adapter;
6. memory adapter;
7. tools adapter;
8. security adapter;
9. configuration;
10. CLI;
11. tests;
12. packaging.

No se hará un “big bang rewrite”.

## 35. Retirada de legado

Un módulo histórico solo se elimina cuando:

- su funcionalidad existe en 1.0;
- existe test equivalente;
- existe documentación;
- instalación limpia no lo necesita;
- benchmark no empeora;
- no rompe la API pública.

Los elementos históricos podrán conservarse en `legacy/` cuando tengan valor documental.## 36. Fases 96–100: publicación

96. Clean install.
97. Wheel/sdist validation.
98. TestPyPI qualification.
99. PyPI release candidate.
100. v1.0.0 final.

## 37. Release Gate 1.0

Debe cumplirse todo. **El modelo propio es obligatorio; un wrapper de LLM externo no cumple el gate.**

- [ ] `pip install cerebro-zero`
- [ ] `cerebro` arranca
- [ ] `from cerebro_zero import Cerebro`
- [ ] modelo local operativo
- [ ] **Cerebro Model 1.0 propio, entrenado y versionado**
- [ ] checkpoint reproducible con hash
- [ ] dataset de entrenamiento versionado
- [ ] teacher registry y provenance completos
- [ ] benchmark de Cerebro Model 1.0 superado
- [ ] regresión contra checkpoints anteriores
- [ ] provider remoto operativo
- [ ] OpenRouter opcional
- [ ] memoria persistente
- [ ] retrieval
- [ ] tools con permisos
- [ ] API keys
- [ ] auditoría
- [ ] límites de ejecución
- [ ] REST API
- [ ] evaluación
- [ ] benchmarks
- [ ] CPU training smoke test
- [ ] backend GPU definido
- [ ] artifacts reproducibles
- [ ] CI verde
- [ ] documentación completa
- [ ] instalación limpia
- [ ] TestPyPI verificado
- [ ] PyPI verificado
- [ ] 1.0 changelog
- [ ] migration guide

## 38. Estrategia de IA real

La estrategia será híbrida:

```
             Cerebro Zero
                  |
       +----------+----------+
       |                     |
  Model Provider       Cognitive Runtime
       |                     |
 +-----+------+       memory/reason/tools
 |     |      |
local OpenRouter own
```

El LLM no será “todo Cerebro”. Será un componente del cerebro.

Esto permite cambiar de modelo sin reescribir memoria, seguridad, herramientas o planificación.## 39. Cerebro Model y linaje de versiones

La identidad de Cerebro Zero será una combinación inseparable de runtime y modelo propio:

### Capa 1 — Cerebro Runtime
Arquitectura cognitiva, memoria, tools, políticas, evaluación y control.

### Capa 2 — Cerebro Training System
Teachers, Data Factory, distillation, SFT, PEFT, evaluación, selección y reproducción.

### Capa 3 — Cerebro Model
Los pesos propios que constituyen la línea neuronal de Cerebro.

### Política de linaje

```text
Cerebro Model 1.0
      ↓ training adicional + datos nuevos + mejoras validadas
Cerebro Model 1.1
      ↓
Cerebro Model 1.2
      ↓
Cerebro Model 1.3 ...
```

Cada 1.x deberá partir del checkpoint de la versión anterior o documentar explícitamente una migración de arquitectura. No se reiniciará la identidad del modelo desde cero por comodidad.

“Más inteligente” no significa únicamente “más parámetros”. Cada versión documentará qué cambia en capacidad, conocimiento, datos, arquitectura, herramientas, memoria y métricas.

No afirmaremos que un modelo es entrenado desde cero si realmente parte de otro modelo. Si usamos teachers, eso se documentará como teacher/student training o distillation.

## 40. Datos de entrenamiento

Datasets internos deberán separar:

- instruction;
- reasoning/task traces permitidos;
- tool use;
- memory examples;
- correction examples;
- evaluation-only;
- safety;
- synthetic data;
- human-reviewed data.

Cada dataset tendrá versión y licencia/origen documentados.

Nunca se incorporarán secretos, API keys, datos privados no autorizados o contenido cuya licencia no permita el uso previsto.

## 41. Evolución continua del modelo

Roadmap de modelos:

```text
T0 — teachers externos y modelos de referencia
T1 — dataset Cerebro inicial
T2 — Cerebro student inicial
T3 — Cerebro Model 1.0
T4 — Cerebro Model 1.1
T5 — Cerebro Model 1.2 ...
```

La versión del runtime y la versión del modelo serán independientes, pero la release 1.0 deberá incluir obligatoriamente un modelo Cerebro 1.0.

Ejemplo:

```text
Cerebro Runtime 1.0.0
Cerebro Model 1.0
```

Para cada nueva versión 1.x se conservarán:

- checkpoint padre;
- dataset y deltas de datos;
- teacher(s) usados;
- configuración de entrenamiento;
- tokenizer;
- métricas antes/después;
- benchmark holdout;
- hashes de artefactos;
- changelog del modelo.

Una nueva versión solo será candidata a publicación si supera el gate de regresión y aporta una mejora documentada en capacidades, cobertura o eficiencia.

## 42. Dependencias opcionales

Instalación mínima:

```bash
pip install cerebro-zero
```

Extras:

```bash
pip install cerebro-zero[local]
pip install cerebro-zero[openrouter]
pip install cerebro-zero[training]
pip install cerebro-zero[server]
pip install cerebro-zero[all]
```

La instalación base no deberá obligar a descargar PyTorch/GPU stacks enormes si el usuario solo quiere usar una API remota.

## 43. Documentación

Documentos finales:

- README;
- Quickstart;
- Architecture;
- Model Providers;
- Local Models;
- OpenRouter;
- Memory;
- Tools;
- Security;
- API;
- CLI;
- Training;
- Compute;
- GitHub Actions;
- Deployment;
- Android/Termux;
- Evaluation;
- Troubleshooting;
- Migration V5 → 1.0;
- Release guide;
- Model card.

## 44. Testing

Capas:

1. unit;
2. integration;
3. runtime;
4. model-provider;
5. memory;
6. tools;
7. security;
8. API;
9. CLI;
10. packaging;
11. benchmark;
12. clean-install;
13. Termux qualification.

Los tests históricos se conservarán durante la migración para detectar regresiones.## 45. CI

Pipeline:

```
lint
  ↓
unit
  ↓
integration
  ↓
security
  ↓
benchmark
  ↓
build
  ↓
install wheel
  ↓
smoke test
  ↓
artifact
```

Training:

```
dataset validation
  ↓
CPU smoke training
  ↓
evaluation
  ↓
artifact
```

Los jobs GPU serán separados de los jobs obligatorios de PR.

## 46. GitHub automation

Workflows previstos:

- `ci.yml`
- `package.yml`
- `benchmark.yml`
- `training-smoke.yml`
- `training-gpu.yml`
- `release.yml`
- `security.yml`

El workflow de GPU solo se activará sobre un runner compatible o backend externo explícito.

## 47. Compatibilidad

Target inicial:

- Python 3.10–3.14;
- Linux;
- Android/Termux con capacidades adaptadas.

La compatibilidad de modelos dependerá de hardware/backend y se declarará explícitamente.

## 48. Versionado

Antes de 1.0:

- `1.0.0aN`
- `1.0.0bN`
- `1.0.0rcN`

Lanzamiento:

- `1.0.0`

Después:

- `1.1.x`
- `1.2.x`
- etc.

Breaking changes de la API pública requerirán una nueva major version.## 49. Definition of Done

Cerebro Zero 1.0 está terminado cuando un usuario nuevo pueda:

1. instalarlo;
2. ejecutar `cerebro`;
3. configurar un provider;
4. hablar con un modelo;
5. activar memoria;
6. usar tools autorizadas;
7. inspeccionar logs;
8. evaluar resultados;
9. cambiar de modelo;
10. ejecutar un modelo local compatible;
11. ejecutar/adaptar un modelo pequeño;
12. delegar training pesado;
13. recuperar un artefacto;
14. actualizar el paquete sin romper la API pública.

## 50. Orden de ejecución real

No construiremos todo a la vez, pero **el entrenamiento del modelo propio no se dejará para después del lanzamiento**.

### Bloque A
Namespace + contratos + facade.

### Bloque B
Adapter V5 + tests.

### Bloque C
ModelProvider + local + tokenizer.

### Bloque D
Teachers + OpenRouter + Data Factory.

### Bloque E
Memoria unificada.

### Bloque F
Tools + security.

### Bloque G
CLI + REST.

### Bloque H
Training system + distillation + SFT/PEFT.

### Bloque I
Cerebro Model 1.0 + evaluación + model gate.

### Bloque J
Compute/GitHub + reproducibilidad.

### Bloque K
Packaging + release.

**Regla:** no se declara terminado el bloque K hasta que el bloque I haya producido y validado un checkpoint propio de Cerebro.

## 51. Primera milestone

**Milestone M1 — Cerebro Zero 1.0 Foundation**

Objetivo:

```
from cerebro_zero import Cerebro
```

funciona sin romper la suite existente.

Además:

- provider contract;
- result contract;
- config contract;
- runtime adapter;
- tests;
- documentación.

Después de M1 comenzará en paralelo la integración de teachers, modelos locales y el pipeline de entrenamiento propio. El modelo Cerebro 1.0 debe existir antes del release final.

## 52. Decisión estratégica definitiva

Sí: utilizaremos IA ya entrenada y, cuando sea conveniente, modelos con miles de millones de parámetros, **pero como teachers y herramientas de desarrollo, no como sustituto del modelo Cerebro**.

La arquitectura será:

```text
                    TEACHERS
        ┌─────────────┼─────────────┐
      código       razonamiento    general
        │              │              │
        └────────── Data Factory ─────┘
                       ↓
              Filter / Evaluate
                       ↓
                 Cerebro Student
                       ↓
                 Cerebro Model 1.0
                       ↓
              Runtime + Memory + Tools
```

Los teachers podrán incluir proveedores remotos, modelos open-weight y varios especialistas. Se elegirán por tarea y por permisos de uso.

**Regla de identidad:** Cerebro Zero 1.0 no será “un wrapper alrededor de OpenRouter”. OpenRouter y otros LLMs externos podrán ayudar a construirlo, evaluarlo o compararlo; el artefacto publicado deberá contener una línea propia de modelo Cerebro entrenada y reproducible.

## 53. Meta final

```
                    CEREBRO ZERO 1.0
                           |
       +-------------------+-------------------+
       |                   |                   |
    Cognition           Memory              Models
       |                   |                   |
 Reason/Plan          Retrieve/Store      Local/API/Own
       |                   |                   |
       +-------------------+-------------------+
                           |
                         Tools
                           |
                        Security
                           |
                      Evaluation
                           |
                     Compute/Training
                           |
                       Packaging
                           |
                         PyPI
```

**Este documento será el roadmap maestro.** Toda implementación futura debe comprobar primero en qué fase encaja y qué contrato de 1.0 afecta.
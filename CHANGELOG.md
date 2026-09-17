# 📝 CHANGELOG

## [2.1.0] - 2026-09-17

### 🔴 TECHNICAL INTEGRITY RELEASE

#### FASE 2.1.8 — Documentación coherente
- README actualizado al estado real (308 tests)
- CHANGELOG con todas las fases 2.1
- Limpieza de archivos muertos

#### FASE 2.1.7 — Persistencia segura
- Formato JSON + NPY (sin pickle)
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

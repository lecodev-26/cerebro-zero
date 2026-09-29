from dataclasses import dataclass
from .contracts import DatasetRecord

@dataclass(frozen=True)
class PromptSpec:
    prompt: str
    task: str
    language: str = "es"

class DatasetQuality:
    def score(self, record: DatasetRecord) -> float:
        instruction = record.instruction.strip()
        response = record.response.strip()
        if not instruction or not response:
            return 0.0
        score = 0.5
        if len(response) >= 40: score += 0.15
        if len(response) <= 12000: score += 0.10
        if response.lower() != instruction.lower(): score += 0.10
        if any(mark in response for mark in (".", ":", "?", "!")): score += 0.05
        if record.task != "general": score += 0.05
        if record.metadata.get("model"): score += 0.05
        return min(score, 1.0)

class PromptCatalog:
    DEFAULTS = (
        PromptSpec("Explica qué es una función en Python y muestra un ejemplo sencillo.", "coding"),
        PromptSpec("Escribe un ejemplo de un bucle for en Python y explica qué hace.", "coding"),
        PromptSpec("Explica qué es una lista en Python y cuándo conviene usarla.", "coding"),
        PromptSpec("Explica la diferencia entre una clase y una función en Python.", "coding"),
        PromptSpec("Explica qué es una API REST y muestra un ejemplo conceptual.", "coding"),
        PromptSpec("Explica qué es una excepción en Python y cómo manejarla.", "coding"),
        PromptSpec("Resuelve 17 * 23 y explica el cálculo paso a paso de forma breve.", "math"),
        PromptSpec("Explica cómo comprobar si un número es primo.", "math"),
        PromptSpec("Explica la diferencia entre media, mediana y moda.", "math"),
        PromptSpec("Explica por qué el cielo puede verse azul durante el día.", "science"),
        PromptSpec("Explica de forma sencilla qué es la fotosíntesis.", "science"),
        PromptSpec("Describe qué ocurre cuando el agua pasa de líquido a vapor.", "science"),
        PromptSpec("Resume en tres puntos cómo abordarías un problema complejo.", "reasoning"),
        PromptSpec("Compara dos soluciones y explica qué información falta para decidir.", "reasoning"),
        PromptSpec("Detecta una contradicción en un requisito y explica cómo resolverla.", "reasoning"),
        PromptSpec("Explica cómo separar hechos, hipótesis y opiniones en un análisis.", "reasoning"),
        PromptSpec("Convierte este objetivo en un plan de tres pasos: crear una API sencilla.", "planning"),
        PromptSpec("Diseña un plan para migrar un proyecto Python sin romper la API pública.", "planning"),
        PromptSpec("Divide en tareas pequeñas este objetivo: añadir memoria persistente.", "planning"),
        PromptSpec("Propón criterios de aceptación para una nueva función de software.", "planning"),
        PromptSpec("Analiza este requisito y menciona dos riesgos: guardar datos sin cifrar.", "analysis"),
        PromptSpec("Analiza un cambio de software que aumenta complejidad y propone cómo medirlo.", "analysis"),
        PromptSpec("Revisa un diseño de API y enumera posibles errores de seguridad.", "analysis"),
        PromptSpec("Analiza una afirmación técnica y explica qué evidencia la verificaría.", "analysis"),
        PromptSpec("Responde en español de forma clara y concisa a una pregunta técnica.", "instruction"),
        PromptSpec("Explica una idea técnica para una persona que empieza a programar.", "instruction"),
        PromptSpec("Reescribe una explicación técnica para que sea más clara y estructurada.", "instruction"),
        PromptSpec("Resume una respuesta larga en cinco puntos sin perder lo esencial.", "instruction"),
        PromptSpec("Explica qué información necesitas antes de ejecutar una acción peligrosa.", "safety"),
        PromptSpec("Propón una forma segura de validar una entrada de usuario.", "safety"),
        PromptSpec("Explica por qué una API key no debe aparecer en código fuente.", "safety"),
        PromptSpec("Describe qué debería registrarse en una auditoría de herramientas.", "safety"),
        PromptSpec("Explica qué es una memoria episódica en un agente de IA.", "ai"),
        PromptSpec("Explica la diferencia entre memoria semántica y procedimental.", "ai"),
        PromptSpec("Describe el flujo observe, reason, plan, act y verify.", "ai"),
        PromptSpec("Explica para qué sirve un KV cache en un Transformer.", "ai"),
        PromptSpec("Describe cómo usarías una herramienta externa con permisos y auditoría.", "tools"),
        PromptSpec("Diseña un esquema sencillo para validar argumentos de una herramienta.", "tools"),
        PromptSpec("Explica qué ocurre cuando una herramienta solicita una acción de alto riesgo.", "tools"),
        PromptSpec("Propón cómo registrar el resultado de una herramienta para aprendizaje.", "tools"),
        PromptSpec("Explain in English what a REST API is and give a short example.", "instruction", "en"),
        PromptSpec("Explain in English why reproducible machine learning experiments need fixed seeds.", "reasoning", "en"),
        PromptSpec("Write a short English explanation of train, validation and test splits.", "instruction", "en"),
        PromptSpec("Explain in English why benchmark data must remain outside the training set.", "safety", "en"),
    )

    def sample(self, tasks=None, limit=10):
        wanted = set(tasks or ())
        rows = [p for p in self.DEFAULTS if not wanted or p.task in wanted]
        if wanted:
            return rows[:max(0, limit)]
        buckets = {}
        for row in rows:
            buckets.setdefault(row.task, []).append(row)
        ordered = []
        keys = list(buckets)
        while keys and len(ordered) < max(0, limit):
            next_keys = []
            for key in keys:
                if buckets[key] and len(ordered) < limit:
                    ordered.append(buckets[key].pop(0))
                if buckets[key]:
                    next_keys.append(key)
            keys = next_keys
        return ordered

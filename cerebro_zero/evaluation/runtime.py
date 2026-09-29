from ..compat import V5Evaluator

class RuntimeEvaluator(V5Evaluator):
    def evaluate(self, value):
        if hasattr(value, "evaluation") and value.evaluation:
            return float(value.evaluation.score)
        return float(value) if isinstance(value, (int, float)) else 0.0

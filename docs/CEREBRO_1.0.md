# Cerebro Zero 1.0

Cerebro Zero 1.0 unifies the V2–V5 work under one public package and stable facade.

Python API:

    from cerebro_zero import Cerebro
    ai = Cerebro()
    response = ai.run("Analiza este proyecto y propón mejoras")
    print(response.text)

CLI:

    pip install cerebro-zero
    cerebro --version
    cerebro "Analiza este proyecto"

The runtime contains cognitive orchestration, unified memory, guarded tools, security policy, evaluation and a model/training subsystem. External teacher models may create supervised or synthetic training data, while the release model remains a Cerebro-owned checkpoint with explicit lineage and evaluation.

V2, V3, V4 and V5 remain historical stages. New development after 1.0 follows the 1.1, 1.2, ... evolution policy.

"""
Cerebro Zero CLI — Interfaz de línea de comandos
==================================================

Uso:
    python cli.py info
    python cli.py chat "5 + 3"
    python cli.py plan "Entrenar el modelo"
    python cli.py benchmark
    python cli.py test
    python cli.py --version
"""

import sys
import os
import argparse

# Asegurar que podemos importar módulos del proyecto
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

VERSION = "3.0.0"


# ============================================
# COMANDOS
# ============================================

def cmd_info(args):
    """Muestra información del proyecto"""
    print("=" * 60)
    print(f"🧠 CEREBRO ZERO {VERSION}")
    print("=" * 60)
    print()
    print("Autonomous Learning Cognitive Agent")
    print("Construido desde cero en Python + NumPy")
    print("Sin frameworks. Sin trampas. Solo código.")
    print()
    
    try:
        from agent.cerebro_v3 import CerebroV3
        cerebro = CerebroV3(nombre="info", verbose=False)
        stats = cerebro.stats()
        
        print("📊 ESTADO DEL SISTEMA")
        print("-" * 60)
        print(f"  Nombre:        {stats['nombre']}")
        print(f"  Versión:       {stats['version']}")
        print(f"  Tools:         {stats['tools_registradas']}")
        print(f"  Tokenizer:     {stats['tokenizer_vocab']} tokens")
        print(f"  Working mem:   {stats['working']}")
        print()
        
        print("🧩 COMPONENTES 3.0")
        print("-" * 60)
        componentes = [
            ("3.1", "Working Memory", "✅"),
            ("3.2", "Procedural Memory", "✅"),
            ("3.3", "World Model", "✅"),
            ("3.4", "Planner 3.0", "✅"),
            ("3.5", "Learning Engine", "✅"),
            ("3.6", "RL básico", "✅"),
            ("3.7", "Benchmark 4.0", "✅"),
            ("3.8", "Tokenizer 3.0", "✅"),
            ("3.9", "Tool System 3.0", "✅"),
            ("3.10", "Reproducibilidad", "✅"),
            ("3.11", "Integración final", "✅"),
            ("3.12", "Packaging", "✅"),
        ]
        for num, nombre, estado in componentes:
            print(f"  {estado} {num}: {nombre}")
        print()
    except Exception as e:
        print(f"⚠️ Error cargando componentes: {e}")


def cmd_chat(args):
    """Procesa un input y muestra la respuesta"""
    from agent.cerebro_v3 import CerebroV3
    
    cerebro = CerebroV3(nombre="cli", verbose=False)
    
    if args.texto:
        # Modo one-shot
        r = cerebro.procesar(args.texto)
        print(f"👤 {args.texto}")
        print(f"🤖 {r.output}")
        if r.tool_usada:
            print(f"   [tool={r.tool_usada}, conf={r.confianza:.2f}, "
                  f"lat={r.latencia*1000:.1f}ms]")
        if r.plan:
            print(f"   [plan={len(r.plan.subgoals)} subgoals]")
    else:
        # Modo interactivo
        print(f"🧠 Cerebro Zero {VERSION} — Modo chat")
        print("Escribe 'salir' para terminar")
        print()
        
        while True:
            try:
                texto = input("👤 > ").strip()
                if texto.lower() in ("salir", "exit", "quit", "q"):
                    break
                if not texto:
                    continue
                
                r = cerebro.procesar(texto)
                print(f"🤖 {r.output}")
                print()
            except (KeyboardInterrupt, EOFError):
                print()
                break
        
        print(f"\n👋 Hasta luego. Procesados: {cerebro.num_procesados}")


def cmd_plan(args):
    """Genera un plan y lo muestra"""
    from agent.cerebro_v3 import CerebroV3
    
    cerebro = CerebroV3(nombre="plan", verbose=False)
    plan = cerebro.planificar(args.goal)
    
    print(f"🎯 Objetivo: {plan.goal}")
    print(f"📋 Plan con {len(plan.subgoals)} subgoals")
    print()
    
    orden = plan.orden_topologico()
    for i, sg_id in enumerate(orden, 1):
        sg = plan.get_subgoal(sg_id)
        deps = f" (deps: {sg.depende_de})" if sg.depende_de else ""
        print(f"  {i}. {sg.id}{deps}")
        print(f"     → {sg.descripcion}")


def cmd_benchmark(args):
    """Ejecuta un benchmark rápido"""
    from evaluation.benchmark_v4 import (
        Benchmark, handler_math, handler_memory,
        handler_planning, handler_rl, handler_learning,
    )
    
    print("📊 Ejecutando benchmark...")
    print()
    
    bench = Benchmark(seed=42)
    bench.registrar_handler("math", handler_math)
    bench.registrar_handler("memory", handler_memory)
    bench.registrar_handler("planning", handler_planning)
    bench.registrar_handler("rl", handler_rl)
    bench.registrar_handler("learning", handler_learning)
    bench.cargar_casos()
    
    reporte = bench.ejecutar()
    print(reporte.resumen_texto())


def cmd_test(args):
    """Ejecuta los tests"""
    import subprocess
    print("🧪 Ejecutando tests...")
    print()
    resultado = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v"],
    )
    sys.exit(resultado.returncode)


# ============================================
# PARSER
# ============================================

def crear_parser():
    parser = argparse.ArgumentParser(
        prog="cerebro-zero",
        description=f"Cerebro Zero {VERSION} — Autonomous Learning Cognitive Agent",
    )
    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"Cerebro Zero {VERSION}",
    )
    
    subparsers = parser.add_subparsers(dest="comando")
    
    # info
    p_info = subparsers.add_parser("info", help="Muestra info del sistema")
    p_info.set_defaults(func=cmd_info)
    
    # chat
    p_chat = subparsers.add_parser("chat", help="Chat con el cerebro")
    p_chat.add_argument("texto", nargs="?", help="Texto a procesar (opcional)")
    p_chat.set_defaults(func=cmd_chat)
    
    # plan
    p_plan = subparsers.add_parser("plan", help="Genera un plan")
    p_plan.add_argument("goal", help="Objetivo a planificar")
    p_plan.set_defaults(func=cmd_plan)
    
    # benchmark
    p_bench = subparsers.add_parser("benchmark", help="Ejecuta benchmark")
    p_bench.set_defaults(func=cmd_benchmark)
    
    # test
    p_test = subparsers.add_parser("test", help="Ejecuta tests")
    p_test.set_defaults(func=cmd_test)
    
    return parser


def main():
    parser = crear_parser()
    args = parser.parse_args()
    
    if not args.comando:
        parser.print_help()
        return 0
    
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())

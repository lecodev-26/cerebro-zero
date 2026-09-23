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

from utils.visual import (
    consola, titulo, ok, warn, error, info, dim,
    seccion, bullet, kv, panel, tabla
)

VERSION = "3.0.0"


# ============================================
# COMANDOS
# ============================================

def cmd_info(args):
    """Muestra información del proyecto"""
    titulo(f"CEREBRO ZERO {VERSION}")
    
    info("Autonomous Learning Cognitive Agent")
    dim("Construido desde cero en Python + NumPy")
    dim("Sin frameworks. Sin trampas. Solo código.")
    consola.print()
    
    try:
        from agent.cerebro_v3 import CerebroV3
        cerebro = CerebroV3(nombre="info", verbose=False)
        stats = cerebro.stats()
        
        seccion("📊 Estado del sistema")
        tabla(
            ["Campo", "Valor"],
            [
                ["Nombre", stats["nombre"]],
                ["Versión", stats["version"]],
                ["Tools", stats["tools_registradas"]],
                ["Tokenizer", f"{stats['tokenizer_vocab']} tokens"],
                ["Working mem", stats["working"]],
            ],
            titulo="Sistema"
        )
        
        seccion("🧩 Componentes 3.0")
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
        tabla(
            ["#", "Componente", "Estado"],
            [[num, nombre, estado] for num, nombre, estado in componentes],
            titulo="Fases del bloque 3.x"
        )
        
    except Exception as e:
        error(f"Error cargando componentes: {e}")


def cmd_chat(args):
    """Procesa un input y muestra la respuesta"""
    from agent.cerebro_v3 import CerebroV3
    
    cerebro = CerebroV3(nombre="cli", verbose=False)
    
    if args.texto:
        # Modo one-shot
        r = cerebro.procesar(args.texto)
        
        consola.print(f"\n  [bold]👤[/bold] [white]{args.texto}[/white]")
        consola.print(f"  [bold]🤖[/bold] [green]{r.output}[/green]")
        
        detalles = []
        if r.tool_usada:
            detalles.append(f"tool=[cyan]{r.tool_usada}[/cyan]")
            detalles.append(f"conf=[green]{r.confianza:.2f}[/green]")
            detalles.append(f"lat=[dim]{r.latencia*1000:.1f}ms[/dim]")
        if r.plan:
            detalles.append(f"plan=[yellow]{len(r.plan.subgoals)} subgoals[/yellow]")
        
        if detalles:
            consola.print(f"       [dim]({' | '.join(detalles)})[/dim]")
        consola.print()
    else:
        # Modo interactivo
        titulo(f"Cerebro Zero {VERSION} — Modo chat")
        dim("Escribe 'salir' para terminar")
        consola.print()
        
        while True:
            try:
                texto = consola.input("[bold cyan]👤 >[/bold cyan] ").strip()
                if texto.lower() in ("salir", "exit", "quit", "q"):
                    break
                if not texto:
                    continue
                
                r = cerebro.procesar(texto)
                consola.print(f"[bold]🤖[/bold] [green]{r.output}[/green]")
                consola.print()
            except (KeyboardInterrupt, EOFError):
                consola.print()
                break
        
        consola.print()
        ok(f"Hasta luego. Procesados: {cerebro.num_procesados}")


def cmd_plan(args):
    """Genera un plan y lo muestra"""
    from agent.cerebro_v3 import CerebroV3
    
    cerebro = CerebroV3(nombre="plan", verbose=False)
    plan = cerebro.planificar(args.goal)
    
    titulo("PLANIFICACIÓN")
    kv("Objetivo", plan.goal, color="yellow")
    kv("Subgoals", len(plan.subgoals), color="cyan")
    consola.print()
    
    orden = plan.orden_topologico()
    for i, sg_id in enumerate(orden, 1):
        sg = plan.get_subgoal(sg_id)
        deps_str = f" [dim](deps: {sg.depende_de})[/dim]" if sg.depende_de else ""
        consola.print(f"  [bold cyan]{i:>2}.[/bold cyan] {sg.id}{deps_str}")
        consola.print(f"      [dim]→ {sg.descripcion}[/dim]")


def cmd_benchmark(args):
    """Ejecuta un benchmark rápido"""
    from evaluation.benchmark_v4 import (
        Benchmark, handler_math, handler_memory,
        handler_planning, handler_rl, handler_learning,
    )
    
    titulo("BENCHMARK 4.0")
    info("Ejecutando benchmark...")
    consola.print()
    
    bench = Benchmark(seed=42)
    bench.registrar_handler("math", handler_math)
    bench.registrar_handler("memory", handler_memory)
    bench.registrar_handler("planning", handler_planning)
    bench.registrar_handler("rl", handler_rl)
    bench.registrar_handler("learning", handler_learning)
    bench.cargar_casos()
    
    reporte = bench.ejecutar()
    
    # Mostrar tabla global
    g = reporte.global_metrics()
    seccion("🌍 Resultados globales")
    tabla(
        ["Métrica", "Valor"],
        [
            ["Total", g.total],
            ["Accuracy", f"[green]{g.accuracy*100:.2f}%[/green]"],
            ["Exact match", f"[green]{g.exact_match_rate*100:.2f}%[/green]"],
            ["Error rate", f"[red]{g.error_rate*100:.2f}%[/red]"],
            ["Latency p50", f"[cyan]{g.latency_p50*1000:.2f}ms[/cyan]"],
            ["Latency p95", f"[cyan]{g.latency_p95*1000:.2f}ms[/cyan]"],
        ],
        titulo="Métricas globales"
    )
    
    # Tabla por categoría
    seccion("📂 Resultados por categoría")
    filas = []
    for cat, m in sorted(reporte.por_categoria().items()):
        filas.append([
            cat,
            f"{m.correctos}/{m.total}",
            f"[green]{m.accuracy*100:.2f}%[/green]",
        ])
    tabla(["Categoría", "Correctos", "Accuracy"], filas, titulo="Por categoría")
    
    # Tabla por split
    seccion("🔀 Resultados por split")
    filas = []
    for split, m in sorted(reporte.por_split().items()):
        filas.append([
            split,
            f"{m.correctos}/{m.total}",
            f"[green]{m.accuracy*100:.2f}%[/green]",
        ])
    tabla(["Split", "Correctos", "Accuracy"], filas, titulo="Por split")


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

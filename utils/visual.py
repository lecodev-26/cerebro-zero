"""
Módulo visual — helpers con rich para toda la salida del proyecto
==================================================================

Uso:
    from utils.visual import (
        ok, warn, error, info, titulo,
        panel, tabla, barra_progreso, consola
    )
"""

from rich.console import Console
from rich.progress import (
    Progress, BarColumn, TextColumn, TimeRemainingColumn,
    TimeElapsedColumn, SpinnerColumn,
)
from rich.panel import Panel
from rich.table import Table
from rich.live import Live


# ============================================
# CONSOLA GLOBAL
# ============================================

consola = Console()


# ============================================
# MENSAJES SEMÁNTICOS
# ============================================

def ok(msg: str):
    """Mensaje de éxito (verde)"""
    consola.print(f"[bold green]✅ {msg}[/bold green]")


def warn(msg: str):
    """Mensaje de aviso (amarillo)"""
    consola.print(f"[bold yellow]⚠️  {msg}[/bold yellow]")


def error(msg: str):
    """Mensaje de error (rojo)"""
    consola.print(f"[bold red]❌ {msg}[/bold red]")


def info(msg: str):
    """Mensaje informativo (azul)"""
    consola.print(f"[bold blue]ℹ️  {msg}[/bold blue]")


def titulo(texto: str):
    """Título grande con separadores"""
    consola.print()
    consola.rule(f"[bold magenta]{texto}[/bold magenta]")
    consola.print()


def dim(msg: str):
    """Mensaje atenuado"""
    consola.print(f"[dim]{msg}[/dim]")


# ============================================
# PANELES
# ============================================

def panel(contenido: str, titulo: str = "", color: str = "cyan"):
    """Muestra un panel con borde"""
    consola.print(Panel(
        contenido,
        title=titulo if titulo else None,
        border_style=color,
    ))


# ============================================
# TABLAS
# ============================================

def tabla(cabeceras: list, filas: list, titulo: str = "") -> Table:
    """Crea y muestra una tabla"""
    t = Table(title=titulo if titulo else None)
    for c in cabeceras:
        t.add_column(c)
    for fila in filas:
        t.add_row(*[str(x) for x in fila])
    consola.print(t)
    return t


# ============================================
# BARRAS DE PROGRESO
# ============================================

def barra_progreso(descripcion: str = "Progreso"):
    """
    Devuelve un contexto de Progress listo para usar.
    
    Uso:
        with barra_progreso("Entrenando") as (progress, task):
            for i in range(10):
                progress.update(task, advance=1, loss=0.5)
    """
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}[/bold blue]"),
        BarColumn(bar_width=None),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TextColumn("[dim]ETA[/dim]"),
        TimeRemainingColumn(),
        console=consola,
    )
    task = progress.add_task(descripcion, total=100)
    return progress, task


def crear_progress(total: int, descripcion: str = "Procesando"):
    """
    Crea un Progress y un task con un total concreto.
    
    Devuelve (progress, task).
    
    Uso:
        progress, task = crear_progress(100, "Entrenando")
        with progress:
            for i in range(100):
                progress.update(task, advance=1)
    """
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}[/bold blue]"),
        BarColumn(bar_width=None),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=consola,
    )
    task = progress.add_task(descripcion, total=total)
    return progress, task


# ============================================
# UTILIDADES
# ============================================

def seccion(texto: str):
    """Subtítulo de sección"""
    consola.print()
    consola.print(f"[bold cyan]▶ {texto}[/bold cyan]")


def bullet(texto: str, color: str = "white"):
    """Punto de lista"""
    consola.print(f"  [{color}]•[/{color}] {texto}")


def kv(clave: str, valor, color: str = "green"):
    """Clave-valor alineado"""
    consola.print(f"  [dim]{clave}:[/dim] [{color}]{valor}[/{color}]")


# ============================================
# TEST
# ============================================

if __name__ == "__main__":
    titulo("TEST MÓDULO VISUAL")
    
    ok("Todo funciona correctamente")
    warn("Esto es un aviso")
    error("Esto es un error")
    info("Esto es información")
    dim("Esto es un mensaje atenuado")
    
    seccion("Sección de prueba")
    bullet("Primer punto")
    bullet("Segundo punto", color="yellow")
    
    consola.print()
    kv("Nombre", "Cerebro Zero")
    kv("Versión", "4.0.0", color="cyan")
    kv("Tests", "742", color="green")
    
    panel("Este es un panel con contenido\nmultilínea y borde cyan",
          titulo="Panel de prueba", color="cyan")
    
    consola.print()
    tabla(
        ["Modelo", "Accuracy", "Estado"],
        [
            ["MLP", "95.2%", "✅"],
            ["Transformer", "97.8%", "✅"],
            ["CNN", "—", "⏳"],
        ],
        titulo="Resultados"
    )
    
    consola.print()
    import time
    progress, task = crear_progress(20, "Simulando entrenamiento")
    with progress:
        for i in range(20):
            time.sleep(0.05)
            progress.update(task, advance=1)
    
    consola.print()
    ok("Test visual completado")

"""
Reinforcement Learning básico — Q-learning + GridWorld
========================================================

Filosofía:
    Environment → Agent → State → Action → Reward → Policy

Componentes:
- GridWorld: entorno simple NxN con meta y obstáculos
- QLearning: tabla Q con epsilon-greedy
- PoliticaNeuronal: MLP como aproximador de política (opcional)
- EntrenadorRL: orquesta episodios y métricas

NO pretende ser AGI. Es RL didáctico y verificable.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime


# ============================================
# GRIDWORLD
# ============================================

class GridWorld:
    """
    Entorno de rejilla NxN.
    
    Estados: (fila, col)
    Acciones: 0=arriba, 1=abajo, 2=izquierda, 3=derecha
    Recompensas:
        - Llegar a meta: +10
        - Caer en obstáculo: -5 (y reset a start)
        - Cada paso: -0.1 (fomenta caminos cortos)
    """
    
    def __init__(
        self,
        filas: int = 5,
        columnas: int = 5,
        inicio: Tuple[int, int] = (0, 0),
        meta: Tuple[int, int] = None,
        obstaculos: Optional[List[Tuple[int, int]]] = None,
        max_pasos: int = 100,
    ):
        self.filas = filas
        self.columnas = columnas
        self.inicio = inicio
        self.meta = meta if meta else (filas - 1, columnas - 1)
        self.obstaculos = set(obstaculos or [])
        self.max_pasos = max_pasos
        
        # Quitar meta del set de obstáculos si se solapan
        self.obstaculos.discard(self.meta)
        
        self.estado_actual = inicio
        self.pasos = 0
        self.recompensa_acumulada = 0.0
    
    def reset(self) -> Tuple[int, int]:
        self.estado_actual = self.inicio
        self.pasos = 0
        self.recompensa_acumulada = 0.0
        return self.estado_actual
    
    def acciones_disponibles(self) -> List[int]:
        return [0, 1, 2, 3]
    
    def num_estados(self) -> int:
        return self.filas * self.columnas
    
    def num_acciones(self) -> int:
        return 4
    
    def _aplicar_accion(
        self, estado: Tuple[int, int], accion: int
    ) -> Tuple[int, int]:
        f, c = estado
        if accion == 0:    # arriba
            f = max(0, f - 1)
        elif accion == 1:  # abajo
            f = min(self.filas - 1, f + 1)
        elif accion == 2:  # izquierda
            c = max(0, c - 1)
        elif accion == 3:  # derecha
            c = min(self.columnas - 1, c + 1)
        return (f, c)
    
    def step(self, accion: int) -> Tuple[Tuple[int, int], float, bool, dict]:
        """
        Devuelve (nuevo_estado, recompensa, terminado, info)
        """
        self.pasos += 1
        
        nuevo_estado = self._aplicar_accion(self.estado_actual, accion)
        
        # Calcular recompensa
        if nuevo_estado == self.meta:
            recompensa = 10.0
            terminado = True
        elif nuevo_estado in self.obstaculos:
            recompensa = -5.0
            terminado = False
            nuevo_estado = self.inicio  # reset a inicio
        else:
            recompensa = -0.1
            terminado = False
        
        # Timeout
        if self.pasos >= self.max_pasos:
            terminado = True
        
        self.estado_actual = nuevo_estado
        self.recompensa_acumulada += recompensa
        
        info = {"pasos": self.pasos, "estado": nuevo_estado}
        return nuevo_estado, recompensa, terminado, info
    
    def estado_a_indice(self, estado: Tuple[int, int]) -> int:
        return estado[0] * self.columnas + estado[1]
    
    def indice_a_estado(self, idx: int) -> Tuple[int, int]:
        return (idx // self.columnas, idx % self.columnas)
    
    def render(self) -> str:
        """Devuelve una representación visual del entorno"""
        lineas = []
        for f in range(self.filas):
            linea = ""
            for c in range(self.columnas):
                if (f, c) == self.estado_actual:
                    linea += " A "
                elif (f, c) == self.meta:
                    linea += " 🎯"
                elif (f, c) in self.obstaculos:
                    linea += " X "
                else:
                    linea += " . "
            lineas.append(linea)
        return "\n".join(lineas)
    
    def __repr__(self):
        return (
            f"GridWorld({self.filas}x{self.columnas}, "
            f"inicio={self.inicio}, meta={self.meta}, "
            f"obstaculos={len(self.obstaculos)})"
        )


# ============================================
# Q-LEARNING
# ============================================

class QLearning:
    """
    Q-Learning tabular con epsilon-greedy.
    
    Q[s][a] = valor esperado de tomar a en s
    """
    
    def __init__(
        self,
        num_estados: int,
        num_acciones: int,
        alpha: float = 0.1,     # learning rate
        gamma: float = 0.95,    # discount
        epsilon: float = 0.1,   # exploration
    ):
        self.num_estados = num_estados
        self.num_acciones = num_acciones
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Tabla Q inicializada a 0
        self.Q = [[0.0] * num_acciones for _ in range(num_estados)]
    
    def elegir_accion(self, estado: int) -> int:
        """Epsilon-greedy"""
        if random.random() < self.epsilon:
            return random.randint(0, self.num_acciones - 1)
        return self.mejor_accion(estado)
    
    def mejor_accion(self, estado: int) -> int:
        valores = self.Q[estado]
        max_val = max(valores)
        # Desempate aleatorio
        mejores = [a for a, v in enumerate(valores) if v == max_val]
        return random.choice(mejores)
    
    def aprender(
        self,
        estado: int,
        accion: int,
        recompensa: float,
        siguiente_estado: int,
        terminado: bool,
    ):
        """Actualización Q-Learning"""
        q_actual = self.Q[estado][accion]
        
        if terminado:
            target = recompensa
        else:
            target = recompensa + self.gamma * max(self.Q[siguiente_estado])
        
        self.Q[estado][accion] = q_actual + self.alpha * (target - q_actual)
    
    def valor_estado(self, estado: int) -> float:
        return max(self.Q[estado])
    
    def politica(self) -> List[int]:
        """Devuelve la mejor acción para cada estado"""
        return [self.mejor_accion(s) for s in range(self.num_estados)]
    
    def reset(self):
        self.Q = [[0.0] * self.num_acciones for _ in range(self.num_estados)]
    
    def __repr__(self):
        return f"QLearning(states={self.num_estados}, actions={self.num_acciones})"


# ============================================
# ENTRENADOR RL
# ============================================

@dataclass
class Episodio:
    """Un episodio de entrenamiento"""
    num: int
    pasos: int
    recompensa_total: float
    exito: bool
    epsilon: float
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> dict:
        return {
            "num": self.num,
            "pasos": self.pasos,
            "recompensa_total": self.recompensa_total,
            "exito": self.exito,
            "epsilon": self.epsilon,
        }


class EntrenadorRL:
    """
    Entrena un agente Q-Learning en un entorno.
    """
    
    def __init__(
        self,
        entorno: GridWorld,
        agente: Optional[QLearning] = None,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
    ):
        self.entorno = entorno
        self.agente = agente or QLearning(
            num_estados=entorno.num_estados(),
            num_acciones=entorno.num_acciones(),
        )
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.episodios: List[Episodio] = []
    
    def ejecutar_episodio(self, num: int, max_pasos: int = 100) -> Episodio:
        estado = self.entorno.reset()
        estado_idx = self.entorno.estado_a_indice(estado)
        
        recompensa_total = 0.0
        pasos = 0
        exito = False
        
        for _ in range(max_pasos):
            accion = self.agente.elegir_accion(estado_idx)
            nuevo_estado, recompensa, terminado, _ = self.entorno.step(accion)
            nuevo_idx = self.entorno.estado_a_indice(nuevo_estado)
            
            self.agente.aprender(
                estado_idx, accion, recompensa, nuevo_idx, terminado
            )
            
            estado_idx = nuevo_idx
            recompensa_total += recompensa
            pasos += 1
            
            if nuevo_estado == self.entorno.meta:
                exito = True
            
            if terminado:
                break
        
        # Decay epsilon
        self.agente.epsilon = max(
            self.epsilon_min,
            self.agente.epsilon * self.epsilon_decay,
        )
        
        ep = Episodio(
            num=num,
            pasos=pasos,
            recompensa_total=recompensa_total,
            exito=exito,
            epsilon=self.agente.epsilon,
        )
        self.episodios.append(ep)
        return ep
    
    def entrenar(self, num_episodios: int = 100) -> dict:
        """Entrena N episodios y devuelve métricas"""
        for i in range(num_episodios):
            self.ejecutar_episodio(i)
        
        return self.metricas()
    
    def metricas(self) -> dict:
        if not self.episodios:
            return {"episodios": 0}
        
        exitos = sum(1 for e in self.episodios if e.exito)
        recompensas = [e.recompensa_total for e in self.episodios]
        pasos_exitosos = [e.pasos for e in self.episodios if e.exito]
        
        # Últimos 20 episodios
        ultimos = self.episodios[-20:]
        tasa_ultimos = (
            sum(1 for e in ultimos if e.exito) / len(ultimos)
            if ultimos else 0.0
        )
        
        return {
            "episodios": len(self.episodios),
            "exitos": exitos,
            "tasa_exito": exitos / len(self.episodios),
            "tasa_exito_ultimos_20": tasa_ultimos,
            "recompensa_media": sum(recompensas) / len(recompensas),
            "recompensa_max": max(recompensas),
            "pasos_medios_exitosos": (
                sum(pasos_exitosos) / len(pasos_exitosos)
                if pasos_exitosos else 0
            ),
            "epsilon_final": self.agente.epsilon,
        }
    
    def evaluar(self, num_episodios: int = 10) -> dict:
        """
        Evalúa la política aprendida SIN exploración.
        """
        epsilon_original = self.agente.epsilon
        self.agente.epsilon = 0.0  # sin exploración
        
        exitos = 0
        pasos_totales = 0
        recompensa_total = 0.0
        
        for _ in range(num_episodios):
            estado = self.entorno.reset()
            estado_idx = self.entorno.estado_a_indice(estado)
            
            for _ in range(100):
                accion = self.agente.mejor_accion(estado_idx)
                nuevo_estado, recompensa, terminado, _ = self.entorno.step(accion)
                estado_idx = self.entorno.estado_a_indice(nuevo_estado)
                pasos_totales += 1
                recompensa_total += recompensa
                
                if nuevo_estado == self.entorno.meta:
                    exitos += 1
                
                if terminado:
                    break
        
        self.agente.epsilon = epsilon_original
        
        return {
            "episodios": num_episodios,
            "exitos": exitos,
            "tasa_exito": exitos / num_episodios,
            "pasos_medios": pasos_totales / num_episodios,
            "recompensa_media": recompensa_total / num_episodios,
        }
    
    def guardar(self, archivo: str):
        os.makedirs(os.path.dirname(archivo) or ".", exist_ok=True)
        data = {
            "Q": self.agente.Q,
            "alpha": self.agente.alpha,
            "gamma": self.agente.gamma,
            "epsilon": self.agente.epsilon,
            "num_estados": self.agente.num_estados,
            "num_acciones": self.agente.num_acciones,
            "metricas": self.metricas(),
            "guardado": datetime.now().isoformat(),
        }
        with open(archivo, "w") as f:
            json.dump(data, f, indent=2)
    
    def __repr__(self):
        return f"EntrenadorRL({len(self.episodios)} episodios)"


# ============================================
# TEST MANUAL
# ============================================

if __name__ == "__main__":
    from utils.visual import (
        consola, titulo, ok, warn, error, info, dim,
        seccion, bullet, kv, panel, tabla, crear_progress
    )
    import time

    titulo("RL BÁSICO — Q-Learning en GridWorld")

    # ============================================
    # GRIDWORLD 3x3
    # ============================================
    seccion("🌍 GridWorld 3x3")
    
    env = GridWorld(filas=3, columnas=3, inicio=(0, 0), meta=(2, 2))
    
    consola.print()
    consola.print(env.render())
    consola.print()
    
    kv("Estado inicial", env.reset(), color="cyan")
    kv("Meta", env.meta, color="yellow")
    kv("Estados totales", env.num_estados())
    kv("Acciones", env.num_acciones())

    # ============================================
    # STEPS MANUALES
    # ============================================
    seccion("🎮 Steps manuales (→ → ↓ ↓)")
    
    for accion in [3, 3, 1, 1]:
        estado, r, term, info = env.step(accion)
        flecha = ["↑", "↓", "←", "→"][accion]
        color = "green" if r > 0 else "dim"
        consola.print(
            f"  [bold]{flecha}[/bold] → "
            f"estado=[cyan]{estado}[/cyan]  "
            f"recompensa=[{color}]{r:+.2f}[/{color}]  "
            f"terminado=[yellow]{term}[/yellow]"
        )
    
    consola.print()
    consola.print(env.render())

    # ============================================
    # ENTRENAMIENTO
    # ============================================
    seccion("🧠 Entrenando Q-Learning en GridWorld 5x5")
    
    env2 = GridWorld(filas=5, columnas=5)
    agente = QLearning(num_estados=25, num_acciones=4, epsilon=0.5)
    entrenador = EntrenadorRL(env2, agente, epsilon_decay=0.99)
    
    num_episodios = 200
    progress, task = crear_progress(num_episodios, "Entrenando")
    
    exitos_recientes = []
    
    with progress:
        for i in range(num_episodios):
            ep = entrenador.ejecutar_episodio(i)
            exitos_recientes.append(1 if ep.exito else 0)
            if len(exitos_recientes) > 20:
                exitos_recientes.pop(0)
            
            # Mostrar recompensa actual como descripción
            progress.update(
                task,
                advance=1,
                description=(
                    f"[bold]Ep {i+1:>3}/{num_episodios}[/bold]  "
                    f"R=[cyan]{ep.recompensa_total:+6.2f}[/cyan]  "
                    f"ε=[yellow]{ep.epsilon:.3f}[/yellow]  "
                    f"éxito últimos 20: [green]{sum(exitos_recientes)}/20[/green]"
                )
            )
    
    consola.print()
    
    # ============================================
    # MÉTRICAS
    # ============================================
    seccion("📊 Métricas de entrenamiento")
    
    metricas = entrenador.metricas()
    tabla(
        ["Métrica", "Valor"],
        [
            ["Episodios totales", metricas["episodios"]],
            ["Éxitos", f"[green]{metricas['exitos']}[/green]"],
            ["Tasa de éxito", f"[green]{metricas['tasa_exito']*100:.2f}%[/green]"],
            ["Éxito últimos 20", f"[green]{metricas['tasa_exito_ultimos_20']*100:.2f}%[/green]"],
            ["Recompensa media", f"[cyan]{metricas['recompensa_media']:.3f}[/cyan]"],
            ["Recompensa máxima", f"[cyan]{metricas['recompensa_max']:.3f}[/cyan]"],
            ["Pasos medios (éxito)", metricas["pasos_medios_exitosos"]],
            ["Epsilon final", f"[yellow]{metricas['epsilon_final']:.4f}[/yellow]"],
        ],
        titulo="Resultados del entrenamiento"
    )

    # ============================================
    # EVALUACIÓN SIN EXPLORACIÓN
    # ============================================
    seccion("🎯 Evaluación sin exploración")
    
    eval_metrics = entrenador.evaluar(num_episodios=20)
    tabla(
        ["Métrica", "Valor"],
        [
            ["Episodios de evaluación", eval_metrics["episodios"]],
            ["Éxitos", f"[green]{eval_metrics['exitos']}[/green]"],
            ["Tasa de éxito", f"[green]{eval_metrics['tasa_exito']*100:.2f}%[/green]"],
            ["Pasos medios", eval_metrics["pasos_medios"]],
            ["Recompensa media", f"[cyan]{eval_metrics['recompensa_media']:.3f}[/cyan]"],
        ],
        titulo="Evaluación (política aprendida)"
    )

    # ============================================
    # POLÍTICA APRENDIDA
    # ============================================
    seccion("🗺️  Política aprendida")
    
    flechas = ["↑", "↓", "←", "→"]
    politica = entrenador.agente.politica()
    
    consola.print()
    for f in range(5):
        linea = "  "
        for c in range(5):
            idx = f * 5 + c
            if (f, c) == env2.meta:
                linea += " [bold yellow]🎯[/bold yellow] "
            else:
                color = "green" if flechas[politica[idx]] in ["→", "↓"] else "cyan"
                linea += f" [{color}]{flechas[politica[idx]]}[/{color}] "
        consola.print(linea)
    consola.print()

    # ============================================
    # PERSISTENCIA
    # ============================================
    seccion("💾 Persistencia")
    
    tmp = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "test_rl_tmp.json"
    )
    entrenador.guardar(tmp)
    kv("Guardado en", os.path.basename(tmp), color="cyan")
    kv("Existe", "✅" if os.path.exists(tmp) else "❌")
    os.unlink(tmp)
    dim("(archivo temporal eliminado)")

    # ============================================
    # FINAL
    # ============================================
    consola.print()
    panel(
        "[bold green]RL BÁSICO FUNCIONANDO[/bold green]\n"
        f"[dim]{entrenador}[/dim]",
        titulo="✅ Éxito",
        color="green"
    )

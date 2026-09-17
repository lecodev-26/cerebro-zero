"""
Reproducibilidad 3.0 — Snapshots verificables de experimentos
================================================================

Filosofía:
    Todo experimento debe poder decir "con esta config exacta,
    en este entorno exacto, salió este resultado".

Componentes:
- EntornoCaptura: qué máquina, qué versiones, qué git commit
- HashConfig: hash determinista de cualquier estructura
- RegistroExperimento: snapshot completo
- Reproducibilidad: gestor central
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hashlib
import json
import platform
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


# ============================================
# HASH UTILITIES
# ============================================

def hash_estructura(obj: Any) -> str:
    """
    Hash SHA256 determinista de cualquier estructura serializable.
    
    - Ordena claves de dicts para determinismo
    - Soporta int, float, str, bool, None, list, dict, tuple
    - Ignora objetos no serializables (usa repr)
    """
    def normalizar(o):
        if o is None:
            return None
        if isinstance(o, (int, float, str, bool)):
            return o
        if isinstance(o, (list, tuple)):
            return [normalizar(x) for x in o]
        if isinstance(o, dict):
            return {str(k): normalizar(v) for k, v in sorted(o.items())}
        return repr(o)
    
    try:
        json_str = json.dumps(normalizar(obj), sort_keys=True, default=str)
    except Exception:
        json_str = repr(obj)
    
    return hashlib.sha256(json_str.encode("utf-8")).hexdigest()


def hash_archivo(archivo: str, chunk_size: int = 65536) -> Optional[str]:
    """Hash SHA256 de un archivo"""
    if not os.path.exists(archivo):
        return None
    h = hashlib.sha256()
    try:
        with open(archivo, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


# ============================================
# CAPTURA DEL ENTORNO
# ============================================

@dataclass
class EntornoCaptura:
    """Snapshot del entorno de ejecución"""
    python_version: str = ""
    numpy_version: str = ""
    plataforma: str = ""
    arquitectura: str = ""
    hardware: str = ""
    git_commit: str = ""
    git_branch: str = ""
    git_dirty: bool = False
    cpu_count: int = 0
    timestamp: float = field(default_factory=time.time)
    seed: int = 42
    
    def to_dict(self) -> dict:
        return {
            "python_version": self.python_version,
            "numpy_version": self.numpy_version,
            "plataforma": self.plataforma,
            "arquitectura": self.arquitectura,
            "hardware": self.hardware,
            "git_commit": self.git_commit,
            "git_branch": self.git_branch,
            "git_dirty": self.git_dirty,
            "cpu_count": self.cpu_count,
            "timestamp": self.timestamp,
            "seed": self.seed,
        }
    
    @property
    def fingerprint(self) -> str:
        """Hash del entorno (sin timestamp)"""
        d = self.to_dict()
        d.pop("timestamp", None)
        return hash_estructura(d)[:16]
    
    @classmethod
    def capturar(
        cls,
        seed: int = 42,
        repo_path: Optional[str] = None,
    ) -> "EntornoCaptura":
        """Captura el entorno actual"""
        env = cls(seed=seed)
        
        # Python
        env.python_version = platform.python_version()
        
        # NumPy
        try:
            import numpy as np
            env.numpy_version = np.__version__
        except ImportError:
            env.numpy_version = "no instalado"
        
        # Plataforma
        env.plataforma = platform.platform()
        env.arquitectura = platform.machine()
        
        # Hardware
        try:
            env.hardware = platform.processor() or platform.node() or "desconocido"
        except Exception:
            env.hardware = "desconocido"
        
        # CPU
        try:
            env.cpu_count = os.cpu_count() or 0
        except Exception:
            env.cpu_count = 0
        
        # Git
        if repo_path is None:
            repo_path = os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )
        env._capturar_git(repo_path)
        
        return env
    
    def _capturar_git(self, repo_path: str):
        """Captura info de git"""
        try:
            r = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_path, capture_output=True, text=True, timeout=2,
            )
            if r.returncode == 0:
                self.git_commit = r.stdout.strip()
            
            r = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=repo_path, capture_output=True, text=True, timeout=2,
            )
            if r.returncode == 0:
                self.git_branch = r.stdout.strip()
            
            r = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_path, capture_output=True, text=True, timeout=2,
            )
            if r.returncode == 0:
                self.git_dirty = bool(r.stdout.strip())
        except Exception:
            pass
    
    def __repr__(self):
        return (
            f"EntornoCaptura(py={self.python_version}, "
            f"numpy={self.numpy_version}, "
            f"git={self.git_commit[:7] if self.git_commit else 'none'})"
        )


# ============================================
# REGISTRO DE EXPERIMENTO
# ============================================

@dataclass
class RegistroExperimento:
    """Snapshot completo de un experimento"""
    nombre: str
    entorno: EntornoCaptura
    config: dict = field(default_factory=dict)
    config_hash: str = ""
    dataset_hashes: Dict[str, str] = field(default_factory=dict)
    model_hashes: Dict[str, str] = field(default_factory=dict)
    resultado: dict = field(default_factory=dict)
    notas: str = ""
    creado: float = field(default_factory=time.time)
    
    def __post_init__(self):
        if not self.config_hash:
            self.config_hash = hash_estructura(self.config)
    
    def añadir_dataset(self, nombre: str, ruta: str):
        """Registra el hash de un dataset"""
        h = hash_archivo(ruta)
        if h is None:
            raise FileNotFoundError(f"Dataset no encontrado: {ruta}")
        self.dataset_hashes[nombre] = h
    
    def añadir_modelo(self, nombre: str, ruta: str):
        """Registra el hash de un modelo"""
        h = hash_archivo(ruta)
        if h is None:
            raise FileNotFoundError(f"Modelo no encontrado: {ruta}")
        self.model_hashes[nombre] = h
    
    @property
    def fingerprint(self) -> str:
        """
        Fingerprint completo: config + entorno + datasets + modelos.
        NO incluye resultado ni timestamp.
        """
        d = {
            "config_hash": self.config_hash,
            "entorno": self.entorno.fingerprint,
            "dataset_hashes": self.dataset_hashes,
            "model_hashes": self.model_hashes,
        }
        return hash_estructura(d)[:16]
    
    def to_dict(self) -> dict:
        return {
            "nombre": self.nombre,
            "fingerprint": self.fingerprint,
            "entorno": self.entorno.to_dict(),
            "config": self.config,
            "config_hash": self.config_hash,
            "dataset_hashes": self.dataset_hashes,
            "model_hashes": self.model_hashes,
            "resultado": self.resultado,
            "notas": self.notas,
            "creado": self.creado,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "RegistroExperimento":
        entorno_d = d.get("entorno", {})
        entorno = EntornoCaptura(
            python_version=entorno_d.get("python_version", ""),
            numpy_version=entorno_d.get("numpy_version", ""),
            plataforma=entorno_d.get("plataforma", ""),
            arquitectura=entorno_d.get("arquitectura", ""),
            hardware=entorno_d.get("hardware", ""),
            git_commit=entorno_d.get("git_commit", ""),
            git_branch=entorno_d.get("git_branch", ""),
            git_dirty=entorno_d.get("git_dirty", False),
            cpu_count=entorno_d.get("cpu_count", 0),
            timestamp=entorno_d.get("timestamp", 0),
            seed=entorno_d.get("seed", 42),
        )
        
        return cls(
            nombre=d["nombre"],
            entorno=entorno,
            config=d.get("config", {}),
            config_hash=d.get("config_hash", ""),
            dataset_hashes=d.get("dataset_hashes", {}),
            model_hashes=d.get("model_hashes", {}),
            resultado=d.get("resultado", {}),
            notas=d.get("notas", ""),
            creado=d.get("creado", 0),
        )
    
    def __repr__(self):
        return f"RegistroExperimento('{self.nombre}', fp={self.fingerprint})"


# ============================================
# GESTOR DE REPRODUCIBILIDAD
# ============================================

class Reproducibilidad:
    """
    Gestor central de reproducibilidad.
    
    Uso:
        repro = Reproducibilidad()
        reg = repro.nuevo_experimento("mi_experimento", config={...})
        reg.resultado = {"accuracy": 0.95}
        repro.guardar(reg)
        
        # Después, comparar dos runs
        repro.comparar(reg1, reg2)
    """
    
    def __init__(self, directorio: str = None):
        self.directorio = directorio or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "experiments", "reproducibilidad",
        )
        os.makedirs(self.directorio, exist_ok=True)
        self.registros: List[RegistroExperimento] = []
    
    def nuevo_experimento(
        self,
        nombre: str,
        config: Optional[dict] = None,
        seed: int = 42,
        notas: str = "",
    ) -> RegistroExperimento:
        """Crea un nuevo registro con entorno capturado"""
        entorno = EntornoCaptura.capturar(seed=seed)
        reg = RegistroExperimento(
            nombre=nombre,
            entorno=entorno,
            config=config or {},
            notas=notas,
        )
        return reg
    
    def guardar(self, reg: RegistroExperimento) -> str:
        """Guarda el registro en JSON con nombre único"""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        archivo = os.path.join(
            self.directorio,
            f"{reg.nombre}_{reg.fingerprint}_{ts}.json",
        )
        with open(archivo, "w") as f:
            json.dump(reg.to_dict(), f, indent=2, default=str)
        self.registros.append(reg)
        return archivo
    
    def cargar(self, archivo: str) -> Optional[RegistroExperimento]:
        if not os.path.exists(archivo):
            return None
        try:
            with open(archivo, "r") as f:
                data = json.load(f)
            reg = RegistroExperimento.from_dict(data)
            return reg
        except Exception:
            return None
    
    def listar(self) -> List[str]:
        """Lista archivos en el directorio"""
        try:
            return sorted([
                f for f in os.listdir(self.directorio)
                if f.endswith(".json")
            ])
        except Exception:
            return []
    
    # ============================================
    # COMPARACIÓN
    # ============================================
    
    def comparar(
        self,
        reg1: RegistroExperimento,
        reg2: RegistroExperimento,
    ) -> dict:
        """
        Compara dos experimentos.
        Devuelve dict con diferencias.
        """
        iguales = reg1.fingerprint == reg2.fingerprint
        
        diferencias = {}
        
        # Config
        if reg1.config_hash != reg2.config_hash:
            diferencias["config"] = {
                "reg1": reg1.config_hash[:12],
                "reg2": reg2.config_hash[:12],
            }
        
        # Entorno
        if reg1.entorno.fingerprint != reg2.entorno.fingerprint:
            diferencias["entorno"] = {
                "python": (reg1.entorno.python_version,
                           reg2.entorno.python_version),
                "numpy": (reg1.entorno.numpy_version,
                          reg2.entorno.numpy_version),
                "git": (reg1.entorno.git_commit[:7],
                        reg2.entorno.git_commit[:7]),
            }
        
        # Datasets
        if reg1.dataset_hashes != reg2.dataset_hashes:
            diferencias["datasets"] = {
                "reg1": reg1.dataset_hashes,
                "reg2": reg2.dataset_hashes,
            }
        
        # Modelos
        if reg1.model_hashes != reg2.model_hashes:
            diferencias["modelos"] = {
                "reg1": reg1.model_hashes,
                "reg2": reg2.model_hashes,
            }
        
        # Resultados (pueden diferir sin invalidar reproducibilidad)
        resultados_difieren = reg1.resultado != reg2.resultado
        
        return {
            "identicos": iguales,
            "diferencias": diferencias,
            "resultados_difieren": resultados_difieren,
            "fp1": reg1.fingerprint,
            "fp2": reg2.fingerprint,
        }
    
    def verificar_reproducible(
        self,
        reg1: RegistroExperimento,
        reg2: RegistroExperimento,
        tolerancia: float = 1e-6,
    ) -> dict:
        """
        Verifica que dos runs con MISMO fingerprint dan resultados similares.
        """
        mismo_fp = reg1.fingerprint == reg2.fingerprint
        
        if not mismo_fp:
            return {
                "reproducible": False,
                "razon": "Fingerprints distintos",
                "diferencias": self.comparar(reg1, reg2)["diferencias"],
            }
        
        # Comparar resultados numéricos
        diff_max = 0.0
        campos_diferentes = []
        
        for k, v1 in reg1.resultado.items():
            v2 = reg2.resultado.get(k)
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                d = abs(v1 - v2)
                diff_max = max(diff_max, d)
                if d > tolerancia:
                    campos_diferentes.append(k)
            elif v1 != v2:
                campos_diferentes.append(k)
        
        return {
            "reproducible": len(campos_diferentes) == 0,
            "razon": (
                "Resultados idénticos"
                if not campos_diferentes
                else f"Campos difieren: {campos_diferentes}"
            ),
            "diff_max": diff_max,
            "campos_diferentes": campos_diferentes,
        }
    
    # ============================================
    # UTILIDADES
    # ============================================
    
    def resumen(self) -> dict:
        return {
            "directorio": self.directorio,
            "registros_en_memoria": len(self.registros),
            "archivos_guardados": len(self.listar()),
        }
    
    def __repr__(self):
        return f"Reproducibilidad(dir='{self.directorio}', n={len(self.listar())})"


# ============================================
# TEST MANUAL
# ============================================

if __name__ == "__main__":
    print("🧪 PROBANDO REPRODUCIBILIDAD 3.0")
    print("=" * 60)
    
    # ============================================
    # HASH UTILITIES
    # ============================================
    print("\n🔐 Test hashes:")
    h1 = hash_estructura({"a": 1, "b": [1, 2, 3]})
    h2 = hash_estructura({"b": [1, 2, 3], "a": 1})  # orden distinto
    h3 = hash_estructura({"a": 2, "b": [1, 2, 3]})
    
    print(f"   dict A:       {h1[:16]}")
    print(f"   dict A reord: {h2[:16]}  {'✅ igual' if h1 == h2 else '❌ difiere'}")
    print(f"   dict B:       {h3[:16]}  {'✅ difiere' if h1 != h3 else '❌ igual'}")
    
    # ============================================
    # ENTORNO
    # ============================================
    print("\n🌍 Capturando entorno:")
    env = EntornoCaptura.capturar(seed=42)
    print(f"   {env}")
    print(f"   Python:    {env.python_version}")
    print(f"   NumPy:     {env.numpy_version}")
    print(f"   Plataforma: {env.plataforma}")
    print(f"   Hardware:  {env.hardware}")
    print(f"   CPU cores: {env.cpu_count}")
    print(f"   Git commit: {env.git_commit[:12] if env.git_commit else 'N/A'}")
    print(f"   Git branch: {env.git_branch}")
    print(f"   Git dirty: {env.git_dirty}")
    print(f"   Fingerprint: {env.fingerprint}")
    
    # ============================================
    # EXPERIMENTO
    # ============================================
    print("\n🧬 Creando experimento:")
    repro = Reproducibilidad()
    
    config = {
        "modelo": "transformer",
        "capas": 2,
        "heads": 4,
        "lr": 0.001,
        "epochs": 5,
    }
    
    reg = repro.nuevo_experimento(
        nombre="test_reproducibilidad",
        config=config,
        seed=42,
        notas="Test manual de reproducibilidad",
    )
    
    print(f"   {reg}")
    print(f"   Config hash: {reg.config_hash[:16]}")
    
    # Resultados
    reg.resultado = {
        "train_loss": 0.42,
        "val_loss": 0.51,
        "accuracy": 0.873,
        "epochs_ejecutados": 5,
    }
    
    print(f"   Resultado: {reg.resultado}")
    
    # ============================================
    # GUARDAR / CARGAR
    # ============================================
    print("\n💾 Test persistencia:")
    archivo = repro.guardar(reg)
    print(f"   Guardado: {os.path.basename(archivo)}")
    
    reg2 = repro.cargar(archivo)
    print(f"   Cargado:  {reg2}")
    print(f"   Mismo fingerprint: {reg.fingerprint == reg2.fingerprint}")
    
    # Limpiar
    os.unlink(archivo)
    
    # ============================================
    # COMPARAR
    # ============================================
    print("\n⚖️ Comparando dos experimentos:")
    
    regA = repro.nuevo_experimento("A", config=config, seed=42)
    regA.resultado = {"accuracy": 0.9}
    
    regB = repro.nuevo_experimento("B", config=config, seed=42)
    regB.resultado = {"accuracy": 0.91}  # mismo fingerprint, distinto resultado
    
    comp = repro.comparar(regA, regB)
    print(f"   Fingerprints iguales: {comp['identicos']}")
    print(f"   Resultados difieren: {comp['resultados_difieren']}")
    
    verif = repro.verificar_reproducible(regA, regB)
    print(f"   Reproducible: {verif['reproducible']}")
    print(f"   Diff max: {verif.get('diff_max', 0):.4f}")
    
    # ============================================
    # COMPARAR CON CONFIG DISTINTA
    # ============================================
    print("\n⚖️ Comparando con config distinta:")
    
    config2 = config.copy()
    config2["lr"] = 0.01
    
    regC = repro.nuevo_experimento("C", config=config2, seed=42)
    regC.resultado = {"accuracy": 0.85}
    
    comp2 = repro.comparar(regA, regC)
    print(f"   Fingerprints iguales: {comp2['identicos']}")
    print(f"   Diferencias: {list(comp2['diferencias'].keys())}")
    
    # ============================================
    # RESUMEN
    # ============================================
    print(f"\n📊 Resumen:")
    print(f"   {repro}")
    
    # Limpiar directorio
    for f in os.listdir(repro.directorio):
        try:
            os.unlink(os.path.join(repro.directorio, f))
        except Exception:
            pass
    
    print("\n✅ REPRODUCIBILIDAD 3.0 FUNCIONANDO")

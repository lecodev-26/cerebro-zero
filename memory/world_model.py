"""
World Model — Estado persistente del mundo con rutas anidadas
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional
from datetime import datetime


@dataclass
class Cambio:
    """Un cambio registrado en el mundo"""
    ruta: str
    valor_anterior: Any
    valor_nuevo: Any
    timestamp: float = field(default_factory=time.time)
    razon: str = ""
    
    def to_dict(self) -> dict:
        return {
            'ruta': self.ruta,
            'valor_anterior': str(self.valor_anterior)[:100],
            'valor_nuevo': str(self.valor_nuevo)[:100],
            'razon': self.razon,
            'timestamp': self.timestamp,
        }


class WorldModel:
    """
    Modelo del mundo con:
    - Rutas anidadas ("user.nombre")
    - Persistencia JSON
    - Historial de cambios
    - Suscripciones
    """
    
    def __init__(self, archivo: str = None):
        self.archivo = archivo or "modelos_guardados/world_model.json"
        self.estado: dict = {}
        self.historial: List[Cambio] = []
        self.suscritos: dict = {}  # ruta -> lista de callbacks
        self.cargar()
    
    # ============================================
    # SET / GET / UPDATE / DELETE
    # ============================================
    
    def set(self, ruta: str, valor: Any, razon: str = "") -> bool:
        """
        Establece un valor en una ruta.
        
        Uso:
            world.set("user.nombre", "Manuel")
            world.set("proyecto.status", "training")
        """
        try:
            anterior = self.get(ruta)
            
            # Navegar hasta el padre y crear estructura si no existe
            partes = ruta.split('.')
            actual = self.estado
            
            for parte in partes[:-1]:
                if parte not in actual or not isinstance(actual[parte], dict):
                    actual[parte] = {}
                actual = actual[parte]
            
            actual[partes[-1]] = valor
            
            # Registrar cambio
            cambio = Cambio(
                ruta=ruta,
                valor_anterior=anterior,
                valor_nuevo=valor,
                razon=razon,
            )
            self.historial.append(cambio)
            if len(self.historial) > 200:
                self.historial = self.historial[-200:]
            
            # Notificar suscritos
            self._notificar(ruta, anterior, valor)
            
            self.guardar()
            return True
        
        except Exception as e:
            print(f"⚠️ Error en set: {e}")
            return False
    
    def get(self, ruta: str, default: Any = None) -> Any:
        """
        Obtiene un valor de una ruta.
        
        Uso:
            world.get("user.nombre")  # → "Manuel"
            world.get("user.edad", 0)  # → 0 si no existe
        """
        partes = ruta.split('.')
        actual = self.estado
        
        for parte in partes:
            if isinstance(actual, dict) and parte in actual:
                actual = actual[parte]
            else:
                return default
        
        return actual
    
    def update(self, ruta: str, valor: Any, razon: str = "") -> bool:
        """Alias de set() para claridad"""
        return self.set(ruta, valor, razon)
    
    def delete(self, ruta: str) -> bool:
        """Elimina una clave en la ruta"""
        partes = ruta.split('.')
        actual = self.estado
        
        try:
            for parte in partes[:-1]:
                if parte not in actual:
                    return False
                actual = actual[parte]
            
            if partes[-1] in actual:
                anterior = actual[partes[-1]]
                del actual[partes[-1]]
                
                cambio = Cambio(
                    ruta=ruta,
                    valor_anterior=anterior,
                    valor_nuevo=None,
                    razon="delete",
                )
                self.historial.append(cambio)
                self._notificar(ruta, anterior, None)
                self.guardar()
                return True
            return False
        except Exception:
            return False
    
    def existe(self, ruta: str) -> bool:
        """Comprueba si existe una ruta"""
        return self.get(ruta, "__NONE__") != "__NONE__"
    
    def incrementar(self, ruta: str, delta: float = 1) -> float:
        """Incrementa un valor numérico"""
        actual = self.get(ruta, 0)
        nuevo = actual + delta
        self.set(ruta, nuevo, razon=f"+{delta}")
        return nuevo
    
    def append(self, ruta: str, item: Any) -> list:
        """Añade a una lista"""
        actual = self.get(ruta, [])
        if not isinstance(actual, list):
            actual = []
        nuevo = actual + [item]
        self.set(ruta, nuevo, razon="append")
        return nuevo
    
    # ============================================
    # SUSCRIPCIONES
    # ============================================
    
    def suscribir(self, ruta: str, callback: Callable):
        """
        Suscribe un callback a cambios en una ruta.
        
        Uso:
            def on_change(ruta, viejo, nuevo):
                print(f"{ruta}: {viejo} → {nuevo}")
            world.suscribir("user.nombre", on_change)
        """
        if ruta not in self.suscritos:
            self.suscritos[ruta] = []
        self.suscritos[ruta].append(callback)
    
    def _notificar(self, ruta: str, viejo: Any, nuevo: Any):
        """Notifica a los suscritos de la ruta y de rutas padre"""
        # Notificar suscritos exactos
        if ruta in self.suscritos:
            for cb in self.suscritos[ruta]:
                try:
                    cb(ruta, viejo, nuevo)
                except Exception:
                    pass
        
        # Notificar suscritos a rutas padre
        partes = ruta.split('.')
        for i in range(1, len(partes)):
            padre = '.'.join(partes[:i])
            if padre in self.suscritos:
                for cb in self.suscritos[padre]:
                    try:
                        cb(ruta, viejo, nuevo)
                    except Exception:
                        pass
    
    # ============================================
    # PERSISTENCIA
    # ============================================
    
    def guardar(self):
        """Guarda el estado en JSON"""
        try:
            os.makedirs(os.path.dirname(self.archivo) or ".", exist_ok=True)
            data = {
                'estado': self.estado,
                'historial': [c.to_dict() for c in self.historial[-50:]],
                'guardado': datetime.now().isoformat(),
            }
            with open(self.archivo, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            print(f"⚠️ Error guardando world model: {e}")
    
    def cargar(self):
        """Carga el estado desde JSON"""
        try:
            if os.path.exists(self.archivo):
                with open(self.archivo, 'r') as f:
                    data = json.load(f)
                self.estado = data.get('estado', {})
        except Exception:
            self.estado = {}
    
    # ============================================
    # UTILIDADES
    # ============================================
    
    def clear(self):
        """Limpia todo el mundo"""
        self.estado = {}
        self.historial = []
        self.guardar()
    
    def keys(self) -> List[str]:
        """Lista las claves del primer nivel"""
        return list(self.estado.keys())
    
    def get_historial(self, n: int = 10) -> List[dict]:
        """Últimos cambios"""
        return [c.to_dict() for c in self.historial[-n:]]
    
    def to_dict(self) -> dict:
        return self.estado.copy()
    
    def resumen(self) -> dict:
        return {
            'keys': self.keys(),
            'total_claves': len(self.historial),
            'archivo': self.archivo,
            'estado': self.estado,
        }
    
    def __repr__(self):
        return f"WorldModel(keys={self.keys()})"


# ============================================
# TEST
# ============================================

if __name__ == "__main__":
    print("🧪 PROBANDO WORLD MODEL")
    print("="*60)
    
    # Usar archivo temporal para el test
    import tempfile
    tmpfile = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
    
    wm = WorldModel(archivo=tmpfile)
    wm.clear()
    
    # SET
    print("\n📝 Estableciendo valores:")
    wm.set("user.nombre", "Manuel", razon="inicial")
    wm.set("user.edad", 25)
    wm.set("proyecto.nombre", "Cerebro Zero")
    wm.set("proyecto.version", "3.0")
    wm.set("proyecto.status", "desarrollo")
    wm.set("entorno.cwd", "/home/cerebro_zero")
    
    print(f"   {wm}")
    
    # GET
    print(f"\n🔍 Obtener valores:")
    print(f"   user.nombre = {wm.get('user.nombre')}")
    print(f"   user.edad = {wm.get('user.edad')}")
    print(f"   proyecto.nombre = {wm.get('proyecto.nombre')}")
    print(f"   proyecto.version = {wm.get('proyecto.version')}")
    print(f"   no.existe = {wm.get('no.existe', 'DEFAULT')}")
    
    # UPDATE
    print(f"\n✏️ Actualizar:")
    wm.update("proyecto.status", "test", razon="cambio de fase")
    print(f"   proyecto.status = {wm.get('proyecto.status')}")
    
    # INCREMENT
    print(f"\n➕ Incrementar:")
    wm.set("contador.epochs", 0)
    wm.incrementar("contador.epochs")
    wm.incrementar("contador.epochs", 5)
    print(f"   contador.epochs = {wm.get('contador.epochs')}")
    
    # APPEND
    print(f"\n📎 Append:")
    wm.set("logs", [])
    wm.append("logs", "Primer mensaje")
    wm.append("logs", "Segundo mensaje")
    print(f"   logs = {wm.get('logs')}")
    
    # EXISTE
    print(f"\n❓ Existe:")
    print(f"   user.nombre existe? {wm.existe('user.nombre')}")
    print(f"   no.existe existe? {wm.existe('no.existe')}")
    
    # DELETE
    print(f"\n🗑️ Delete:")
    wm.delete("user.edad")
    print(f"   user.edad = {wm.get('user.edad', 'BORRADO')}")
    
    # SUSCRIPCIÓN
    print(f"\n🔔 Suscripción:")
    cambios = []
    def on_change(ruta, viejo, nuevo):
        cambios.append((ruta, viejo, nuevo))
    
    wm.suscribir("proyecto.status", on_change)
    wm.set("proyecto.status", "finalizado")
    print(f"   Cambios capturados: {len(cambios)}")
    for c in cambios:
        print(f"   {c[0]}: {c[1]} → {c[2]}")
    
    # HISTORIAL
    print(f"\n📜 Historial (últimos 5):")
    for h in wm.get_historial(5):
        print(f"   {h['ruta']}: {h['valor_anterior'][:20]} → {h['valor_nuevo'][:20]}")
    
    # PERSISTENCIA
    print(f"\n💾 Test persistencia:")
    wm2 = WorldModel(archivo=tmpfile)
    print(f"   Cargado user.nombre = {wm2.get('user.nombre')}")
    print(f"   Cargado proyecto.nombre = {wm2.get('proyecto.nombre')}")
    
    print("\n✅ WORLD MODEL FUNCIONANDO")
    
    # Limpiar
    os.unlink(tmpfile)

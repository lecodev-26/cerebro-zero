"""
Working Memory — Memoria de trabajo con capacidad limitada y TTL
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from dataclasses import dataclass, field
from typing import Any, Optional, List
from datetime import datetime


@dataclass
class WorkingItem:
    """Un item en la memoria de trabajo"""
    key: str
    value: Any
    prioridad: float = 0.5      # 0.0 a 1.0
    ttl: float = 300.0           # segundos antes de expirar
    creado: float = field(default_factory=time.time)
    accedido: float = field(default_factory=time.time)
    veces_accedido: int = 0
    
    def edad(self) -> float:
        """Tiempo desde su creación (segundos)"""
        return time.time() - self.creado
    
    def inactividad(self) -> float:
        """Tiempo desde el último acceso"""
        return time.time() - self.accedido
    
    def esta_expirado(self) -> bool:
        """¿Ha pasado el TTL?"""
        return self.edad() > self.ttl
    
    def relevancia(self) -> float:
        """
        Calcula la relevancia actual del item.
        
        Combina:
        - Prioridad base
        - Tiempo sin acceder (decay)
        - Veces accedido (refuerzo)
        """
        # Factor de decaimiento por inactividad
        decay = max(0.0, 1.0 - self.inactividad() / self.ttl)
        
        # Bonus por veces accedido (max 1.0)
        refuerzo = min(1.0, self.veces_accedido * 0.1)
        
        return (self.prioridad * 0.5 + decay * 0.3 + refuerzo * 0.2)
    
    def to_dict(self) -> dict:
        return {
            'key': self.key,
            'value': str(self.value)[:50],
            'prioridad': self.prioridad,
            'ttl': self.ttl,
            'edad': self.edad(),
            'veces_accedido': self.veces_accedido,
            'relevancia': self.relevancia(),
        }


class WorkingMemory:
    """
    Memoria de trabajo con:
    - Capacidad limitada
    - Prioridades
    - TTL (time-to-live)
    - Decay (olvido gradual)
    - Recuperación por relevancia
    """
    
    def __init__(self, max_items: int = 10, default_ttl: float = 300.0):
        self.max_items = max_items
        self.default_ttl = default_ttl
        self.items = {}  # key -> WorkingItem
        self.historial = []  # últimos items expulsados
    
    def add(self, key: str, value: Any, prioridad: float = 0.5,
            ttl: float = None) -> bool:
        """
        Añade o actualiza un item.
        
        Returns:
            True si se añadió, False si fue rechazado.
        """
        if ttl is None:
            ttl = self.default_ttl
        
        # Si ya existe, actualizar
        if key in self.items:
            item = self.items[key]
            item.value = value
            item.prioridad = prioridad
            item.ttl = ttl
            item.accedido = time.time()
            return True
        
        # Si está lleno, expulsar el menos relevante
        if len(self.items) >= self.max_items:
            self._expulsar_menos_relevante()
        
        # Añadir nuevo
        item = WorkingItem(key=key, value=value, prioridad=prioridad, ttl=ttl)
        self.items[key] = item
        return True
    
    def get(self, key: str) -> Optional[Any]:
        """Recupera un item y actualiza su relevancia"""
        if key not in self.items:
            return None
        
        item = self.items[key]
        
        # Si está expirado, eliminarlo
        if item.esta_expirado():
            del self.items[key]
            return None
        
        # Actualizar acceso
        item.accedido = time.time()
        item.veces_accedido += 1
        
        return item.value
    
    def get_relevante(self, query: str, k: int = 3) -> List[WorkingItem]:
        """
        Devuelve los k items más relevantes cuya key contenga la query.
        """
        self.tick()  # Limpiar expirados primero
        
        # Filtrar por query
        matching = [
            item for item in self.items.values()
            if query.lower() in item.key.lower()
        ]
        
        # Ordenar por relevancia
        matching.sort(key=lambda x: x.relevancia(), reverse=True)
        
        return matching[:k]
    
    def get_top_relevantes(self, k: int = 5) -> List[WorkingItem]:
        """Devuelve los k items más relevantes"""
        self.tick()
        items = sorted(self.items.values(), 
                       key=lambda x: x.relevancia(), reverse=True)
        return items[:k]
    
    def tick(self):
        """
        Avanza el tiempo de la memoria.
        - Elimina items expirados
        - Reduce relevancia de items no accedidos
        """
        expirados = [k for k, item in self.items.items() if item.esta_expirado()]
        for k in expirados:
            self.historial.append(self.items[k])
            del self.items[k]
        
        # Limitar historial
        if len(self.historial) > 50:
            self.historial = self.historial[-50:]
    
    def _expulsar_menos_relevante(self):
        """Expulsa el item menos relevante"""
        if not self.items:
            return
        
        # Encontrar el menos relevante
        menos = min(self.items.values(), key=lambda x: x.relevancia())
        self.historial.append(menos)
        del self.items[menos.key]
    
    def clear(self):
        """Limpia toda la memoria"""
        self.items.clear()
    
    def size(self) -> int:
        return len(self.items)
    
    def is_full(self) -> bool:
        return len(self.items) >= self.max_items
    
    def keys(self) -> List[str]:
        return list(self.items.keys())
    
    def resumen(self) -> dict:
        """Resumen del estado actual"""
        self.tick()
        return {
            'size': self.size(),
            'max_items': self.max_items,
            'items': [item.to_dict() for item in 
                     sorted(self.items.values(), 
                            key=lambda x: x.relevancia(), 
                            reverse=True)],
        }
    
    def __repr__(self):
        return f"WorkingMemory({self.size()}/{self.max_items})"


# ============================================
# TEST
# ============================================

if __name__ == "__main__":
    print("🧪 PROBANDO WORKING MEMORY")
    print("="*60)
    
    wm = WorkingMemory(max_items=5, default_ttl=2.0)
    
    # Añadir items
    print("\n📝 Añadiendo items:")
    wm.add("objetivo", "entrenar modelo", prioridad=1.0, ttl=10.0)
    wm.add("dataset", "train.txt", prioridad=0.8, ttl=10.0)
    wm.add("epochs", 10, prioridad=0.5, ttl=10.0)
    wm.add("lr", 0.001, prioridad=0.5, ttl=10.0)
    wm.add("modelo", "transformer", prioridad=0.7, ttl=10.0)
    
    print(f"   {wm}")
    print(f"   Keys: {wm.keys()}")
    
    # Recuperar
    print(f"\n🔍 Recuperar 'objetivo': {wm.get('objetivo')}")
    print(f"🔍 Recuperar 'dataset': {wm.get('dataset')}")
    print(f"🔍 Recuperar 'no_existe': {wm.get('no_existe')}")
    
    # Añadir uno más → debe expulsar el menos relevante
    print(f"\n📝 Añadir 'nuevo' (capacidad 5/5):")
    wm.add("nuevo", "valor", prioridad=0.9, ttl=10.0)
    print(f"   {wm}")
    print(f"   Keys: {wm.keys()}")
    
    # Búsqueda por relevancia
    print(f"\n🎯 Top 3 relevantes:")
    for item in wm.get_top_relevantes(k=3):
        print(f"   {item.key} = {item.value} (rel={item.relevancia():.3f})")
    
    # Búsqueda por query
    print(f"\n🔍 Buscar 'dataset':")
    for item in wm.get_relevante("dataset"):
        print(f"   {item.key} = {item.value}")
    
    # Test TTL
    print(f"\n⏱️ Test TTL (creando memoria con TTL=1s):")
    wm2 = WorkingMemory(max_items=3, default_ttl=1.0)
    wm2.add("temp1", "valor1")
    wm2.add("temp2", "valor2")
    print(f"   Antes: {wm2.size()} items")
    time.sleep(1.5)
    wm2.tick()
    print(f"   Después de 1.5s: {wm2.size()} items (expirados)")
    
    # Resumen
    print(f"\n📊 Resumen de wm:")
    resumen = wm.resumen()
    print(f"   Size: {resumen['size']}/{resumen['max_items']}")
    for item in resumen['items']:
        print(f"   {item['key']} = {item['value']} (rel={item['relevancia']:.3f})")
    
    print("\n✅ WORKING MEMORY FUNCIONANDO")

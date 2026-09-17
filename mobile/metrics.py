"""
Medición de recursos en móvil
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np


class ResourceMonitor:
    """
    Monitor de recursos para móvil.
    
    Uso:
        monitor = ResourceMonitor()
        with monitor.measure("forward"):
            model.forward(x)
        monitor.print_report()
    """
    
    def __init__(self):
        self.measurements = []
        self.active = False
        self.start_time = None
        self.start_label = None
    
    def measure(self, label: str):
        """
        Context manager para medir el tiempo.
        
        Uso:
            with monitor.measure("forward"):
                # código a medir
        """
        return self._MeasureContext(self, label)
    
    class _MeasureContext:
        def __init__(self, monitor, label):
            self.monitor = monitor
            self.label = label
        
        def __enter__(self):
            self.start = time.time()
            return self
        
        def __exit__(self, *args):
            elapsed = (time.time() - self.start) * 1000  # ms
            self.monitor.measurements.append({
                'label': self.label,
                'time_ms': elapsed,
            })
    
    def get_ram_usage_mb(self) -> float:
        """
        Estima el uso de RAM en MB (aproximado).
        """
        try:
            import resource
            # En Termux/Android puede no funcionar igual
            usage = resource.getrusage(resource.RUSAGE_SELF)
            return usage.ru_maxrss / 1024  # KB a MB (Linux)
        except:
            return 0.0
    
    def get_stats(self) -> dict:
        """Devuelve estadísticas agregadas"""
        if not self.measurements:
            return {'count': 0}
        
        times = [m['time_ms'] for m in self.measurements]
        return {
            'count': len(times),
            'total_ms': sum(times),
            'avg_ms': np.mean(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
        }
    
    def print_report(self):
        """Imprime un reporte de las mediciones"""
        print("\n⏱️ RESOURCE MONITOR REPORT")
        print("="*60)
        
        # Agrupar por label
        by_label = {}
        for m in self.measurements:
            label = m['label']
            if label not in by_label:
                by_label[label] = []
            by_label[label].append(m['time_ms'])
        
        for label, times in by_label.items():
            print(f"\n{label}:")
            print(f"   Llamadas: {len(times)}")
            print(f"   Total:    {sum(times):.2f}ms")
            print(f"   Media:    {np.mean(times):.2f}ms")
            print(f"   Mín/Máx:  {np.min(times):.2f}/{np.max(times):.2f}ms")
        
        print("\n" + "="*60)
        print(f"RAM usage: {self.get_ram_usage_mb():.2f} MB")
        print("="*60)
    
    def clear(self):
        """Limpia las mediciones"""
        self.measurements.clear()
    
    def __repr__(self):
        return f"ResourceMonitor(measurements={len(self.measurements)})"


if __name__ == "__main__":
    print("🧪 PROBANDO RESOURCE MONITOR")
    print("="*50)
    
    monitor = ResourceMonitor()
    
    # Medir operaciones
    with monitor.measure("suma"):
        a = np.random.randn(1000, 1000).astype(np.float32)
        b = np.random.randn(1000, 1000).astype(np.float32)
        c = a + b
    
    with monitor.measure("matmul"):
        a = np.random.randn(500, 500).astype(np.float32)
        b = np.random.randn(500, 500).astype(np.float32)
        c = np.dot(a, b)
    
    with monitor.measure("matmul"):
        a = np.random.randn(500, 500).astype(np.float32)
        b = np.random.randn(500, 500).astype(np.float32)
        c = np.dot(a, b)
    
    monitor.print_report()
    
    print("\n✅ RESOURCE MONITOR FUNCIONANDO")

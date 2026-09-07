import numpy as np

class Visualizador:
    @staticmethod
    def graficar_texto(X, y, predicciones=None, ancho=40, alto=10):
        if len(X.shape) > 1:
            X = X.flatten()
        if len(y.shape) > 1:
            y = y.flatten()
        
        x_min, x_max = X.min(), X.max()
        y_min, y_max = y.min(), y.max()
        
        if y_max - y_min < 0.001:
            print("⚠️ Datos sin variación")
            return
        
        grid = [[' ' for _ in range(ancho)] for _ in range(alto)]
        
        for i in range(len(X)):
            x_idx = int((X[i] - x_min) / (x_max - x_min) * (ancho-1))
            y_idx = int((y[i] - y_min) / (y_max - y_min) * (alto-1))
            y_idx = alto - 1 - y_idx
            
            if 0 <= x_idx < ancho and 0 <= y_idx < alto:
                if predicciones is not None and len(predicciones) > i:
                    pred = predicciones[i]
                    grid[y_idx][x_idx] = '●' if pred > 0.5 else '○'
                else:
                    grid[y_idx][x_idx] = '•'
        
        print("\n📈 GRÁFICO DE DATOS:")
        print("-" * (ancho + 4))
        for fila in grid:
            print("| " + ''.join(fila) + " |")
        print("-" * (ancho + 4))
        print(f"   Mínimo: {y_min:.2f}   Máximo: {y_max:.2f}")
    
    @staticmethod
    def mostrar_pesos(cerebro, capa_idx=0):
        if capa_idx >= len(cerebro.pesos):
            print(f"⚠️ Solo hay {len(cerebro.pesos)} capas")
            return
        
        pesos = cerebro.pesos[capa_idx]
        print(f"\n🧬 PESOS CAPA {capa_idx+1} ({pesos.shape[0]}→{pesos.shape[1]}):")
        
        for i in range(min(10, pesos.shape[0])):
            fila = pesos[i][:10]
            texto = ' '.join([f"{v:6.2f}" for v in fila])
            print(f"   {texto}")
        
        if pesos.shape[0] > 10 or pesos.shape[1] > 10:
            print(f"   ... (y {pesos.size - 100} más)")

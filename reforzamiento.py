import numpy as np
import random

class CerebroReforzado:
    def __init__(self, estados, acciones):
        self.estados = estados
        self.acciones = acciones
        self.q_table = np.zeros((estados, acciones))
        self.tasa_aprendizaje = 0.1
        self.descuento = 0.9
        self.exploracion = 0.1
    
    def elegir_accion(self, estado):
        if random.random() < self.exploracion:
            return random.randint(0, self.acciones-1)
        return np.argmax(self.q_table[estado])
    
    def aprender(self, estado, accion, recompensa, siguiente_estado):
        mejor_siguiente = np.max(self.q_table[siguiente_estado])
        actual = self.q_table[estado, accion]
        self.q_table[estado, accion] = actual + self.tasa_aprendizaje * (
            recompensa + self.descuento * mejor_siguiente - actual
        )
    
    def jugar(self, entorno, episodios=100):
        print("🎮 ENTRENANDO POR REFORZAMIENTO")
        
        for ep in range(episodios):
            estado = entorno.reset()
            total_recompensa = 0
            pasos = 0
            
            while pasos < 100:
                accion = self.elegir_accion(estado)
                nuevo_estado, recompensa, done = entorno.step(accion)
                self.aprender(estado, accion, recompensa, nuevo_estado)
                
                total_recompensa += recompensa
                estado = nuevo_estado
                pasos += 1
                
                if done:
                    break
            
            if ep % 20 == 0:
                print(f"   Episodio {ep}: recompensa = {total_recompensa:.1f}")
        
        return self.q_table

class EntornoSimple:
    def __init__(self):
        self.estado = 0
        self.meta = 10
    
    def reset(self):
        self.estado = 0
        return self.estado
    
    def step(self, accion):
        if accion == 1:
            self.estado += 1
        else:
            self.estado -= 1
        
        self.estado = max(0, min(self.estado, self.meta))
        
        if self.estado == self.meta:
            recompensa = 10
            done = True
        else:
            recompensa = -1
            done = False
        
        return self.estado, recompensa, done

if __name__ == "__main__":
    entorno = EntornoSimple()
    cerebro = CerebroReforzado(estados=11, acciones=2)
    q_table = cerebro.jugar(entorno, episodios=100)
    print("\n✅ Q-Table aprendida:")
    print(q_table[:5])

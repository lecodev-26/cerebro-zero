"""
Entrenador de Language Model con BACKPROP REAL
Usa loss.backward() y optimizer.step() — sin aproximaciones.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.serialization import safe_save_dict, safe_load_dict
import time
from core.tensor import Tensor
from core.optimizers import Adam
from models.transformer import Transformer


class LMTrainer:
    """
    Entrenador de Language Model con backprop real.
    
    Uso:
        trainer = LMTrainer(model, learning_rate=0.01)
        history = trainer.train(train_ds, val_ds, epochs=10, batch_size=8)
    """
    
    def __init__(self, model: Transformer, learning_rate: float = 0.01):
        self.model = model
        self.lr = learning_rate
        self.optimizer = Adam(model.parameters(), lr=learning_rate)
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_ppl': [],
            'val_ppl': [],
        }
        self.best_val_loss = float('inf')
        self.best_epoch = -1
        self.best_weights = None
    
    # ============================================
    # LOSS Y GRADIENTES
    # ============================================
    
    def cross_entropy_loss(self, logits: Tensor, targets: np.ndarray) -> Tensor:
        """
        Cross-entropy loss dentro del grafo.
        
        logits: Tensor (batch, seq_len, vocab_size)
        targets: np.ndarray (batch, seq_len)
        
        Loss = -mean(log(softmax(logits)[target]))
        
        Implementación:
        - Aplicar softmax sobre logits
        - Codificar targets como one-hot
        - Loss = -sum(target_onehot * log(probs))
        """
        batch, seq_len, vocab_size = logits.data.shape
        targets_flat = targets.reshape(-1).astype(int)
        batch_seq = batch * seq_len
        
        # Softmax con estabilidad numérica
        probs = logits.softmax(axis=-1)
        
        # Clip para evitar log(0)
        eps = 1e-10
        probs_clipped = probs + eps
        
        # Log de las probabilidades
        log_probs = probs_clipped.log()  # (batch, seq, vocab)
        
        # Crear one-hot de los targets
        targets_onehot = np.zeros((batch, seq_len, vocab_size), dtype=log_probs.data.dtype)
        for b in range(batch):
            for s in range(seq_len):
                targets_onehot[b, s, targets[b, s]] = 1.0
        
        # Loss = -mean(sum(target_onehot * log_probs, axis=-1))
        # Equivalente a: -sum(target_onehot * log_probs) / batch_seq
        # Usamos multiplicación elemento a elemento + sum
        
        # Multiplicar elemento a elemento
        product = log_probs * Tensor(targets_onehot, requires_grad=False)  # (batch, seq, vocab)
        
        # Sumar sobre vocab
        summed = product.sum(axis=-1)  # (batch, seq)
        
        # Negativo y media
        loss = -summed.mean()
        
        return loss
    
    # ============================================
    # TRAIN STEP
    # ============================================
    
    def train_step(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Un paso de entrenamiento con backprop REAL.
        
        X: (batch, seq_len) — tokens de entrada
        y: (batch, seq_len) — tokens objetivo
        """
        # Forward
        x_t = Tensor(X.astype(int))
        logits = self.model.forward(x_t)
        
        # Loss
        loss = self.cross_entropy_loss(logits, y)
        
        # Backward REAL (sin aproximaciones)
        loss.backward()
        
        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return float(loss.data)
    
    # ============================================
    # EVALUACIÓN
    # ============================================
    
    def evaluate(self, dataset, batch_size: int = 8, num_batches: int = 10) -> dict:
        """Evalúa el modelo sobre un dataset"""
        losses = []
        for _ in range(num_batches):
            X, y = dataset.get_batch(batch_size=batch_size, shuffle=True)
            x_t = Tensor(X.astype(int))
            logits = self.model.forward(x_t)
            loss = self.cross_entropy_loss(logits, y)
            losses.append(float(loss.data))
        
        avg_loss = float(np.mean(losses))
        ppl = float(np.exp(min(avg_loss, 20)))  # Cap para evitar overflow
        
        return {'loss': avg_loss, 'ppl': ppl}
    
    # ============================================
    # ENTRENAMIENTO
    # ============================================
    
    def train(self, train_ds, val_ds=None, epochs: int = 10,
              batch_size: int = 8, batches_per_epoch: int = 20,
              verbose: bool = True, eval_every: int = 1,
              early_stopping_patience: int = None):
        """
        Bucle de entrenamiento completo con backprop REAL.
        """
        if verbose:
            print(f"🏋️ Entrenando {self.model.name}")
            print(f"   LR: {self.lr}")
            print(f"   Épocas: {epochs}")
            print(f"   Batches/época: {batches_per_epoch}")
            print(f"   Batch size: {batch_size}")
            print(f"   Train samples: {len(train_ds)}")
            if val_ds:
                print(f"   Val samples: {len(val_ds)}")
            print("="*60)
        
        start_time = time.time()
        epochs_no_improvement = 0
        
        for epoch in range(epochs):
            epoch_start = time.time()
            train_losses = []
            
            for _ in range(batches_per_epoch):
                X, y = train_ds.get_batch(batch_size=batch_size, shuffle=True)
                loss = self.train_step(X, y)
                train_losses.append(loss)
            
            avg_train_loss = float(np.mean(train_losses))
            train_ppl = float(np.exp(min(avg_train_loss, 20)))
            
            self.history['train_loss'].append(avg_train_loss)
            self.history['train_ppl'].append(train_ppl)
            
            val_info = ""
            improved = False
            if val_ds is not None and (epoch + 1) % eval_every == 0:
                val_metrics = self.evaluate(val_ds, batch_size=batch_size, num_batches=5)
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_ppl'].append(val_metrics['ppl'])
                val_info = f" | val_loss={val_metrics['loss']:.4f}, val_ppl={val_metrics['ppl']:.2f}"
                
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.best_epoch = epoch
                    # Guardar mejor peso
                    self.best_weights = [p.data.copy() for p in self.model.parameters()]
                    improved = True
                    epochs_no_improvement = 0
                else:
                    epochs_no_improvement += 1
            
            epoch_time = time.time() - epoch_start
            if verbose:
                marker = " ⭐" if improved else ""
                print(f"Epoch {epoch+1}/{epochs} | train_loss={avg_train_loss:.4f}, "
                      f"train_ppl={train_ppl:.2f}{val_info} | {epoch_time:.2f}s{marker}")
            
            # Early stopping
            if early_stopping_patience is not None and epochs_no_improvement >= early_stopping_patience:
                if verbose:
                    print(f"⏹️ Early stopping: {early_stopping_patience} épocas sin mejora")
                break
        
        total_time = time.time() - start_time
        if verbose:
            print("="*60)
            print(f"✅ Entrenamiento completado en {total_time:.2f}s")
            if self.best_epoch >= 0:
                print(f"   Mejor época: {self.best_epoch+1}")
                print(f"   Mejor val_loss: {self.best_val_loss:.4f}")
                # Restaurar mejores pesos
                if self.best_weights is not None:
                    for p, w in zip(self.model.parameters(), self.best_weights):
                        p.data = w.copy()
                    if verbose:
                        print(f"   ✅ Mejores pesos restaurados")
        
        self.model._trained = True

        # Generar gráfico automáticamente si hay historial
        try:
            if self.history.get('train_loss'):
                self.guardar_grafico()
        except Exception as e:
            print(f"⚠️ No se pudo generar el gráfico: {e}")

        return self.history
    
    # ============================================
    # PERSISTENCIA
    # ============================================
    

    def guardar_grafico(self, path: str = "learning_curve.png") -> str:
        """
        Genera y guarda el gráfico de la curva de aprendizaje.
        
        Returns:
            Ruta del PNG generado.
        """
        from utils.plots import plot_learning_curve
        
        if not self.history.get('train_loss'):
            raise ValueError("No hay historial de entrenamiento para graficar")
        
        return plot_learning_curve(
            train_loss=self.history['train_loss'],
            val_loss=self.history.get('val_loss', []),
            path=path,
            titulo="Cerebro Zero — Curva de aprendizaje",
        )

    def save(self, path: str):
        """Guarda el modelo y el histórico (formato seguro JSON+NPY, sin pickle)"""
        # Si path acaba en .pkl, quitamos extensión
        if path.endswith(".pkl") or path.endswith(".pickle"):
            path = path.rsplit(".", 1)[0]

        # Convertir lista de ndarrays a dict con claves únicas
        # (safe_save_dict espera dict, no list)
        weights = [p.data.copy() for p in self.model.parameters()]
        weights_dict = {f"weight_{i}": w for i, w in enumerate(weights)}

        state = {
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'vocab_size': self.model.vocab_size,
            'd_model': self.model.d_model,
            'num_heads': self.model.num_heads,
            'num_layers': self.model.num_layers,
            'num_weights': len(weights),
        }
        state.update(weights_dict)
        safe_save_dict(state, path)
        print(f"💾 Modelo guardado en {path}")
    
    def load(self, path: str):
        """Carga el modelo y el histórico (formato seguro JSON+NPY, sin pickle)"""
        if path.endswith(".pkl") or path.endswith(".pickle"):
            path = path.rsplit(".", 1)[0]

        state = safe_load_dict(path)

        # Reconstruir pesos desde el dict
        num_weights = state.get('num_weights', 0)
        params = self.model.parameters()
        for i, p in enumerate(params):
            key = f"weight_{i}"
            if key in state:
                p.data = state[key].copy()

        self.history = state.get('history', self.history)
        self.best_val_loss = state.get('best_val_loss', float('inf'))
        self.best_epoch = state.get('best_epoch', -1)
        print(f"📂 Modelo cargado desde {path}")


if __name__ == "__main__":
    import sys as _sys
    import os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    
    from utils.visual import (
        consola, titulo, ok, warn, error, info, dim,
        seccion, bullet, kv, panel, tabla
    )
    from language.dataset import TextDataset
    from language.tokenizer_v3 import BPETokenizer
    
    titulo("LM TRAINER — Entrenamiento visual")
    
    # ============================================
    # PREPARAR DATOS
    # ============================================
    seccion("📚 Preparando datos")
    
    # Corpus de ejemplo
    corpus_texto = (
        "la inteligencia artificial aprende de los datos "
        "cerebro zero es un proyecto en python y numpy "
        "entrenar un modelo requiere paciencia y datos "
        "aprender es el proceso de mejorar con la experiencia "
    ) * 50
    
    # Tokenizer
    tok = BPETokenizer(vocab_size=260)
    tok.entrenar(corpus_texto)
    kv("Vocab size", tok.vocab_size, color="cyan")
    kv("Compresión", f"{tok.compresion_media(corpus_texto):.2f}x", color="green")
    
    # Datasets
    train_path = "datasets/train.txt"
    val_path = "datasets/validation.txt"
    
    if not _os.path.exists(train_path):
        _os.makedirs("datasets", exist_ok=True)
        with open(train_path, "w") as f:
            f.write(corpus_texto)
        with open(val_path, "w") as f:
            f.write(corpus_texto[:len(corpus_texto)//5])
        info(f"Creado {train_path} y {val_path}")
    
    train_ds = TextDataset(train_path, tok, context_len=8)
    val_ds = TextDataset(val_path, tok, context_len=8)
    kv("Train samples", len(train_ds))
    kv("Val samples", len(val_ds))
    
    # ============================================
    # MODELO
    # ============================================
    seccion("🧠 Creando modelo Transformer")
    
    model = Transformer(
        vocab_size=tok.vocab_size,
        d_model=16,
        num_heads=2,
        d_ff=32,
        num_layers=1,
        max_len=8,
    )
    kv("Parámetros", model.num_parameters(), color="cyan")
    kv("d_model", model.d_model)
    kv("num_heads", model.num_heads)
    kv("num_layers", model.num_layers)
    
    # ============================================
    # ENTRENAMIENTO
    # ============================================
    seccion("🏋️  Entrenando")
    
    trainer = LMTrainer(model, learning_rate=0.01)
    
    info("Entrenando con backprop REAL...")
    history = trainer.train(
        train_ds, val_ds,
        epochs=20,
        batch_size=8,
        batches_per_epoch=10,
        verbose=False,
        early_stopping_patience=5,
    )
    
    # ============================================
    # RESULTADOS
    # ============================================
    seccion("📊 Resultados")
    
    if history['train_loss']:
        loss_ini = history['train_loss'][0]
        loss_fin = history['train_loss'][-1]
        mejora = (loss_ini - loss_fin) / loss_ini * 100
        
        tabla(
            ["Métrica", "Valor"],
            [
                ["Loss inicial", f"{loss_ini:.4f}"],
                ["Loss final", f"[green]{loss_fin:.4f}[/green]"],
                ["Mejora", f"[green]{mejora:+.2f}%[/green]"],
                ["Épocas", len(history['train_loss'])],
                ["Mejor val_loss", f"[cyan]{trainer.best_val_loss:.4f}[/cyan]"],
            ],
            titulo="Métricas de entrenamiento"
        )
    
    # ============================================
    # GENERACIÓN DE EJEMPLO
    # ============================================
    seccion("🔮 Generación de ejemplo")
    
    prompt = tok.encode("la inteligencia")
    generated = model.generate(prompt, max_new_tokens=8, temperature=0.8)
    texto = tok.decode(generated)
    
    consola.print()
    kv("Prompt", "la inteligencia", color="yellow")
    kv("Generado", texto, color="green")
    
    # ============================================
    # GRÁFICO
    # ============================================
    seccion("📈 Gráfico")
    
    png_path = trainer.guardar_grafico("learning_curve.png")
    kv("PNG generado", png_path, color="cyan")
    kv("Tamaño", f"{_os.path.getsize(png_path)/1024:.1f} KB")
    
    # ============================================
    # FINAL
    # ============================================
    consola.print()
    panel(
        "[bold green]LM TRAINER 4.0 FUNCIONANDO[/bold green]\n"
        f"[dim]Loss: {history['train_loss'][0]:.4f} → {history['train_loss'][-1]:.4f}[/dim]\n"
        f"[dim]Gráfico: {png_path}[/dim]",
        titulo="✅ Éxito",
        color="green"
    )

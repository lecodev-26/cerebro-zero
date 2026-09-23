"""
Módulo de gráficos — genera PNGs con Pillow (sin matplotlib)
==============================================================

Uso:
    from utils.plots import plot_learning_curve
    plot_learning_curve(
        train_loss=[4.5, 3.2, 2.8, 2.1],
        val_loss=[4.6, 3.3, 3.0, 2.3],
        path="learning_curve.png",
    )
"""

import os
from typing import List, Optional
from PIL import Image, ImageDraw, ImageFont


# ============================================
# PALETA DE COLORES
# ============================================

COLORES = {
    "fondo": (18, 18, 24),          # fondo oscuro tipo GitHub dark
    "grid": (45, 45, 55),           # rejilla
    "eje": (100, 100, 110),         # ejes
    "texto": (220, 220, 225),       # texto principal
    "texto_dim": (140, 140, 150),   # texto secundario
    "train": (100, 200, 255),       # azul (train)
    "val": (255, 140, 200),         # rosa (val)
    "mejora": (100, 230, 140),      # verde
    "titulo": (255, 255, 255),      # blanco
}


# ============================================
# FUENTES
# ============================================

def _get_font(size: int = 14, bold: bool = False) -> ImageFont.ImageFont:
    """
    Intenta cargar una fuente TrueType, cae a la default si falla.
    """
    candidatas = [
        "/system/fonts/Roboto-Regular.ttf",
        "/system/fonts/DroidSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else None,
    ]
    for path in candidatas:
        if path and os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


# ============================================
# HELPERS
# ============================================

def _escalar(valor, vmin, vmax, pmin, pmax):
    """Escala valor de rango [vmin, vmax] a [pmin, pmax]"""
    if vmax == vmin:
        return (pmin + pmax) / 2
    return pmin + (valor - vmin) * (pmax - pmin) / (vmax - vmin)


def _dibujar_ejes(draw, x0, y0, x1, y1, vmin, vmax, epochs):
    """Dibuja ejes, grid y etiquetas"""
    color_eje = COLORES["eje"]
    color_grid = COLORES["grid"]
    color_texto = COLORES["texto"]
    
    # Ejes principales
    draw.line([(x0, y0), (x0, y1)], fill=color_eje, width=2)  # eje Y
    draw.line([(x0, y1), (x1, y1)], fill=color_eje, width=2)  # eje X
    
    # Grid horizontal + etiquetas Y
    n_lineas = 5
    font = _get_font(11)
    for i in range(n_lineas + 1):
        y = y0 + (y1 - y0) * i / n_lineas
        valor = vmax - (vmax - vmin) * i / n_lineas
        draw.line([(x0, y), (x1, y)], fill=color_grid, width=1)
        draw.text((x0 - 8, y - 6), f"{valor:.2f}",
                  fill=color_texto, font=font, anchor="ra")
    
    # Etiquetas X (epochs)
    if epochs > 0:
        n_etiq = min(epochs, 10)
        step = max(1, epochs // n_etiq)
        for i in range(0, epochs + 1, step):
            x = x0 + (x1 - x0) * i / max(epochs, 1)
            if 0 <= i <= epochs:
                draw.text((x, y1 + 8), str(i),
                          fill=color_texto, font=font, anchor="ma")


def _dibujar_linea(draw, puntos, color, width=2):
    """Dibuja una polilínea"""
    if len(puntos) < 2:
        if len(puntos) == 1:
            x, y = puntos[0]
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=color)
        return
    draw.line(puntos, fill=color, width=width, joint="curve")
    # Puntos
    for x, y in puntos:
        draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=color)


# ============================================
# FUNCIÓN PRINCIPAL
# ============================================

def plot_learning_curve(
    train_loss: List[float],
    val_loss: Optional[List[float]] = None,
    path: str = "learning_curve.png",
    titulo: str = "Curva de aprendizaje",
    ancho: int = 900,
    alto: int = 550,
) -> str:
    """
    Genera un PNG con la curva de aprendizaje (loss vs epochs).
    
    Args:
        train_loss: lista de losses de entrenamiento
        val_loss: lista opcional de losses de validación
        path: ruta del archivo PNG a guardar
        titulo: título del gráfico
        ancho: ancho en px
        alto: alto en px
    
    Returns:
        La ruta del archivo generado.
    """
    if not train_loss:
        raise ValueError("train_loss está vacío")
    
    val_loss = val_loss or []
    
    # ============================================
    # CREAR IMAGEN
    # ============================================
    img = Image.new("RGB", (ancho, alto), COLORES["fondo"])
    draw = ImageDraw.Draw(img)
    
    # Márgenes
    margen_izq = 70
    margen_der = 40
    margen_sup = 80
    margen_inf = 60
    
    x0 = margen_izq
    y0 = margen_sup
    x1 = ancho - margen_der
    y1 = alto - margen_inf
    
    # ============================================
    # CALCULAR RANGOS
    # ============================================
    todos = list(train_loss)
    if val_loss:
        todos += list(val_loss)
    
    vmin = min(todos) * 0.95
    vmax = max(todos) * 1.05
    if vmax == vmin:
        vmax = vmin + 1
    epochs = len(train_loss) - 1
    
    # ============================================
    # TÍTULO
    # ============================================
    font_titulo = _get_font(22, bold=True)
    draw.text((ancho // 2, 25), titulo,
              fill=COLORES["titulo"], font=font_titulo, anchor="ma")
    
    # Subtítulo con info
    font_sub = _get_font(12)
    subtitulo = f"Epochs: {epochs + 1}  ·  "
    subtitulo += f"Train final: {train_loss[-1]:.4f}"
    if val_loss:
        subtitulo += f"  ·  Val final: {val_loss[-1]:.4f}"
    draw.text((ancho // 2, 55), subtitulo,
              fill=COLORES["texto_dim"], font=font_sub, anchor="ma")
    
    # ============================================
    # EJES + GRID
    # ============================================
    _dibujar_ejes(draw, x0, y0, x1, y1, vmin, vmax, epochs)
    
    # Etiquetas de ejes
    font_eje = _get_font(13)
    draw.text(((x0 + x1) // 2, alto - 25), "Epoch",
              fill=COLORES["texto"], font=font_eje, anchor="ma")
    draw.text((20, (y0 + y1) // 2), "Loss",
              fill=COLORES["texto"], font=font_eje, anchor="ma")
    
    # ============================================
    # DIBUJAR LÍNEAS
    # ============================================
    # Train
    puntos_train = [
        (x0 + (x1 - x0) * i / max(epochs, 1),
         _escalar(v, vmin, vmax, y1, y0))
        for i, v in enumerate(train_loss)
    ]
    _dibujar_linea(draw, puntos_train, COLORES["train"], width=3)
    
    # Val
    if val_loss:
        puntos_val = [
            (x0 + (x1 - x0) * i / max(epochs, 1),
             _escalar(v, vmin, vmax, y1, y0))
            for i, v in enumerate(val_loss)
        ]
        _dibujar_linea(draw, puntos_val, COLORES["val"], width=3)
    
    # ============================================
    # LEYENDA
    # ============================================
    font_leyenda = _get_font(13)
    leyenda_x = x1 - 180
    leyenda_y = y0 + 10
    
    # Train
    draw.line([(leyenda_x, leyenda_y + 7),
               (leyenda_x + 25, leyenda_y + 7)],
              fill=COLORES["train"], width=3)
    draw.text((leyenda_x + 35, leyenda_y),
              "Train loss", fill=COLORES["texto"], font=font_leyenda)
    
    # Val
    if val_loss:
        draw.line([(leyenda_x, leyenda_y + 30),
                   (leyenda_x + 25, leyenda_y + 30)],
                  fill=COLORES["val"], width=3)
        draw.text((leyenda_x + 35, leyenda_y + 23),
                  "Val loss", fill=COLORES["texto"], font=font_leyenda)
    
    # ============================================
    # MEJORA (esquina inferior)
    # ============================================
    mejora = (train_loss[0] - train_loss[-1]) / train_loss[0] * 100
    color_mejora = COLORES["mejora"] if mejora > 0 else (255, 100, 100)
    draw.text((x0, y1 + 35),
              f"Mejora: {mejora:+.1f}%",
              fill=color_mejora, font=font_leyenda)
    
    # ============================================
    # GUARDAR
    # ============================================
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    img.save(path, "PNG", optimize=True)
    
    return path


# ============================================
# TEST MANUAL
# ============================================

if __name__ == "__main__":
    import sys
    import os as _os
    sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    
    from utils.visual import titulo, ok, info, kv
    
    titulo("TEST PLOTS")
    
    # Datos simulados
    train_loss = [4.5, 3.8, 3.2, 2.8, 2.5, 2.2, 2.0, 1.85, 1.72, 1.62,
                  1.54, 1.48, 1.42, 1.38, 1.34]
    val_loss = [4.6, 3.9, 3.3, 2.9, 2.6, 2.35, 2.15, 2.0, 1.9, 1.82,
                1.76, 1.72, 1.69, 1.68, 1.68]
    
    info("Generando learning_curve.png...")
    path = plot_learning_curve(
        train_loss=train_loss,
        val_loss=val_loss,
        path="learning_curve_demo.png",
        titulo="Cerebro Zero — Curva de aprendizaje",
    )
    
    kv("Archivo", path, color="cyan")
    kv("Tamaño", f"{os.path.getsize(path) / 1024:.1f} KB")
    kv("Train loss final", f"{train_loss[-1]:.4f}")
    kv("Val loss final", f"{val_loss[-1]:.4f}")
    
    consola_ok = os.path.exists(path)
    if consola_ok:
        ok(f"PNG generado correctamente en {path}")
    else:
        print("❌ No se generó el PNG")

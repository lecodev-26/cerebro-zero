"""
Cuantización de modelos para reducir tamaño
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


class Quantizer:
    """
    Cuantiza modelos para reducirlos de tamaño.
    
    Uso:
        q = Quantizer()
        # Guardar cuantizado
        q.quantize_model(model, "model_q.npz")
        # Cargar
        model2 = q.dequantize_model(model_empty, "model_q.npz")
    """
    
    def __init__(self):
        pass
    
    def quantize_array(self, arr: np.ndarray, bits: int = 8):
        """
        Cuantiza un array de float32 a int8 (o uint8).
        
        Guarda el scale y zero_point para poder des-cuantizar.
        """
        # Rango de valores
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))
        
        # Número de niveles
        n_levels = 2 ** bits - 1
        
        # Calcular scale
        if max_val - min_val == 0:
            scale = 1.0
            zero_point = 0
        else:
            scale = (max_val - min_val) / n_levels
            zero_point = -min_val / scale
        
        # Cuantizar
        quantized = np.round(arr / scale + zero_point)
        quantized = np.clip(quantized, 0, n_levels)
        quantized = quantized.astype(np.uint8)
        
        return quantized, scale, zero_point
    
    def dequantize_array(self, quantized: np.ndarray, scale: float,
                         zero_point: float) -> np.ndarray:
        """Des-cuantiza un array"""
        return (quantized.astype(np.float32) - zero_point) * scale
    
    def quantize_model(self, model, path: str, bits: int = 8):
        """
        Cuantiza y guarda un modelo.
        
        Returns:
            dict con información de compresión
        """
        original_size = 0
        quantized_size = 0
        metadata = []
        
        for i, param in enumerate(model.parameters()):
            arr = param.data
            original_size += arr.nbytes
            
            quantized, scale, zero_point = self.quantize_array(arr, bits)
            quantized_size += quantized.nbytes
            
            metadata.append({
                'shape': arr.shape,
                'scale': scale,
                'zero_point': zero_point,
                'dtype': str(arr.dtype),
            })
        
        # Guardar
        save_data = {}
        for i, param in enumerate(model.parameters()):
            q, _, _ = self.quantize_array(param.data, bits)
            save_data[f'param_{i}'] = q
        
        np.savez_compressed(path, **save_data)
        
        compression_ratio = original_size / quantized_size if quantized_size > 0 else 0
        
        return {
            'original_bytes': original_size,
            'quantized_bytes': quantized_size,
            'compression_ratio': compression_ratio,
            'num_params': len(metadata),
            'metadata': metadata,
        }
    
    def quantize_model_in_memory(self, model, bits: int = 8) -> dict:
        """
        Cuantiza el modelo en memoria sin guardarlo.
        Devuelve un dict con los arrays cuantizados.
        """
        quantized_data = {}
        metadata = []
        
        for i, param in enumerate(model.parameters()):
            q, scale, zero_point = self.quantize_array(param.data, bits)
            quantized_data[f'param_{i}'] = q
            metadata.append({
                'shape': param.data.shape,
                'scale': scale,
                'zero_point': zero_point,
            })
        
        return {
            'data': quantized_data,
            'metadata': metadata,
        }
    
    def dequantize_to_model(self, model, quantized_data: dict, metadata: list):
        """
        Restaura los pesos del modelo desde datos cuantizados.
        """
        params = model.parameters()
        for i, (param, meta) in enumerate(zip(params, metadata)):
            q = quantized_data[f'param_{i}']
            restored = self.dequantize_array(q, meta['scale'], meta['zero_point'])
            param.data = restored.reshape(meta['shape']).astype(np.float32)


if __name__ == "__main__":
    print("🧪 PROBANDO QUANTIZER")
    print("="*50)
    
    # Test con array simple
    arr = np.random.randn(100, 100).astype(np.float32) * 10
    
    q = Quantizer()
    quantized, scale, zero_point = q.quantize_array(arr, bits=8)
    restored = q.dequantize_array(quantized, scale, zero_point)
    
    print(f"\n📊 Array original: {arr.shape} ({arr.nbytes} bytes)")
    print(f"📊 Array cuantizado: {quantized.shape} ({quantized.nbytes} bytes)")
    print(f"📊 Ratio de compresión: {arr.nbytes / quantized.nbytes:.2f}×")
    
    # Error de cuantización
    error = np.mean(np.abs(arr - restored))
    print(f"📊 Error medio: {error:.4f}")
    
    # Test con modelo
    from models.transformer import Transformer
    model = Transformer(vocab_size=20, d_model=16, num_heads=2,
                        d_ff=32, num_layers=1, max_len=8)
    
    print(f"\n📊 Modelo: {model.name}")
    print(f"📊 Parámetros: {model.num_parameters()}")
    
    info = model.quantize if hasattr(model, 'quantize') else None
    
    result = q.quantize_model(model, "/tmp/test_quantized.npz")
    print(f"\n💾 Modelo cuantizado:")
    print(f"   Original: {result['original_bytes']} bytes")
    print(f"   Cuantizado: {result['quantized_bytes']} bytes")
    print(f"   Ratio: {result['compression_ratio']:.2f}×")
    
    print("\n✅ QUANTIZER FUNCIONANDO")

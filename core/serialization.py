"""
Persistencia segura para Cerebro Zero.
Formato: JSON + NPY (sin pickle, sin ejecución de código)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import hashlib
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Optional
from core.tensor import Tensor


# ============================================
# FORMATO SEGURO
# ============================================

MODEL_FORMAT_VERSION = "1.0"


def compute_hash(data: bytes) -> str:
    """SHA256 de bytes"""
    return hashlib.sha256(data).hexdigest()


def safe_save_model(model, path: str, metadata: dict = None):
    """
    Guarda un modelo en formato seguro.
    
    Estructura:
        path/
        ├── metadata.json      ← info del modelo
        ├── manifest.json      ← estructura de tensores
        └── tensors/
            ├── 000.npy
            ├── 001.npy
            └── ...
    
    Args:
        model: modelo con método `parameters()`
        path: directorio donde guardar
        metadata: dict adicional
    """
    # Crear directorios
    os.makedirs(path, exist_ok=True)
    tensors_dir = os.path.join(path, "tensors")
    os.makedirs(tensors_dir, exist_ok=True)
    
    # Obtener parámetros
    params = model.parameters()
    
    # Guardar cada tensor
    manifest = {
        'format_version': MODEL_FORMAT_VERSION,
        'num_tensors': len(params),
        'tensors': [],
    }
    
    for i, param in enumerate(params):
        # Nombre del archivo
        filename = f"{i:04d}.npy"
        filepath = os.path.join(tensors_dir, filename)
        
        # Guardar tensor como .npy
        np.save(filepath, param.data)
        
        # Hash del archivo
        with open(filepath, 'rb') as f:
            file_hash = compute_hash(f.read())
        
        # Info en manifest
        manifest['tensors'].append({
            'index': i,
            'file': filename,
            'shape': list(param.data.shape),
            'dtype': str(param.data.dtype),
            'requires_grad': param.requires_grad,
            'hash_sha256': file_hash,
        })
    
    # Metadata del modelo
    model_metadata = {
        'format_version': MODEL_FORMAT_VERSION,
        'name': getattr(model, 'name', model.__class__.__name__),
        'version': getattr(model, 'version', '1.0.0'),
        'class': model.__class__.__name__,
        'saved_at': datetime.now().isoformat(),
        'num_parameters': sum(p.data.size for p in params),
    }
    
    if metadata:
        model_metadata['extra'] = metadata
    
    # Guardar metadata.json
    with open(os.path.join(path, "metadata.json"), 'w') as f:
        json.dump(model_metadata, f, indent=2)
    
    # Guardar manifest.json
    with open(os.path.join(path, "manifest.json"), 'w') as f:
        json.dump(manifest, f, indent=2)
    
    return model_metadata


def safe_load_model(model, path: str, verify_hashes: bool = True):
    """
    Carga un modelo desde formato seguro.
    
    Args:
        model: modelo vacío con la misma estructura
        path: directorio donde está guardado
        verify_hashes: verificar integridad con SHA256
    """
    # Leer manifest
    manifest_path = os.path.join(path, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"No se encontró {manifest_path}")
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    # Verificar versión
    if manifest.get('format_version') != MODEL_FORMAT_VERSION:
        print(f"⚠️ Versión de formato diferente: {manifest.get('format_version')}")
    
    # Leer metadata
    metadata_path = os.path.join(path, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    # Cargar cada tensor
    params = model.parameters()
    tensors_dir = os.path.join(path, "tensors")
    
    if len(params) != manifest['num_tensors']:
        raise ValueError(
            f"Número de parámetros no coincide: "
            f"modelo={len(params)}, manifest={manifest['num_tensors']}"
        )
    
    for tensor_info in manifest['tensors']:
        i = tensor_info['index']
        filepath = os.path.join(tensors_dir, tensor_info['file'])
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No se encontró {filepath}")
        
        # Verificar hash si se pide
        if verify_hashes:
            with open(filepath, 'rb') as f:
                actual_hash = compute_hash(f.read())
            if actual_hash != tensor_info['hash_sha256']:
                raise ValueError(
                    f"Hash no coincide para tensor {i}: "
                    f"esperado={tensor_info['hash_sha256'][:16]}..., "
                    f"actual={actual_hash[:16]}..."
                )
        
        # Cargar tensor
        data = np.load(filepath)
        
        # Verificar shape
        if list(data.shape) != tensor_info['shape']:
            raise ValueError(
                f"Shape no coincide para tensor {i}: "
                f"esperado={tensor_info['shape']}, actual={list(data.shape)}"
            )
        
        # Asignar al parámetro
        params[i].data = data
    
    return metadata


def safe_save_dict(data: dict, path: str):
    """
    Guarda un diccionario con arrays numpy de forma segura.
    
    Formato:
        path/
        ├── metadata.json
        └── arrays/
            ├── key1.npy
            └── key2.npy
    """
    os.makedirs(path, exist_ok=True)
    arrays_dir = os.path.join(path, "arrays")
    os.makedirs(arrays_dir, exist_ok=True)
    
    manifest = {
        'format_version': MODEL_FORMAT_VERSION,
        'saved_at': datetime.now().isoformat(),
        'keys': [],
    }
    
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            filename = f"{key}.npy"
            filepath = os.path.join(arrays_dir, filename)
            np.save(filepath, value)
            
            with open(filepath, 'rb') as f:
                file_hash = compute_hash(f.read())
            
            manifest['keys'].append({
                'key': key,
                'type': 'ndarray',
                'file': filename,
                'shape': list(value.shape),
                'dtype': str(value.dtype),
                'hash_sha256': file_hash,
            })
        else:
            # Guardar como JSON puro
            manifest['keys'].append({
                'key': key,
                'type': 'json',
                'value': value,
            })
    
    with open(os.path.join(path, "metadata.json"), 'w') as f:
        json.dump(manifest, f, indent=2)


def safe_load_dict(path: str, verify_hashes: bool = True) -> dict:
    """Carga un diccionario desde formato seguro"""
    manifest_path = os.path.join(path, "metadata.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"No se encontró {manifest_path}")
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    result = {}
    arrays_dir = os.path.join(path, "arrays")
    
    for entry in manifest['keys']:
        key = entry['key']
        
        if entry['type'] == 'ndarray':
            filepath = os.path.join(arrays_dir, entry['file'])
            
            if verify_hashes:
                with open(filepath, 'rb') as f:
                    actual_hash = compute_hash(f.read())
                if actual_hash != entry['hash_sha256']:
                    raise ValueError(f"Hash no coincide para {key}")
            
            result[key] = np.load(filepath)
        else:
            result[key] = entry['value']
    
    return result


# ============================================
# TEST
# ============================================

if __name__ == "__main__":
    print("🧪 PROBANDO PERSISTENCIA SEGURA")
    print("="*60)
    
    import tempfile
    from models.transformer import Transformer
    
    # Crear modelo
    model = Transformer(vocab_size=20, d_model=16, num_heads=2,
                        d_ff=32, num_layers=1, max_len=8)
    
    print(f"\n📦 Modelo original:")
    print(f"   Nombre: {model.name}")
    print(f"   Parámetros: {model.num_parameters()}")
    
    # Guardar
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "modelo_test")
        metadata = safe_save_model(model, save_path, {'descripcion': 'Modelo de prueba'})
        print(f"\n💾 Guardado en {save_path}")
        
        # Ver estructura
        print(f"\n📂 Estructura:")
        for root, dirs, files in os.walk(save_path):
            level = root.replace(save_path, '').count(os.sep)
            indent = '   ' * level
            print(f"{indent}{os.path.basename(root)}/")
            for file in sorted(files)[:5]:
                size = os.path.getsize(os.path.join(root, file))
                print(f"{indent}   {file} ({size} bytes)")
        
        # Cargar en modelo nuevo
        model2 = Transformer(vocab_size=20, d_model=16, num_heads=2,
                             d_ff=32, num_layers=1, max_len=8)
        safe_load_model(model2, save_path)
        
        # Verificar que son idénticos
        print(f"\n🔍 Verificando...")
        params1 = model.parameters()
        params2 = model2.parameters()
        
        all_equal = True
        for i, (p1, p2) in enumerate(zip(params1, params2)):
            if not np.array_equal(p1.data, p2.data):
                print(f"   ❌ Tensor {i} no coincide")
                all_equal = False
        
        if all_equal:
            print(f"   ✅ TODOS los {len(params1)} tensores coinciden")
        
        # Test de integridad
        print(f"\n🔒 Test de integridad (modificar un archivo):")
        # Encontrar un archivo .npy y corromperlo
        tensors_dir = os.path.join(save_path, "tensors")
        target = os.path.join(tensors_dir, "0000.npy")
        
        # Modificar el archivo
        data = np.load(target)
        data_bad = data.copy()
        data_bad[0, 0] = 99999
        np.save(target, data_bad)
        
        # Intentar cargar → debe fallar por hash
        model3 = Transformer(vocab_size=20, d_model=16, num_heads=2,
                             d_ff=32, num_layers=1, max_len=8)
        try:
            safe_load_model(model3, save_path)
            print(f"   ❌ NO detectó la corrupción")
        except ValueError as e:
            print(f"   ✅ Corrupción detectada: {str(e)[:60]}...")
    
    print("\n✅ PERSISTENCIA SEGURA FUNCIONANDO")

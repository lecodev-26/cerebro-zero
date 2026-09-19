"""
Serialization v2 — soporta dicts anidados con ndarrays
========================================================

Mejora sobre v1: recorre recursivamente dicts y listas,
aplana ndarrays a rutas (ej: 'weights.0.weight') y los guarda
como archivos .npy separados.

Formato:
    path/
    ├── metadata.json       ← describe la estructura original
    └── arrays/
        ├── weights.0.weight.npy
        ├── weights.0.bias.npy
        └── ...
"""

import json
import os
import numpy as np
from datetime import datetime

from core.serialization import compute_hash, MODEL_FORMAT_VERSION


def _aplanar(obj, prefix=""):
    """
    Recorre recursivamente obj (dict/list/ndarray/scalar).
    
    Devuelve:
        (aplanado, estructura)
        
        - aplanado: dict {ruta: ndarray} solo para ndarrays
        - estructura: representación serializable en JSON
                      con marcadores para reconstruir
    """
    aplanado = {}
    
    if isinstance(obj, np.ndarray):
        aplanado[prefix] = obj
        return aplanado, {"__type__": "ndarray", "__path__": prefix}
    
    if isinstance(obj, dict):
        estructura = {"__type__": "dict", "__keys__": {}}
        for k, v in obj.items():
            sub_aplanado, sub_estructura = _aplanar(v, f"{prefix}.{k}" if prefix else str(k))
            aplanado.update(sub_aplanado)
            estructura["__keys__"][str(k)] = sub_estructura
        return aplanado, estructura
    
    if isinstance(obj, (list, tuple)):
        estructura = {
            "__type__": "list" if isinstance(obj, list) else "tuple",
            "__items__": [],
        }
        for i, v in enumerate(obj):
            sub_aplanado, sub_estructura = _aplanar(v, f"{prefix}.{i}" if prefix else str(i))
            aplanado.update(sub_aplanado)
            estructura["__items__"].append(sub_estructura)
        return aplanado, estructura
    
    # Escalar: json puro
    return aplanado, {"__type__": "json", "__value__": obj}


def _reconstruir(estructura, arrays):
    """Reconstruye el objeto original desde estructura + arrays"""
    t = estructura.get("__type__")
    
    if t == "ndarray":
        path = estructura["__path__"]
        if path in arrays:
            return arrays[path]
        raise ValueError(f"Array '{path}' no encontrado")
    
    if t == "dict":
        return {
            k: _reconstruir(v, arrays)
            for k, v in estructura["__keys__"].items()
        }
    
    if t == "list":
        return [_reconstruir(item, arrays) for item in estructura["__items__"]]
    
    if t == "tuple":
        return tuple(_reconstruir(item, arrays) for item in estructura["__items__"])
    
    if t == "json":
        return estructura["__value__"]
    
    raise ValueError(f"Tipo desconocido: {t}")


def safe_save_dict_v2(data: dict, path: str):
    """
    Guarda un dict con ndarrays anidados.
    Soporta cualquier profundidad de dict/list.
    """
    os.makedirs(path, exist_ok=True)
    arrays_dir = os.path.join(path, "arrays")
    os.makedirs(arrays_dir, exist_ok=True)
    
    aplanado, estructura = _aplanar(data)
    
    manifest_arrays = []
    for ruta, arr in aplanado.items():
        # Sanitizar nombre del archivo
        filename = ruta.replace(".", "_") + ".npy"
        filepath = os.path.join(arrays_dir, filename)
        np.save(filepath, arr)
        
        with open(filepath, "rb") as f:
            file_hash = compute_hash(f.read())
        
        manifest_arrays.append({
            "path": ruta,
            "file": filename,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "hash_sha256": file_hash,
        })
    
    manifest = {
        "format_version": MODEL_FORMAT_VERSION,
        "saved_at": datetime.now().isoformat(),
        "estructura": estructura,
        "arrays": manifest_arrays,
    }
    
    with open(os.path.join(path, "metadata.json"), "w") as f:
        json.dump(manifest, f, indent=2)


def safe_load_dict_v2(path: str, verify_hashes: bool = True) -> dict:
    """Carga un dict guardado con safe_save_dict_v2"""
    manifest_path = os.path.join(path, "metadata.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"No se encontró {manifest_path}")
    
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    
    # Verificar formato
    if "estructura" not in manifest:
        # Formato v1: usar el loader original
        from core.serialization import safe_load_dict
        return safe_load_dict(path, verify_hashes=verify_hashes)
    
    arrays_dir = os.path.join(path, "arrays")
    arrays = {}
    
    for entry in manifest["arrays"]:
        filepath = os.path.join(arrays_dir, entry["file"])
        
        if verify_hashes:
            with open(filepath, "rb") as f:
                actual_hash = compute_hash(f.read())
            if actual_hash != entry["hash_sha256"]:
                raise ValueError(f"Hash no coincide para {entry['path']}")
        
        arrays[entry["path"]] = np.load(filepath)
    
    return _reconstruir(manifest["estructura"], arrays)

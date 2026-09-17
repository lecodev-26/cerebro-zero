"""
Tests de Reproducibilidad 3.0
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import tempfile
import shutil
from evaluation.reproducibility import (
    hash_estructura, hash_archivo,
    EntornoCaptura, RegistroExperimento, Reproducibilidad,
)


# ============================================
# FIXTURES
# ============================================

@pytest.fixture
def repro():
    tmpdir = tempfile.mkdtemp(prefix="repro_test_")
    r = Reproducibilidad(directorio=tmpdir)
    yield r
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def entorno():
    return EntornoCaptura.capturar(seed=42)


@pytest.fixture
def config_ejemplo():
    return {
        "modelo": "transformer",
        "capas": 2,
        "lr": 0.001,
        "epochs": 5,
    }


# ============================================
# TESTS DE HASH ESTRUCTURA
# ============================================

def test_hash_determinista():
    h1 = hash_estructura({"a": 1, "b": 2})
    h2 = hash_estructura({"a": 1, "b": 2})
    assert h1 == h2


def test_hash_orden_no_importa():
    h1 = hash_estructura({"a": 1, "b": 2})
    h2 = hash_estructura({"b": 2, "a": 1})
    assert h1 == h2


def test_hash_contenido_distinto():
    h1 = hash_estructura({"a": 1})
    h2 = hash_estructura({"a": 2})
    assert h1 != h2


def test_hash_string():
    h1 = hash_estructura("hola")
    h2 = hash_estructura("hola")
    assert h1 == h2
    assert h1 != hash_estructura("adios")


def test_hash_lista():
    h1 = hash_estructura([1, 2, 3])
    h2 = hash_estructura([1, 2, 3])
    h3 = hash_estructura([3, 2, 1])
    assert h1 == h2
    assert h1 != h3  # orden SÍ importa en listas


def test_hash_anidado():
    h1 = hash_estructura({"a": {"b": [1, 2, {"c": 3}]}})
    h2 = hash_estructura({"a": {"b": [1, 2, {"c": 3}]}})
    assert h1 == h2


def test_hash_none():
    h1 = hash_estructura(None)
    h2 = hash_estructura(None)
    assert h1 == h2


def test_hash_es_hex():
    h = hash_estructura({"a": 1})
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def test_hash_float():
    h1 = hash_estructura({"lr": 0.001})
    h2 = hash_estructura({"lr": 0.001})
    h3 = hash_estructura({"lr": 0.002})
    assert h1 == h2
    assert h1 != h3


def test_hash_objeto_no_serializable():
    """Objetos raros no deben romper"""
    class Foo:
        def __repr__(self):
            return "Foo()"
    h = hash_estructura({"x": Foo()})
    assert isinstance(h, str)
    assert len(h) == 64


# ============================================
# TESTS DE HASH ARCHIVO
# ============================================

def test_hash_archivo_ok():
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"contenido")
        nombre = f.name
    try:
        h = hash_archivo(nombre)
        assert h is not None
        assert len(h) == 64
    finally:
        os.unlink(nombre)


def test_hash_archivo_no_existe():
    assert hash_archivo("/ruta/que/no/existe") is None


def test_hash_archivo_distinto_contenido():
    f1 = tempfile.NamedTemporaryFile(delete=False)
    f1.write(b"abc")
    f1.close()
    f2 = tempfile.NamedTemporaryFile(delete=False)
    f2.write(b"xyz")
    f2.close()
    try:
        assert hash_archivo(f1.name) != hash_archivo(f2.name)
    finally:
        os.unlink(f1.name)
        os.unlink(f2.name)


def test_hash_archivo_mismo_contenido():
    f1 = tempfile.NamedTemporaryFile(delete=False)
    f1.write(b"mismo contenido")
    f1.close()
    f2 = tempfile.NamedTemporaryFile(delete=False)
    f2.write(b"mismo contenido")
    f2.close()
    try:
        assert hash_archivo(f1.name) == hash_archivo(f2.name)
    finally:
        os.unlink(f1.name)
        os.unlink(f2.name)


# ============================================
# TESTS DE ENTORNO
# ============================================

def test_entorno_captura(entorno):
    assert entorno.python_version != ""
    assert entorno.numpy_version != ""
    assert entorno.plataforma != ""
    assert entorno.seed == 42


def test_entorno_fingerprint_determinista():
    e1 = EntornoCaptura(python_version="3.14", numpy_version="2.0",
                        plataforma="x", arquitectura="a", hardware="h",
                        git_commit="abc", git_branch="main", cpu_count=4)
    e2 = EntornoCaptura(python_version="3.14", numpy_version="2.0",
                        plataforma="x", arquitectura="a", hardware="h",
                        git_commit="abc", git_branch="main", cpu_count=4)
    assert e1.fingerprint == e2.fingerprint


def test_entorno_fingerprint_distinto_python():
    e1 = EntornoCaptura(python_version="3.14")
    e2 = EntornoCaptura(python_version="3.13")
    assert e1.fingerprint != e2.fingerprint


def test_entorno_fingerprint_ignora_timestamp():
    e1 = EntornoCaptura(python_version="3.14", timestamp=100)
    e2 = EntornoCaptura(python_version="3.14", timestamp=999)
    assert e1.fingerprint == e2.fingerprint


def test_entorno_to_dict(entorno):
    d = entorno.to_dict()
    assert "python_version" in d
    assert "numpy_version" in d
    assert "seed" in d


def test_entorno_repr(entorno):
    assert "EntornoCaptura" in repr(entorno)


def test_entorno_seed_distinto():
    e1 = EntornoCaptura.capturar(seed=42)
    e2 = EntornoCaptura.capturar(seed=100)
    assert e1.fingerprint != e2.fingerprint


# ============================================
# TESTS DE REGISTRO
# ============================================

def test_registro_creation(entorno, config_ejemplo):
    reg = RegistroExperimento(
        nombre="test",
        entorno=entorno,
        config=config_ejemplo,
    )
    assert reg.nombre == "test"
    assert reg.config_hash != ""


def test_registro_config_hash_auto(entorno, config_ejemplo):
    reg = RegistroExperimento(
        nombre="test",
        entorno=entorno,
        config=config_ejemplo,
    )
    assert reg.config_hash == hash_estructura(config_ejemplo)


def test_registro_fingerprint(entorno, config_ejemplo):
    reg = RegistroExperimento(
        nombre="test",
        entorno=entorno,
        config=config_ejemplo,
    )
    fp = reg.fingerprint
    assert len(fp) == 16


def test_registro_fingerprint_distinto_config(entorno):
    reg1 = RegistroExperimento(
        nombre="a", entorno=entorno, config={"x": 1},
    )
    reg2 = RegistroExperimento(
        nombre="b", entorno=entorno, config={"x": 2},
    )
    assert reg1.fingerprint != reg2.fingerprint


def test_registro_fingerprint_mismo_config(entorno):
    reg1 = RegistroExperimento(
        nombre="a", entorno=entorno, config={"x": 1},
    )
    reg2 = RegistroExperimento(
        nombre="b", entorno=entorno, config={"x": 1},
    )
    assert reg1.fingerprint == reg2.fingerprint


def test_registro_añadir_dataset(entorno):
    reg = RegistroExperimento(nombre="test", entorno=entorno)
    
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"datos")
        nombre = f.name
    
    try:
        reg.añadir_dataset("train", nombre)
        assert "train" in reg.dataset_hashes
        assert len(reg.dataset_hashes["train"]) == 64
    finally:
        os.unlink(nombre)


def test_registro_añadir_dataset_no_existe(entorno):
    reg = RegistroExperimento(nombre="test", entorno=entorno)
    with pytest.raises(FileNotFoundError):
        reg.añadir_dataset("x", "/ruta/falsa")


def test_registro_añadir_modelo(entorno):
    reg = RegistroExperimento(nombre="test", entorno=entorno)
    
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"modelo")
        nombre = f.name
    
    try:
        reg.añadir_modelo("m1", nombre)
        assert "m1" in reg.model_hashes
    finally:
        os.unlink(nombre)


def test_registro_to_dict(entorno, config_ejemplo):
    reg = RegistroExperimento(
        nombre="test",
        entorno=entorno,
        config=config_ejemplo,
        resultado={"acc": 0.9},
    )
    d = reg.to_dict()
    assert d["nombre"] == "test"
    assert d["config"] == config_ejemplo
    assert d["resultado"]["acc"] == 0.9
    assert "fingerprint" in d


def test_registro_from_dict(entorno, config_ejemplo):
    reg1 = RegistroExperimento(
        nombre="test",
        entorno=entorno,
        config=config_ejemplo,
        resultado={"acc": 0.9},
    )
    d = reg1.to_dict()
    reg2 = RegistroExperimento.from_dict(d)
    assert reg2.nombre == reg1.nombre
    assert reg2.fingerprint == reg1.fingerprint


def test_registro_repr(entorno):
    reg = RegistroExperimento(nombre="test", entorno=entorno)
    assert "RegistroExperimento" in repr(reg)


# ============================================
# TESTS DE GESTOR
# ============================================

def test_gestor_creation(repro):
    assert repro.directorio is not None
    assert os.path.isdir(repro.directorio)


def test_gestor_nuevo_experimento(repro, config_ejemplo):
    reg = repro.nuevo_experimento("test", config=config_ejemplo)
    assert reg.nombre == "test"
    assert reg.config == config_ejemplo
    assert reg.entorno.python_version != ""


def test_gestor_nuevo_experimento_sin_config(repro):
    reg = repro.nuevo_experimento("test")
    assert reg.config == {}


def test_gestor_guardar(repro):
    reg = repro.nuevo_experimento("test")
    archivo = repro.guardar(reg)
    assert os.path.exists(archivo)
    os.unlink(archivo)


def test_gestor_cargar(repro):
    reg = repro.nuevo_experimento("test", config={"x": 1})
    archivo = repro.guardar(reg)
    try:
        reg2 = repro.cargar(archivo)
        assert reg2 is not None
        assert reg2.fingerprint == reg.fingerprint
    finally:
        os.unlink(archivo)


def test_gestor_cargar_no_existe(repro):
    assert repro.cargar("/ruta/falsa") is None


def test_gestor_listar(repro):
    reg1 = repro.nuevo_experimento("a")
    reg2 = repro.nuevo_experimento("b")
    a1 = repro.guardar(reg1)
    a2 = repro.guardar(reg2)
    try:
        archivos = repro.listar()
        assert len(archivos) == 2
    finally:
        os.unlink(a1)
        os.unlink(a2)


def test_gestor_resumen(repro):
    r = repro.resumen()
    assert "directorio" in r
    assert "registros_en_memoria" in r


def test_gestor_repr(repro):
    assert "Reproducibilidad" in repr(repro)


# ============================================
# TESTS DE COMPARACIÓN
# ============================================

def test_comparar_identicos(repro, config_ejemplo):
    reg1 = repro.nuevo_experimento("A", config=config_ejemplo, seed=42)
    reg2 = repro.nuevo_experimento("B", config=config_ejemplo, seed=42)
    
    comp = repro.comparar(reg1, reg2)
    assert comp["identicos"] == True
    assert comp["diferencias"] == {}


def test_comparar_config_distinta(repro, config_ejemplo):
    reg1 = repro.nuevo_experimento("A", config=config_ejemplo)
    
    config2 = config_ejemplo.copy()
    config2["lr"] = 0.01
    
    reg2 = repro.nuevo_experimento("B", config=config2)
    
    comp = repro.comparar(reg1, reg2)
    assert comp["identicos"] == False
    assert "config" in comp["diferencias"]


def test_comparar_seed_distinta(repro, config_ejemplo):
    reg1 = repro.nuevo_experimento("A", config=config_ejemplo, seed=42)
    reg2 = repro.nuevo_experimento("B", config=config_ejemplo, seed=100)
    
    comp = repro.comparar(reg1, reg2)
    assert comp["identicos"] == False
    assert "entorno" in comp["diferencias"]


def test_comparar_resultados_difieren(repro, config_ejemplo):
    reg1 = repro.nuevo_experimento("A", config=config_ejemplo)
    reg1.resultado = {"acc": 0.9}
    reg2 = repro.nuevo_experimento("B", config=config_ejemplo)
    reg2.resultado = {"acc": 0.8}
    
    comp = repro.comparar(reg1, reg2)
    assert comp["identicos"] == True  # fingerprint igual
    assert comp["resultados_difieren"] == True


# ============================================
# TESTS DE VERIFICACIÓN
# ============================================

def test_verificar_reproducible(repro, config_ejemplo):
    reg1 = repro.nuevo_experimento("A", config=config_ejemplo)
    reg1.resultado = {"acc": 0.9}
    reg2 = repro.nuevo_experimento("B", config=config_ejemplo)
    reg2.resultado = {"acc": 0.9}
    
    v = repro.verificar_reproducible(reg1, reg2)
    assert v["reproducible"] == True


def test_verificar_no_reproducible_resultados(repro, config_ejemplo):
    reg1 = repro.nuevo_experimento("A", config=config_ejemplo)
    reg1.resultado = {"acc": 0.9}
    reg2 = repro.nuevo_experimento("B", config=config_ejemplo)
    reg2.resultado = {"acc": 0.5}
    
    v = repro.verificar_reproducible(reg1, reg2)
    assert v["reproducible"] == False
    assert "acc" in v["campos_diferentes"]


def test_verificar_no_reproducible_fingerprints(repro, config_ejemplo):
    reg1 = repro.nuevo_experimento("A", config=config_ejemplo)
    config2 = config_ejemplo.copy()
    config2["lr"] = 0.5
    reg2 = repro.nuevo_experimento("B", config=config2)
    
    v = repro.verificar_reproducible(reg1, reg2)
    assert v["reproducible"] == False
    assert "Fingerprints" in v["razon"]


def test_verificar_dentro_tolerancia(repro, config_ejemplo):
    reg1 = repro.nuevo_experimento("A", config=config_ejemplo)
    reg1.resultado = {"acc": 0.9}
    reg2 = repro.nuevo_experimento("B", config=config_ejemplo)
    reg2.resultado = {"acc": 0.9 + 1e-9}
    
    v = repro.verificar_reproducible(reg1, reg2, tolerancia=1e-6)
    assert v["reproducible"] == True


def test_verificar_fuera_tolerancia(repro, config_ejemplo):
    reg1 = repro.nuevo_experimento("A", config=config_ejemplo)
    reg1.resultado = {"acc": 0.9}
    reg2 = repro.nuevo_experimento("B", config=config_ejemplo)
    reg2.resultado = {"acc": 0.9001}
    
    v = repro.verificar_reproducible(reg1, reg2, tolerancia=1e-6)
    assert v["reproducible"] == False


# ============================================
# TESTS DE INTEGRACIÓN
# ============================================

def test_flujo_completo(repro):
    """Flujo típico: crear → guardar → cargar → comparar"""
    config = {"modelo": "mlp", "lr": 0.01}
    
    reg1 = repro.nuevo_experimento("run1", config=config, seed=42)
    reg1.resultado = {"acc": 0.85}
    archivo1 = repro.guardar(reg1)
    
    reg2 = repro.nuevo_experimento("run2", config=config, seed=42)
    reg2.resultado = {"acc": 0.85}
    archivo2 = repro.guardar(reg2)
    
    try:
        cargado = repro.cargar(archivo1)
        comp = repro.comparar(cargado, reg2)
        assert comp["identicos"] == True
        
        verif = repro.verificar_reproducible(cargado, reg2)
        assert verif["reproducible"] == True
    finally:
        os.unlink(archivo1)
        os.unlink(archivo2)


def test_flujo_con_datasets(repro):
    """Registrar dataset en el experimento"""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"dataset data")
        nombre = f.name
    
    try:
        reg = repro.nuevo_experimento("test")
        reg.añadir_dataset("train", nombre)
        assert "train" in reg.dataset_hashes
        
        # Mismo dataset → mismo fingerprint
        reg2 = repro.nuevo_experimento("test")
        reg2.añadir_dataset("train", nombre)
        assert reg.fingerprint == reg2.fingerprint
    finally:
        os.unlink(nombre)


def test_flujo_datasets_diferentes(repro):
    """Datasets distintos → fingerprints distintos"""
    f1 = tempfile.NamedTemporaryFile(delete=False)
    f1.write(b"data1")
    f1.close()
    
    f2 = tempfile.NamedTemporaryFile(delete=False)
    f2.write(b"data2")
    f2.close()
    
    try:
        reg1 = repro.nuevo_experimento("test")
        reg1.añadir_dataset("train", f1.name)
        
        reg2 = repro.nuevo_experimento("test")
        reg2.añadir_dataset("train", f2.name)
        
        assert reg1.fingerprint != reg2.fingerprint
    finally:
        os.unlink(f1.name)
        os.unlink(f2.name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

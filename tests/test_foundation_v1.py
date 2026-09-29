from pathlib import Path
from cerebro_zero import Cerebro
from cerebro_zero.config import Settings
from cerebro_zero.core import PersistenceStore, diagnose

def test_persistence(tmp_path):
    s=PersistenceStore(str(tmp_path)); s.save_execution("x",1.0,{"ok":True}); assert s.count()==1

def test_doctor(tmp_path):
    r=diagnose(Settings(data_dir=str(tmp_path))); assert r["numpy"] and r["data_dir"]

def test_runtime_persists(tmp_path):
    r=Cerebro(settings=Settings(data_dir=str(tmp_path))).run("persist")
    assert r.success and (Path(tmp_path)/"cerebro.db").exists()

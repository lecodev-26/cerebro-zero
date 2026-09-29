import importlib.util
from pathlib import Path

def diagnose(settings):
    data=Path(settings.data_dir).expanduser()
    return {"python":True,"numpy":importlib.util.find_spec("numpy") is not None,"data_dir":str(data),"data_dir_writable":data.exists() or data.parent.exists(),"profile":settings.profile}

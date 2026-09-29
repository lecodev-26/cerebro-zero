from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib,secrets,json,time

@dataclass(frozen=True)
class APIKey:
    key_id:str
    digest:str
    scopes:tuple[str,...]
    expires_at:float|None=None
    revoked:bool=False

class APIKeyStore:
    def __init__(self,path="~/.cerebro-zero/keys.json"):
        self.path=Path(path).expanduser(); self.path.parent.mkdir(parents=True,exist_ok=True)
        self._items={}; self._load()
    def _load(self):
        if self.path.exists():
            raw=json.loads(self.path.read_text(encoding="utf-8")); self._items={k:APIKey(k,v["digest"],tuple(v["scopes"]),v.get("expires_at"),v.get("revoked",False)) for k,v in raw.items()}
    def _save(self):
        self.path.write_text(json.dumps({k:asdict(v) for k,v in self._items.items()},indent=2),encoding="utf-8")
    def create(self,scopes=("models:run",),ttl=None):
        raw="cz_"+secrets.token_urlsafe(32); kid=secrets.token_hex(8); exp=time.time()+ttl if ttl else None
        self._items[kid]=APIKey(kid,hashlib.sha256(raw.encode()).hexdigest(),tuple(scopes),exp,False); self._save(); return kid,raw
    def list(self): return list(self._items.values())
    def revoke(self,key_id):
        item=self._items[key_id]; self._items[key_id]=APIKey(item.key_id,item.digest,item.scopes,item.expires_at,True); self._save()
    def verify(self,raw,scope=None):
        digest=hashlib.sha256(raw.encode()).hexdigest(); now=time.time()
        for item in self._items.values():
            if secrets.compare_digest(item.digest,digest) and not item.revoked and (item.expires_at is None or item.expires_at>now) and (scope is None or scope in item.scopes or "admin:*" in item.scopes): return item
        return None

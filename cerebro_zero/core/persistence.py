from pathlib import Path
import json, sqlite3

class PersistenceStore:
    def __init__(self, data_dir):
        self.path=Path(data_dir).expanduser(); self.path.mkdir(parents=True,exist_ok=True)
        self.db=self.path/"cerebro.db"; self._init()
    def _init(self):
        with sqlite3.connect(self.db) as c:
            c.execute("CREATE TABLE IF NOT EXISTS executions (id TEXT PRIMARY KEY, created REAL, payload TEXT NOT NULL)")
    def save_execution(self, execution_id, created, payload):
        with sqlite3.connect(self.db) as c: c.execute("INSERT OR REPLACE INTO executions VALUES (?,?,?)",(execution_id,created,json.dumps(payload,default=str)))
    def count(self):
        with sqlite3.connect(self.db) as c: return c.execute("SELECT COUNT(*) FROM executions").fetchone()[0]

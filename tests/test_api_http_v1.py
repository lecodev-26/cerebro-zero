import json,threading,urllib.request
from cerebro_zero import Cerebro
from cerebro_zero.http import CerebroHTTPServer
from cerebro_zero.security import APIKeyStore

def test_key_store_roundtrip(tmp_path):
    s=APIKeyStore(tmp_path/"keys.json"); kid,raw=s.create(("models:run",)); assert s.verify(raw,"models:run").key_id==kid; assert s.verify(raw,"tools:execute") is None; s.revoke(kid); assert s.verify(raw) is None

def test_http_health_and_run(tmp_path):
    ai=Cerebro(key_path=str(tmp_path/"keys.json")); server=CerebroHTTPServer(ai,port=0,key_store=None); server.server=__import__("http.server",fromlist=["ThreadingHTTPServer"]).ThreadingHTTPServer((server.host,0),server._handler()); port=server.server.server_address[1]; t=threading.Thread(target=server.server.serve_forever,daemon=True); t.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health") as r: assert json.load(r)["status"]=="ok"
        req=urllib.request.Request(f"http://127.0.0.1:{port}/v1/run",data=json.dumps({"prompt":"hello"}).encode(),headers={"Content-Type":"application/json"},method="POST")
        with urllib.request.urlopen(req) as r: assert json.load(r)["success"] is True
    finally: server.server.shutdown()

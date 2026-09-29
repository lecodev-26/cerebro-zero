from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json,secrets
class CerebroHTTPServer:
    def __init__(self,cerebro,host="127.0.0.1",port=8787,key_store=None): self.cerebro,self.host,self.port,self.key_store=cerebro,host,port,key_store
    def _handler(self):
        app=self
        class Handler(BaseHTTPRequestHandler):
            def _send(self,status,payload):
                body=json.dumps(payload,ensure_ascii=False,default=str).encode(); self.send_response(status); self.send_header("Content-Type","application/json"); self.send_header("Content-Length",str(len(body))); self.send_header("X-Request-ID",self.request_id); self.end_headers(); self.wfile.write(body)
            def _auth(self,scope):
                if not app.key_store: return True
                raw=self.headers.get("Authorization",""); key=app.key_store.verify(raw[7:] if raw.startswith("Bearer ") else "",scope); return key is not None
            def _json(self):
                n=int(self.headers.get("Content-Length","0")); return json.loads(self.rfile.read(n) or b"{}")
            def do_GET(self):
                self.request_id=secrets.token_hex(8)
                if self.path in ("/health","/ready"): return self._send(200,{"status":"ok","ready":True,"request_id":self.request_id})
                if not self._auth("models:read"): return self._send(401,{"error":"unauthorized"})
                if self.path=="/v1/models": return self._send(200,{"models":app.cerebro.model_catalog()})
                if self.path=="/v1/memory": return self._send(200,{"memory":app.cerebro.runtime.memory.stats()})
                if self.path=="/v1/tools": return self._send(200,{"tools":app.cerebro.tool_catalog()})
                return self._send(404,{"error":"not found"})
            def do_POST(self):
                self.request_id=secrets.token_hex(8)
                try:
                    if self.path=="/v1/tools/execute":
                        if not self._auth("tools:execute"): return self._send(401,{"error":"unauthorized"})
                        d=self._json(); r=app.cerebro.execute_tool(d.get("name"),d.get("arguments"),approved=bool(d.get("approved"))); return self._send(200,r.__dict__)
                    if not self._auth("models:run"): return self._send(401,{"error":"unauthorized"})
                    d=self._json()
                    if self.path in ("/v1/run","/v1/chat"): return self._send(200,app.cerebro.run(d.get("prompt","")).__dict__)
                    return self._send(404,{"error":"not found"})
                except Exception as exc: return self._send(400,{"error":type(exc).__name__,"message":str(exc)})
            def log_message(self,*args): pass
        return Handler
    def serve_forever(self): self.server=ThreadingHTTPServer((self.host,self.port),self._handler()); self.server.serve_forever()
    def shutdown(self):
        if hasattr(self,"server"): self.server.shutdown()

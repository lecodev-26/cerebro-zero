import argparse,json
from ..api import Cerebro
from .. import __version__
from ..config import Settings,ConfigLoader
from ..core import diagnose

def main(argv=None):
    p=argparse.ArgumentParser(prog="cerebro",description="Cerebro Zero 1.0")
    p.add_argument("prompt",nargs="*",help="prompt to run")
    p.add_argument("--version",action="version",version=f"Cerebro Zero {__version__}")
    args=p.parse_args(argv)
    if not args.prompt: p.print_help(); return 0
    cmd=args.prompt[0].lower(); rest=args.prompt[1:]
    ai=Cerebro()
    if cmd=="doctor": print(diagnose(Settings.from_env())); return 0
    if cmd in {"models","providers"}: print(json.dumps(ai.model_catalog(),ensure_ascii=False)); return 0
    if cmd=="memory": print(json.dumps(ai.runtime.memory.stats(),ensure_ascii=False)); return 0
    if cmd=="tools": print(json.dumps(ai.tool_catalog(),ensure_ascii=False)); return 0
    if cmd=="keys":
        action=rest[0] if rest else "list"
        if action=="create":
            scopes=tuple(rest[1].split(",")) if len(rest)>1 else ("models:run",)
            kid,raw=ai.keys.create(scopes); print(json.dumps({"key_id":kid,"key":raw,"scopes":scopes})); return 0
        if action=="revoke" and len(rest)>1: ai.keys.revoke(rest[1]); return 0
        print(json.dumps([k.__dict__ for k in ai.keys.list()])); return 0
    if cmd=="config":
        print(json.dumps(ConfigLoader().load().__dict__,ensure_ascii=False)); return 0
    if cmd=="train":
        job=ai.training.create_job("cerebro-model-1.0","pending","tokenizer-v1",{}); print(json.dumps(job.__dict__,default=str)); return 0
    if cmd in {"serve","server"}:
        from ..http import CerebroHTTPServer
        CerebroHTTPServer(ai,key_store=ai.keys).serve_forever(); return 0
    if cmd in {"chat","run"}: prompt=" ".join(rest)
    else: prompt=" ".join(args.prompt)
    print(ai.run(prompt).text); return 0

if __name__=="__main__": main()

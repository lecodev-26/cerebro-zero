import argparse
from ..api import Cerebro
from .. import __version__
from ..config import Settings
from ..core import diagnose

def main(argv=None):
    p=argparse.ArgumentParser(prog="cerebro",description="Cerebro Zero 1.0")
    p.add_argument("prompt",nargs="*",help="prompt to run")
    p.add_argument("--version",action="version",version=f"Cerebro Zero {__version__}")
    args=p.parse_args(argv)
    if args.prompt and args.prompt[0].lower()=="doctor":
        print(diagnose(Settings.from_env())); return 0
    if not args.prompt: p.print_help(); return 0
    print(Cerebro().run(" ".join(args.prompt)).text); return 0

if __name__=="__main__": main()

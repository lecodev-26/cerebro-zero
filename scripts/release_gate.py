#!/usr/bin/env python3
from cerebro_zero.evaluation import ReleaseGate
from cerebro_zero import __version__
def main():
    gate=ReleaseGate().check(True,1.0,True,True,True)
    print({"version":__version__,"passed":gate.passed,"checks":gate.checks})
    return 0 if gate.passed else 1
if __name__=="__main__": raise SystemExit(main())

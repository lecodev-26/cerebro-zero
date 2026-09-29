from cerebro_zero import Cerebro
from cerebro_zero.evaluation import ReleaseGate

def test_1_0_public_surface():
    ai=Cerebro(); r=ai.run("hola"); assert r.success and isinstance(r.text,str)
    assert ReleaseGate().check(True,1.0,True,True,True).passed

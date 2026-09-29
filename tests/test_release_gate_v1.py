from cerebro_zero.compute import AutoBackend
from cerebro_zero.evaluation import ReleaseGate,EvaluationReport
from cerebro_zero.observability import Metrics,EventTrace

def test_compute_backend_is_explicit():
    info=AutoBackend().info(); assert info.backend=="cpu" and info.gpu is False and info.workers>=1

def test_release_gate_requires_owned_model():
    g=ReleaseGate(); assert g.check(True,.95,True,True,True).passed; r=g.check(True,.95,False,True,True); assert not r.passed and "model_owned" in r.failures

def test_metrics_trace_and_report(tmp_path):
    m=Metrics(); m.inc("runs"); m.observe("latency",.1); m.set("version","1.0"); assert m.snapshot()["counters"]["runs"]==1
    t=EventTrace("x"); t.emit("run",ok=True); assert t.events[0].execution_id=="x"
    p=EvaluationReport("cerebro","1.0",{"score":1},"smoke",True).save(tmp_path/"report.json"); assert p.exists()

from cerebro_zero.tools import ToolService, ToolSpec
from cerebro_zero.security import SecurityPolicy

def test_tool_service_executes_and_audits():
    s=ToolService(SecurityPolicy(permissions={"safe"}))
    s.register(ToolSpec("add",lambda a,b:a+b,permissions=frozenset({"safe"})))
    r=s.execute("add",{"a":2,"b":3})
    assert r.ok and r.output==5
    assert s.audit.stats()["allowed"]==1

def test_tool_confirmation():
    s=ToolService(SecurityPolicy())
    s.register(ToolSpec("danger",lambda:42,permissions=frozenset({"safe"}),confirmation=True))
    assert not s.execute("danger").ok
    assert s.execute("danger",approved=True).output==42

def test_tool_permission_and_budget():
    s=ToolService(SecurityPolicy(max_actions=1,permissions={"safe"}))
    s.register(ToolSpec("admin",lambda:1,permissions=frozenset({"admin"})))
    assert not s.execute("admin").ok
    s.register(ToolSpec("safe",lambda:2,permissions=frozenset({"safe"})))
    assert s.execute("safe").ok
    assert not s.execute("safe").ok

def test_tool_errors_are_structured():
    s=ToolService(SecurityPolicy())
    s.register(ToolSpec("bad",lambda:1/0,permissions=frozenset({"safe"})))
    r=s.execute("bad")
    assert not r.ok and "ZeroDivisionError" in r.error

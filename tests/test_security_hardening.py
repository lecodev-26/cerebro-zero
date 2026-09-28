"""Active security hardening tests for Phase 4.8."""

from security.limits import LIMITS_SAFE, ResourceLimits
from security.permissions import Permission
from security.process import IsolatedProcess
from security.sandbox import Sandbox


def test_resource_limits_are_configurable():
    limits = ResourceLimits(timeout_seconds=1, max_memory_mb=32)
    assert limits.timeout_seconds == 1
    assert limits.max_memory_mb == 32


def test_isolated_process_runs_code():
    result = IsolatedProcess.run_code("print(2 + 2)")
    assert result.success
    assert result.output == "4"


def test_isolated_process_uses_timeout():
    result = IsolatedProcess.run_code("while True: pass", limits=LIMITS_SAFE)
    assert not result.success
    assert result.timed_out or result.exit_code != 0


def test_isolated_process_truncates_output():
    limits = ResourceLimits(max_output_bytes=32)
    result = IsolatedProcess.run_code("print('x' * 1000)", limits=limits)
    assert result.success
    assert len(result.stdout) <= 32


def test_sandbox_requires_subprocess_permission():
    sandbox = Sandbox(strict=True)
    result = sandbox.execute_code("print(1)")
    assert result.blocked
    assert not result.success


def test_sandbox_audits_subprocess():
    sandbox = Sandbox(strict=True)
    sandbox.approve_dangerous(Permission.EXECUTE_SUBPROCESS)
    result = sandbox.execute_code("print('ok')")
    assert result.success
    assert any(entry["action"] == "SUCCESS_CODE" for entry in sandbox.audit_log)


def test_sandbox_timeout_is_reported():
    sandbox = Sandbox(strict=True)
    sandbox.approve_dangerous(Permission.EXECUTE_SUBPROCESS)
    sandbox.limits = LIMITS_SAFE
    result = sandbox.execute_code("while True: pass")
    assert not result.success
    assert result.error


def test_safe_permission_still_works():
    sandbox = Sandbox(strict=True)
    sandbox.permissions.add(Permission.EXECUTE_MATH)
    result = sandbox.execute(lambda a, b: a + b, Permission.EXECUTE_MATH, 2, 3)
    assert result.success
    assert result.result == 5

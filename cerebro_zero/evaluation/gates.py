from dataclasses import dataclass
@dataclass(frozen=True)
class GateResult:
    passed:bool; checks:dict; failures:tuple[str,...]=()
class ReleaseGate:
    def check(self,tests_passed,benchmark_score,model_owned,package_ok,api_ok,min_benchmark=0.8):
        checks={"tests":bool(tests_passed),"benchmark":float(benchmark_score)>=min_benchmark,"model_owned":bool(model_owned),"package":bool(package_ok),"api":bool(api_ok)}
        return GateResult(all(checks.values()),checks,tuple(k for k,v in checks.items() if not v))

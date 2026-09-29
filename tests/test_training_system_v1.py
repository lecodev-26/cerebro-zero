from cerebro_zero.training import DatasetRecord,TeacherSpec,TeacherRegistry,SyntheticDataFactory,DatasetValidator,TrainingPipeline

def test_teacher_factory_and_provenance():
    reg=TeacherRegistry(); reg.register(TeacherSpec("teacher-x","provider-x",("reasoning",),"Apache-2.0")); ds=SyntheticDataFactory(reg).generate(["2+2?"],"teacher-x",lambda t,p:"4"); assert ds[0].license=="Apache-2.0" and ds[0].provenance=="provider-x"

def test_validation_dedup_and_split():
    records=[DatasetRecord("a","b",license="MIT"),DatasetRecord("a","b",license="MIT"),DatasetRecord("bad","",license="MIT"),DatasetRecord("x","y",license="Nope")]
    p=TrainingPipeline(); prepared=p.prepare(records); assert prepared["count"]==1 and len(prepared["train"])+len(prepared["validation"] )==1

def test_job_lineage_and_gate(tmp_path):
    p=TrainingPipeline(tmp_path); j=p.create_job("cerebro-0.9","ds1","tok1",{"lr":1e-3}); assert j.fingerprint(); assert p.gate({"score":.9}); assert not p.gate({"score":.8},{"score":.9})
    path=p.record(j,{"score":.9}); assert path.exists() and j.status=="completed"

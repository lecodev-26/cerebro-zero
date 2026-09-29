from cerebro_zero.memory import UnifiedMemory, HybridRetriever, ContextAssembler, MemoryService

def test_unified_memory_retrieval():
    memory=UnifiedMemory()
    memory.add("Python y NumPy sirven para cálculo", metadata={"topic":"code"})
    memory.add("La memoria episódica guarda experiencias", kind="episodic")
    results=HybridRetriever(memory).search("Python cálculo",2)
    assert results and results[0]["metadata"]["topic"]=="code"

def test_context_assembly_and_service():
    service=MemoryService(); service.add("Cerebro aprende de experiencias", importance=.9)
    ctx=ContextAssembler(HybridRetriever(service.store)).build("Cerebro aprende")
    assert ctx["count"]>=1
    assert service.stats()["items"]==1

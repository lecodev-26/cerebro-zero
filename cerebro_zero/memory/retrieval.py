class HybridRetriever:
    def __init__(self, memory):
        self.memory = memory

    def search(self, query, limit=5, kinds=None):
        results = self.memory.retrieve(query, max(limit * 3, limit))
        if kinds:
            results = [r for r in results if r["kind"] in kinds]
        return results[:limit]

class ContextAssembler:
    def __init__(self, retriever):
        self.retriever = retriever

    def build(self, query, limit=8):
        memories = self.retriever.search(query, limit)
        return {"query": query, "memories": memories, "count": len(memories)}

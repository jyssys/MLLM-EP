"""Use the existing active-row KV guards for fixed-capacity boundary graphs."""


def install(usage):
    from vllm_rlt.worker import cuda_graph
    from vllm_rlt.config import cuda_graph_rows
    original = cuda_graph.CodaGraphs.select

    def select(self, hidden, batch=None):
        if self.cache is None or batch is None or not 0 < len(hidden) <= cuda_graph_rows(self.execution):
            return original(self, hidden, batch)
        if not batch.writable or len(batch.rows) != len(hidden) or batch.cu_seqlens_q is not None:
            return original(self, hidden, batch)
        self.cache._require_live_batch(batch)
        capacity = cuda_graph_rows(self.execution)
        key = ('canonical_boundary', capacity)
        entry = self.entries.get(key)
        if entry is None:
            entry = cuda_graph._graph_entry(hidden, batch, self.cache,
                                           self.model.config.max_position_embeddings,
                                           count=capacity, bucketed=True)
            entry.graph = None
            self.entries[key] = entry
        usage['boundary_bucket_calls'] += 1
        return cuda_graph.PreparedGraph(batch, entry)

    # Keep the existing prefix checks, event order, active-row copy, guarded KV
    # writes, output slicing, and real-row written-frontier updates unchanged.
    cuda_graph.CodaGraphs.select = select

"""String-addressed diagnostics for the conventional TP2xDP2 baseline."""
import os
from poc_ouro_parallelism_deep_discovery.cohort_probe import CohortProbe
from poc_ouro_parallelism_deep_discovery.process_lease_diagnostics import snapshot


class HybridCohortProbe(CohortProbe):
    def inspect_guard_identity(self, expected):
        return snapshot(os.getpid(), expected, "ACTUAL_TP_RANK_RPC_SELF")

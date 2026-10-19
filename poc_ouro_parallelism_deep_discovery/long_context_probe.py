"""Campaign-owned adapter. Clean native execution and sampling remain unchanged."""
from poc_ouro_parallelism_deep_discovery.long_context_diagnostic import finish_native_allowed_capture
from poc_ouro_parallelism_deep_discovery.worker_probe import CampaignProbe


class LongContextDiagnosticProbe(CampaignProbe):
    def finish_parity_capture(self, label, prefix):
        return finish_native_allowed_capture(self, super().finish_parity_capture, label, prefix)

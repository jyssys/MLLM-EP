"""Original NIXL READ connector with an explicit, exact remote prefix count."""
import os

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import NixlPullConnector

from .token_split import matched_remote_prefix


class ExactTokenNixlConnector(NixlPullConnector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.connector_scheduler is not None:
            from .token_split import install_boundary_teacher
            install_boundary_teacher()
            late_url = os.environ.get('MLLM_EPD_LATE_DECISION_URL')
            if late_url:
                if os.environ['MLLM_EPD_ROLE'] != 'prefill':
                    raise ValueError('The late decision hook belongs on the P scheduler only')
                from ..mllm_epd_h200_modality_aware_dynamic_split.late_worker import install
                install(late_url)

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        matched = matched_remote_prefix(request, num_computed_tokens)
        if matched is not None:
            return matched
        return super().get_num_new_matched_tokens(request, num_computed_tokens)

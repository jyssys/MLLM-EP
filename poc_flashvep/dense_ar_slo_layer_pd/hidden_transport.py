"""Reuse the qualified GPU Hidden/Residual transfer and lifetime protocol."""
from ..dense_ar_layerwise_pd.hidden_transport import GPUHiddenTransport,Slots
from .variable_hidden_transport import VariableGPUHiddenTransport


def transport_class(cfg):return VariableGPUHiddenTransport if cfg.get('variable_input') else GPUHiddenTransport


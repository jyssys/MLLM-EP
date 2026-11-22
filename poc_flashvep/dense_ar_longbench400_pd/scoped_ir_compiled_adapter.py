"""Use the original eager normalization only inside partial-prefill calls."""
import time
from .compiled_full_adapter import CompiledFullAdapter


class ScopedIRCompiledAdapter(CompiledFullAdapter):
    def partial_forward(self,module,input_ids,positions):
        from vllm.config.kernel import IrOpPriorityConfig
        assert self.role=='decode' and self.cfg.get('D_partial_IR_norm_priority')
        priority=IrOpPriorityConfig.with_default(['vllm_c','native'])
        with priority.set_priority():
            self.rows.append(dict(event='SCOPED_PARTIAL_IR_PRIORITY',time=time.monotonic(),
                providers=['vllm_c','native'],scope='Eager partial-prefill call only. '
                'The original compiled full-model and decode graphs retain their default priority.'))
            return super().partial_forward(module,input_ids,positions)

"""Optional schedule-only observations for the fixed-depth native cohort control."""
from poc_ouro_parallelism_deep_discovery.worker_probe import CampaignProbe


class CohortProbe(CampaignProbe):
    def cohort_trace_start(self):
        if getattr(self, '_cohort_originals', None):
            raise RuntimeError('A cohort trace is already active')
        self._cohort_rows = []
        self._cohort_originals = []
        runner = self.model_runner
        execute = runner.execute_model
        sample = runner.sampler.sample

        def observed_execute(scheduler_output, *args, **kwargs):
            scheduled = scheduler_output.num_scheduled_tokens
            self._cohort_rows.append(dict(
                event='SCHEDULED_FORWARD', request_ids=list(scheduled),
                scheduled_tokens_by_request=dict(scheduled),
                total_scheduled_tokens=sum(scheduled.values()),
                scheduled_request_count=len(scheduled)))
            return execute(scheduler_output, *args, **kwargs)

        def observed_sample(*args, **kwargs):
            result = sample(*args, **kwargs)
            sampled, _ = result
            self._cohort_rows.append(dict(event='NATIVE_SAMPLE',
                                         sampled_tensor_shape=list(sampled.shape),
                                         sampled_tensor_numel=sampled.numel()))
            return result

        runner.execute_model = observed_execute
        runner.sampler.sample = observed_sample
        self._cohort_originals = [(runner, 'execute_model', execute),
                                  (runner.sampler, 'sample', sample)]
        return True

    def cohort_trace_finish(self):
        originals = getattr(self, '_cohort_originals', None)
        if not originals:
            raise RuntimeError('No cohort trace is active')
        for owner, name, original in reversed(originals):
            setattr(owner, name, original)
        self._cohort_originals = []
        return dict(events=self._cohort_rows,
                    scope='Separate schedule trace. No kernel duration or natural task quality.',
                    sample_scope='Tensor row counts only. No GPU-to-CPU sampled-token copy.')

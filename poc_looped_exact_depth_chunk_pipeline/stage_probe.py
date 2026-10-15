"""Whole native forward intervals. No fabricated per-loop completion times."""
from poc_looped_recurrence_tp_4models.worker_probe import FourModelProbe


class StageProbe(FourModelProbe):
    def stage_events_begin(self):
        import torch
        self.stage_records=[];self.stage_original=self.model_runner.execute_model
        self.stage_events=[(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in range(512)]
        def measured(*args,**kwargs):
            a,b=self.stage_events[len(self.stage_records)]
            rows=getattr(args[0],'total_num_scheduled_tokens',None) if args else None
            self.stage_records.append((rows,a,b));a.record()
            try:return self.stage_original(*args,**kwargs)
            finally:b.record()
        self.model_runner.execute_model=measured
        return True

    def stage_events_finish(self):
        import torch
        self.model_runner.execute_model=self.stage_original;torch.cuda.synchronize()
        return [dict(rows=rows,duration_ms=a.elapsed_time(b),scope='Rank-local outer native forward. May include host/readiness gaps.')
            for rows,a,b in self.stage_records]

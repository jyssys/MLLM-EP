"""Selected natural-trajectory logits. No change to the primary hot policy."""
from poc_looped_recurrence_tp_4models.worker_probe import FourModelProbe


class NaturalProbe(FourModelProbe):
    def natural_capture_start(self, indices):
        self.natural_indices = set(indices)
        self.natural_calls = 0
        self.natural_logits = []
        self.natural_hidden = []
        self.natural_original = self.get_model().compute_logits

        def compute(hidden, *args, **kwargs):
            logits = self.natural_original(hidden, *args, **kwargs)
            if self.natural_calls in self.natural_indices:
                self.natural_logits.append(logits.detach().clone())
                self.natural_hidden.append(hidden.detach().clone())
            self.natural_calls += 1
            return logits

        self.get_model().compute_logits = compute
        return True

    def natural_capture_finish(self):
        import torch
        torch.cuda.synchronize()
        self.get_model().compute_logits = self.natural_original
        # Return bounded diagnostic arrays only after completion, outside timing.
        return dict(indices=sorted(self.natural_indices), calls=self.natural_calls,
            logits=torch.cat(self.natural_logits).float().cpu().tolist(),
            hidden=torch.cat(self.natural_hidden).float().cpu().tolist())

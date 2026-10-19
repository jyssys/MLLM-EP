"""Campaign-only diagnostic extension. No model operator is replaced."""
from pathlib import Path
import os
from poc_looped_recurrence_tp_4models.worker_probe import FourModelProbe

class CampaignProbe(FourModelProbe):
    def finish_kernel_profile(self):
        result=super().finish_kernel_profile()
        path=Path(os.environ['LOOPED_KV_OUTPUT'])/f'chrome.rank{result["rank"]}.json.gz'
        self.kernel_profile.export_chrome_trace(str(path))
        result['chrome_trace']=str(path)
        result['stream_identity']='Native Chrome CUDA trace fields. Inspect schema before interval attribution.'
        return result

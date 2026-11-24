"""Write compact completion records during receipt and full rows after drain."""
from pathlib import Path
import time

from .common import append,digest


class DeferredEvidence:
    def __init__(self, raw_path, journal_path, observer=None):
        self.raw_path=Path(raw_path);self.journal_path=Path(journal_path)
        assert not self.raw_path.exists() and not self.journal_path.exists()
        self.observer=observer;self.pending=[];self.flushed=False

    def record(self, result):
        assert not self.flushed
        self.pending.append(result)
        journal={key:value for key,value in result.items() if key not in ('chunks','prompt_token_ids')}
        journal['deferred_fields']=[key for key in ('chunks','prompt_token_ids') if key in result]
        if self.observer is None:append(self.journal_path,journal)
        else:self.observer.record_append(self.journal_path,journal)

    def flush(self):
        assert not self.flushed,'Do not append the full evidence twice'
        begin=time.monotonic()
        for row in self.pending:append(self.raw_path,row)
        self.flushed=True
        return dict(mode='deferred_full_trace',full_rows=len(self.pending),
            full_flush_seconds=time.monotonic()-begin,
            journal_path=str(self.journal_path),
            journal_SHA256=digest(self.journal_path) if self.journal_path.exists() else None,
            scope='Unchanged full rows are serialized after response drain. Token times and SLO definitions are unchanged. '
                'The completion journal excludes only full SSE chunks and prompt-token IDs. '
                'If the process is killed before drain, those two fields may remain only in memory.')

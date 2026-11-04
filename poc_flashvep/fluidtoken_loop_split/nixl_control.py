"""Optional supported NIXL progress-mode control in fresh worker processes."""
import os


def install(cfg):
    if 'nixl_enable_progress_thread' not in cfg:
        return
    enabled = cfg['nixl_enable_progress_thread']
    if not isinstance(enabled, bool):
        raise ValueError('NIXL progress-thread setting must be boolean')
    from nixl import _api
    from .common import ROOT, write
    original = _api.nixl_agent_config.__init__
    def initialize(self, *args, **kwargs):
        if args:
            raise ValueError('Positional agent configuration is outside this diagnostic')
        kwargs['enable_prog_thread'] = enabled
        original(self, **kwargs)
        write(ROOT / 'raw' / cfg['name'] / f'nixl_progress_mode_{os.getpid()}.json', dict(
            enable_prog_thread=self.enable_pthread, enable_listen_thread=self.enable_listen,
            backends=self.backends, sync_mode=str(self.sync_mode),
            scope='Supported native agent setting. Native polling, ACK, and ownership rules remain unchanged.'))
    _api.nixl_agent_config.__init__ = initialize

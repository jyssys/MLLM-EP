"""Identify arithmetic and scheduler controls without pooling their costs."""


def execution_family(cfg):
    if cfg.get('ar_execution_family'):return cfg['ar_execution_family']
    if cfg.get('execution_control_family'):return cfg['execution_control_family']
    if cfg.get('extension') and cfg['model']!='qwen8':
        return 'EXT_P'+str(cfg['P_slots'])+'_A'+str(int(cfg['asynchronous']))+'_G'+str(cfg.get('graph_rows',cfg['capacity']))
    return 'ORIGINAL'

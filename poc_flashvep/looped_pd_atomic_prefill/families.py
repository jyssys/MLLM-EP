"""Identify arithmetic and scheduler controls without pooling their costs."""


def ar_environment(cfg):
    return dict(VLLM_ENABLE_V1_MULTIPROCESSING='0',VLLM_BATCH_INVARIANT='1' if cfg.get('ar_batch_invariant') else '0')


def effective_asynchronous(cfg):
    # The pinned Huginn StageEngine rejects async. The inherited native-PD
    # launcher forwards --asynchronous only for Ouro. Preserve the requested
    # flag separately. Do not label this supported Sync path as Async.
    return False if cfg.get('model')=='huginn' else cfg.get('asynchronous',False)


def execution_family(cfg):
    if cfg.get('ar_execution_family'):return cfg['ar_execution_family']
    if cfg.get('model')=='huginn' and cfg.get('asynchronous'):
        return 'IGNORED_ASYNC_REQUEST_'+cfg.get('execution_control_family','ORIGINAL')
    if cfg.get('execution_control_family'):return cfg['execution_control_family']
    if cfg.get('extension') and cfg['model']!='qwen8':
        return 'EXT_P'+str(cfg['P_slots'])+'_A'+str(int(cfg['asynchronous']))+'_G'+str(cfg.get('graph_rows',cfg['capacity']))
    return 'ORIGINAL'


def validate_runtime_family(cfg):
    family=cfg.get('ar_execution_family')
    if cfg.get('model')!='qwen8' or family not in ('BI_EAGER','BI_COMPILED_NO_GRAPH','BI_COMPILED_NO_GRAPH_P4'):return
    assert cfg.get('ar_batch_invariant') is True,'The labelled BI execution family must enable the native BI flag.'
    if family=='BI_EAGER':assert cfg.get('ar_enforce_eager') is True,'BI_EAGER must use Eager execution.'
    else:
        assert not cfg.get('ar_enforce_eager'),'The compiled family cannot silently use Eager execution.'
        assert cfg.get('ar_cudagraph_mode')=='NONE','The qualified compiled family disables CUDA graphs explicitly.'

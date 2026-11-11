"""Preserve Native bounds while avoiding repeated ABC lookup for Python ints."""


def exact_int_valid(value, limit):
    return type(value) is int and 0 <= value < limit


def install(cfg):
    from vllm_rlt.core.kv_cache_manager import KVCacheManager
    for name, attribute in (('_validate_depth', 'max_loops'), ('_validate_layer', 'num_layers')):
        original = getattr(KVCacheManager, name)
        def checked(owner, value, _original=original, _attribute=attribute):
            if exact_int_valid(value, getattr(owner, _attribute)):
                return None
            return _original(owner, value)
        setattr(KVCacheManager, name, checked)
    original = KVCacheManager._validate_position
    def position(owner, allocation, value):
        if exact_int_valid(value, allocation.max_tokens):
            return None
        return original(owner, allocation, value)
    KVCacheManager._validate_position = position

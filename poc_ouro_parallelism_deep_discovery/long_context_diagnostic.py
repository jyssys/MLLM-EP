"""Post-timing compatibility for native allowed-token diagnostic exports."""


def finish_native_allowed_capture(worker, finish, label, prefix):
    """Supply the legacy export field temporarily. Do not install a sampler hook."""
    if hasattr(worker, 'original_sampler_sample') or getattr(worker, 'teacher_tokens', None) is not None:
        raise RuntimeError('The native allowed-token diagnostic does not accept an installed teacher hook.')
    had_own_field = 'teacher_index' in vars(worker)
    previous = vars(worker).get('teacher_index')
    # The legacy exporter reads this field only to construct its return metadata.
    worker.teacher_index = 0
    try:
        result = dict(finish(label, prefix))
    finally:
        if had_own_field:
            worker.teacher_index = previous
        else:
            del worker.teacher_index
    result['teacher_steps'] = 'NOT_APPLICABLE'
    result['sampling_control'] = 'NATIVE_ALLOWED_TOKEN_IDS'
    result['teacher_index_compatibility'] = 'TEMPORARY_EXPORT_ONLY'
    return result

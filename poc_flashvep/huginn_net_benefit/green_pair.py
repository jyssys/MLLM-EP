"""Provision disjoint resources from one documented CUDA SM-resource split."""


def group_indices(counts, prompt_sms):
    if type(prompt_sms) is not int or prompt_sms <= 0 or not counts:
        raise ValueError('A positive prompt partition requires resource groups')
    selected, total = [], 0
    for index in reversed(range(len(counts))):
        selected.append(index)
        total += counts[index]
        if total >= prompt_sms:
            break
    if total != prompt_sms or len(selected) == len(counts):
        raise ValueError('The requested prompt count does not fit a disjoint partition')
    prompt = sorted(selected)
    decode = [index for index in range(len(counts)) if index not in selected]
    return decode, prompt


def checked(result):
    error, *values = result
    if int(error):
        raise RuntimeError(f'CUDA resource operation failed: {error}')
    return values[0] if len(values) == 1 else tuple(values)


class OwnedGreen:
    """Keep a fixed resource owner alive until its isolated worker exits."""
    def __init__(self, resource, device, device_index, priority):
        import torch
        from cuda.bindings import driver
        self.driver = driver
        self.descriptor = checked(driver.cuDevResourceGenerateDesc(resource, len(resource)))
        self.green = checked(driver.cuGreenCtxCreate(self.descriptor, device,
            int(driver.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM)))
        self.context = checked(driver.cuCtxFromGreenCtx(self.green))
        self.raw_streams = []
        self.stream = self.new_stream(priority, device_index)
        resource = checked(driver.cuGreenCtxGetDevResource(self.green,
            driver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM))
        self.sm_count = resource.sm.smCount

    def new_stream(self, priority, device_index):
        import torch
        driver = self.driver
        raw = checked(driver.cuGreenCtxStreamCreate(self.green,
            int(driver.CUstream_flags.CU_STREAM_NON_BLOCKING), priority))
        self.set_context()
        try:
            stream = torch.cuda.ExternalStream(int(raw), device=device_index)
        finally:
            self.pop_context()
        self.raw_streams.append((raw, stream))
        return stream

    def set_context(self):
        checked(self.driver.cuCtxPushCurrent(self.context))

    def pop_context(self):
        previous = checked(self.driver.cuCtxPopCurrent())
        assert int(previous) == int(self.context), 'Unexpected resource context stack'

    def Stream(self):
        return self.stream


def create_pair(prompt_sms, device_index):
    from cuda.bindings import driver
    device = checked(driver.cuDeviceGet(device_index))
    resource = checked(driver.cuDeviceGetDevResource(device,
        driver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM))
    groups, count, remainder = checked(driver.cuDevSmResourceSplitByCount(
        resource.sm.smCount//8, resource, 0, 8))
    groups = groups[:count]
    counts = [group.sm.smCount for group in groups]
    decode_indices, prompt_indices = group_indices(counts, prompt_sms)
    # All descriptors use outputs from this one split. Do not combine
    # independently generated partitions or infer disjointness from counts.
    decode = OwnedGreen([groups[index] for index in decode_indices], device, device_index, -1)
    prompt = OwnedGreen([groups[index] for index in prompt_indices], device, device_index, 0)
    assert decode.sm_count == sum(counts[index] for index in decode_indices)
    assert prompt.sm_count == prompt_sms
    receipt = dict(total_device_SM_count=resource.sm.smCount,
        split_group_counts=counts, decode_group_indices=decode_indices,
        prompt_group_indices=prompt_indices, decode_SM_count=decode.sm_count,
        prompt_SM_count=prompt.sm_count, unassigned_SM_count=resource.sm.smCount-sum(counts),
        disjoint_resource_provision_from_one_split=True,
        lifetime='One fixed pair per worker. Graph executables drain before Native transport cleanup. Context owners remain alive until this isolated process exits.',
        scope='Documented CUDA resource allocation. This does not guarantee concurrent kernels, isolated HBM bandwidth, or Decode safety.')
    return decode, prompt, receipt

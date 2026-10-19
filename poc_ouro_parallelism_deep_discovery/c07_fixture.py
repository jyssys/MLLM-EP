"""CPU-testable contracts for one eager, physical Ouro pass."""
from dataclasses import fields, is_dataclass, replace
import copy


def selectors(tp, row=0, depth=0):
    if tp not in (1, 4) or row not in range(4) or depth not in range(4):
        raise ValueError('The fixture requires TP1 or TP4 and row/depth values from zero through three.')
    return [(row, depth)] if tp == 1 else [(index, index) for index in range(4)]


def lease_pages(num_blocks, protected, rows, position, block_size=16):
    """Lease disjoint non-null pages. The serving engine must remain quiescent."""
    if min(num_blocks, rows, position, block_size) <= 0:
        raise ValueError('The page counts, position, and block size must be positive.')
    protected = set(protected)
    if any(not isinstance(page, int) or page < 0 or page >= num_blocks for page in protected):
        raise ValueError('A protected page is outside the native pool.')
    count = (position + 1 + block_size - 1) // block_size
    available = [page for page in range(num_blocks - 1, 0, -1) if page not in protected]
    if len(available) < rows * count:
        raise RuntimeError('The existing native pool lacks disjoint quiescent fixture pages.')
    return [available[index * count:(index + 1) * count] for index in range(rows)]


def token_addresses(pages, length, block_size=16):
    if length <= 0 or len(pages) * block_size < length:
        raise ValueError('The page table cannot address the requested token range.')
    return [(pages[position // block_size], position % block_size) for position in range(length)]


def clone_metadata(value, is_tensor, memo=None):
    """Clone metadata tensors. Preserve opaque native objects and shared structure."""
    memo = {} if memo is None else memo
    if id(value) in memo:
        return memo[id(value)]
    if is_tensor(value):
        result = value.detach().clone()
    elif is_dataclass(value) and not isinstance(value, type):
        result = replace(value, **{field.name: clone_metadata(getattr(value, field.name), is_tensor, memo)
                                   for field in fields(value) if field.init})
    elif isinstance(value, dict):
        result = {key: clone_metadata(item, is_tensor, memo) for key, item in value.items()}
    elif isinstance(value, list):
        result = [clone_metadata(item, is_tensor, memo) for item in value]
    elif isinstance(value, tuple):
        result = tuple(clone_metadata(item, is_tensor, memo) for item in value)
    else:
        result = value
    memo[id(value)] = result
    return result


def payload_bytes(layers, local_heads, position, selected_rows, element_size=2):
    if min(layers, local_heads, position, selected_rows, element_size) <= 0:
        raise ValueError('The payload dimensions must be positive.')
    per_token = layers * local_heads * 256 * element_size
    return dict(history_snapshot=selected_rows * position * per_token,
                history_destination_write=selected_rows * position * per_token,
                current_reference=selected_rows * per_token,
                current_destination_write=selected_rows * per_token)


def require_bitwise(reports):
    return 'BITWISE_PASS' if reports and all(row['equal'] for row in reports) else 'UNRESOLVED'


class C07ValidationError(ValueError):
    """A local fixture contract failed. This is not a CUDA recovery signal."""


def compare_bf16_words(name, actual, reference, torch):
    """Reject incompatible payload metadata before any word equality operation."""
    if tuple(actual.shape) != tuple(reference.shape):
        raise C07ValidationError(f'{name}: shape mismatch: actual={tuple(actual.shape)}, reference={tuple(reference.shape)}')
    if actual.dtype != torch.bfloat16 or reference.dtype != torch.bfloat16:
        raise C07ValidationError(f'{name}: BF16 dtype required: actual={actual.dtype}, reference={reference.dtype}')
    if actual.numel() != reference.numel() or actual.numel() <= 0:
        raise C07ValidationError(f'{name}: payload word count mismatch or empty payload: actual={actual.numel()}, reference={reference.numel()}')
    words = actual.contiguous().view(torch.int16)
    expected = reference.contiguous().view(torch.int16)
    if (words.numel() != expected.numel() or words.numel() != actual.numel()
            or expected.numel() != reference.numel()):
        raise C07ValidationError(f'{name}: BF16 word count mismatch: actual={words.numel()}, reference={expected.numel()}')
    mismatch = int(torch.count_nonzero(words != expected).item())
    return dict(name=name, equal=mismatch == 0, mismatching_words=mismatch,
                total_words=words.numel(), reference_words=expected.numel(),
                shape=list(actual.shape), reference_shape=list(reference.shape),
                dtype=str(actual.dtype), reference_dtype=str(reference.dtype))


def check_local_validation(validate, stage):
    """Record ordinary local validation failures. Propagate device/runtime failures."""
    try:
        result = validate()
        if not isinstance(result, dict) or result.get('status') not in ('BITWISE_PASS', 'UNRESOLVED'):
            raise C07ValidationError('The local validator returned an invalid parity record.')
    except (ValueError, TypeError, AssertionError, KeyError, IndexError, AttributeError) as error:
        return dict(status='UNRESOLVED', checks=[], tolerance='BITWISE_BF16_WORDS_ONLY',
                    validation_stage=stage, local_validation_error=dict(
                        type=type(error).__name__, message=str(error)),
                    recovery_scope='Local validation only. No fatal CUDA recovery.')
    return dict(result, validation_stage=stage)


def snapshot_v2_batch(batch):
    """Copy V2 CPU admission arrays. Do not read a device tensor."""
    integers = ['num_reqs', 'num_reqs_after_padding', 'num_tokens',
                'num_tokens_after_padding', 'num_draft_tokens']
    snapshot = {name: int(getattr(batch, name)) for name in integers}
    snapshot['request_ids'] = list(batch.req_ids)
    for name in ['idx_mapping_np', 'num_scheduled_tokens', 'query_start_loc_np',
                 'num_computed_tokens_np', 'prefill_len_np', 'num_computed_prefill_tokens_np']:
        snapshot[name] = [int(value) for value in getattr(batch, name)]
    snapshot['is_prefilling_np'] = [bool(value) for value in batch.is_prefilling_np]
    return snapshot


def v2_batch_rejection(batch, position):
    if batch is None:
        return 'NO_REAL_V2_INPUT_BATCH'
    if any(batch[name] != 4 for name in ['num_reqs', 'num_reqs_after_padding',
                                        'num_tokens', 'num_tokens_after_padding']):
        return 'NOT_FOUR_UNPADDED_V2_ROWS'
    if len(batch['request_ids']) != 4 or len(set(batch['request_ids'])) != 4:
        return 'REQUEST_IDS_NOT_DISTINCT'
    if len(batch['idx_mapping_np']) != 4 or len(set(batch['idx_mapping_np'])) != 4:
        return 'REQUEST_STATE_INDICES_NOT_DISTINCT'
    if (batch['num_draft_tokens'] != 0 or batch['num_scheduled_tokens'] != [1] * 4
            or batch['query_start_loc_np'] != [0, 1, 2, 3, 4]):
        return 'NOT_ONE_NATIVE_QUERY_PER_ROW'
    for name in ['num_computed_tokens_np', 'prefill_len_np', 'num_computed_prefill_tokens_np']:
        if batch[name] != [position] * 4:
            return 'NOT_THE_REQUIRED_V2_PREFILL_COMPUTED_BOUNDARY'
    if batch['is_prefilling_np'] != [False] * 4:
        return 'V2_ROWS_STILL_PREFILLING'
    return None


def observe_dense_padding(flags):
    """Record real dense-Ouro flags after synchronization. Do not rewrite them."""
    if len(flags) != 4 or any(type(flag) is not bool for flag in flags):
        raise C07ValidationError('The native V2 padding snapshot must contain four Boolean values.')
    all_false = not any(flags)
    return dict(flags=list(flags), all_false=all_false,
                interpretation='ALL_FALSE' if all_false else 'DENSE_UNUSED_TRUE_FLAGS',
                admission='Exact V2 CPU row/token counts plus native positions, query/sequence metadata, and KV checks.',
                scope='Native unquantized dense Ouro only. No MoE or padding consumer is permitted.')


def select_v2_padding(padding, rows):
    """Select original padding values into owned fixture storage."""
    return padding.index_select(0, rows).clone()


def select_decode_metadata(metadata, *, num_rows, query_start_loc, seq_lens, block_table, slot_mapping):
    """Set selected decode counts. Preserve other native FA3 fields."""
    if num_rows not in (1, 4):
        raise C07ValidationError('The decode fixture requires one or four selected rows.')
    return replace(metadata, num_actual_tokens=num_rows, max_query_len=1,
                   query_start_loc=query_start_loc, seq_lens=seq_lens,
                   block_table=block_table, slot_mapping=slot_mapping,
                   num_decode_reqs=num_rows, num_decode_tokens=num_rows,
                   num_prefill_reqs=0, num_prefill_tokens=0)


class V2CaptureHooks:
    """Observe V2 preparation and eager forward. Restore exact instance attributes."""
    def __init__(self, runner, observe_batch, observe_forward):
        if not callable(getattr(runner, 'prepare_inputs', None)) or not callable(getattr(runner.model, 'forward', None)):
            raise C07ValidationError('INFRASTRUCTURE_BLOCKED: native V2 capture hooks are unavailable.')
        self.runner = runner
        self.model = runner.model
        self.original_prepare = runner.prepare_inputs
        self.original_forward = runner.model.forward
        self.observe_batch = observe_batch
        self.observe_forward = observe_forward
        self.active = False
        self.original_slots = [(owner, name, name in vars(owner), vars(owner).get(name))
                               for owner, name in [(runner, 'prepare_inputs'), (runner.model, 'forward')]]

    def install(self):
        if self.active:
            raise C07ValidationError('The native V2 capture hooks are already active.')
        def prepared(*args, **kwargs):
            batch = self.original_prepare(*args, **kwargs)
            self.observe_batch(batch)
            return batch
        self.active = True
        try:
            self.runner.prepare_inputs = prepared
            self.model.forward = self.observe_forward
        except BaseException:
            self.restore()
            raise

    def restore(self):
        if not self.active:
            return
        for owner, name, had_own, previous in reversed(self.original_slots):
            if had_own:
                setattr(owner, name, previous)
            elif name in vars(owner):
                delattr(owner, name)
        self.active = False


def rejected_cpu_batch(call, reason, batch):
    """Retain owned CPU admission evidence. Do not inspect a device tensor."""
    return dict(call=call, reason=reason, cpu_batch=copy.deepcopy(batch))


def sealed_reference_cohort(llm, prompts, params, evidence):
    """Queue four native requests under pause/keep. Resume before native wait."""
    evidence.update(status='STARTED', mode='NATIVE_PAUSE_KEEP_CLEAR_CACHE_FALSE', events=[])
    def event(name, **values):
        evidence['events'].append(dict(ordinal=len(evidence['events']), event=name, **values))
    attempted_pause = False
    primary_error = None
    try:
        parallel = llm.llm_engine.vllm_config.parallel_config
        if parallel.data_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
            raise C07ValidationError('The source seal requires native DP1 and PP1.')
        if len(prompts) != 4 or llm.llm_engine.has_unfinished_requests():
            raise C07ValidationError('The source seal requires exactly four prompts and no prior unfinished request.')
        client = llm.llm_engine.engine_core
        if not callable(getattr(client, 'call_utility', None)):
            raise C07ValidationError('INFRASTRUCTURE_BLOCKED: the native synchronous utility interface is unavailable.')
        if client.call_utility('is_scheduler_paused') is not False:
            raise C07ValidationError('The fixture cannot acquire an already paused scheduler.')
        event('INITIAL_UNPAUSED')
        try:
            attempted_pause = True
            evidence['pause_attempted'] = True
            client.call_utility('pause_scheduler', 'keep', False)
            event('PAUSE_ACK')
            if client.call_utility('is_scheduler_paused') is not True:
                raise C07ValidationError('The native scheduler did not acknowledge the requested pause.')
            request_ids = llm.enqueue(prompts, params, use_tqdm=False)
            evidence['request_ids'] = list(request_ids)
            if len(request_ids) != 4 or len(set(request_ids)) != 4:
                raise C07ValidationError('The source seal did not enqueue four distinct native request IDs.')
            event('FOUR_ENQUEUED', request_ids=list(request_ids))
            # This synchronous same-client utility fences the preceding ADD messages.
            if client.call_utility('is_scheduler_paused') is not True:
                raise C07ValidationError('The native scheduler resumed before the source cohort was sealed.')
            event('QUEUE_FENCE_ACK')
        except BaseException as error:
            primary_error = error
            raise
        finally:
            if attempted_pause:
                try:
                    client.call_utility('resume_scheduler')
                    event('RESUME_ACK')
                    if client.call_utility('is_scheduler_paused') is not False:
                        raise C07ValidationError('The native scheduler remains paused after resume.')
                except BaseException as error:
                    evidence['restore_error'] = dict(type=type(error).__name__, message=str(error))
                    if primary_error is None:
                        raise
        event('NATIVE_WAIT_START')
        results = llm.wait_for_completion(use_tqdm=False)
        event('NATIVE_WAIT_RETURNED', results=len(results))
        evidence['status'] = 'COMPLETE'
        evidence['scope'] = 'Untimed source-fixture construction. Not a scheduler or service-time measurement.'
        return results
    except BaseException as error:
        evidence.update(status='FAILED', error=dict(type=type(error).__name__, message=str(error)))
        raise

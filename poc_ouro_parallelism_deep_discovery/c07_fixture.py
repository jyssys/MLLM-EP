"""CPU-testable contracts for one eager, physical Ouro pass."""
from dataclasses import fields, is_dataclass, replace


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

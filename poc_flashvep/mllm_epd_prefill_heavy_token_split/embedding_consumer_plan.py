"""Delivery requirements for a fixed contiguous split, not an axis selector."""
from pathlib import Path
import re

from .common import write
from .token_split import POLICY_KEY, select_prefix


def build(request, prompt):
    policy = (getattr(request,'kv_transfer_params',None) or {}).get(POLICY_KEY)
    if policy is None:
        return None
    assert set(policy) == {'suffix_fraction', 'fixed_prefix'}
    length = len(prompt['prompt_token_ids'])
    prefix = select_prefix(length, policy['suffix_fraction'], policy['fixed_prefix'])
    assert prefix is not None
    hashes = prompt['mm_hashes']['image']
    positions = prompt['mm_placeholders']['image']
    assert len(hashes) == len(positions) and len(hashes) == len(set(hashes))
    images = []
    for identifier, position in zip(hashes, positions):
        left, right = position.offset, position.offset + position.length
        assert 0 <= left < right <= length
        assert position.is_embed is None or bool(position.is_embed.all()), (
            'Only the original contiguous Qwen2.5-VL image spans are qualified for this delivery plan')
        roles = []
        if left < prefix:
            roles.append('prefill')
        if right > prefix:
            roles.append('decode')
        assert roles
        images.append(dict(key=identifier, roles=roles, offset=left, length=position.length))
    return dict(prefix_tokens=prefix, full_prompt_tokens=length, images=images,
        scope='Required consumers of unchanged original image embeddings. No image or token is removed.')


def persist(ec_path, request_key, plan):
    for image in plan['images']:
        assert re.fullmatch('[A-Za-z0-9_-]+', image['key'])
        folder = Path(ec_path) / image['key']
        assert not folder.is_symlink()
        folder.mkdir(parents=True, exist_ok=True, mode=0o700)
        target = folder / 'cuda_ipc_consumers.json'
        assert not target.exists(), 'Never replace an in-flight image delivery plan'
        write(target, dict(image, request_key=request_key,
            prefix_tokens=plan['prefix_tokens'], full_prompt_tokens=plan['full_prompt_tokens']))

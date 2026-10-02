from pathlib import Path
from poc_speculative_moe_ep.contract import MASK, PYTHON, TARGET, TARGET_REV, write, sha

ROOT = Path(__file__).resolve().parent
SPEC = Path('/home/esjung/MLLM-EP-github/poc_flashvep/reports/agentic_moe_ep_poc_spec.md')


def lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y: break
        n += 1
    return n


def observation_frontier(history, tokenizer, current_ids):
    """Exclude historical content even if the native template rewrites it.

    Rendering a sentinel keeps the same last-user index as the real prompt;
    no routing/output labels are used. LCP handles the boundary BPE merge.
    """
    assert history[-1]['role']=='user'
    sentinel='__AGENT_EP_FROZEN_OBSERVATION_BOUNDARY_3407__'
    assert all(sentinel not in m['content'] for m in history)
    changed=[dict(m) for m in history]
    changed[-1]['content']=sentinel
    rendered=tokenizer.apply_chat_template(changed,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    prefix=rendered[:rendered.index(sentinel)]
    return lcp(tokenizer.encode(prefix,add_special_tokens=False),current_ids)

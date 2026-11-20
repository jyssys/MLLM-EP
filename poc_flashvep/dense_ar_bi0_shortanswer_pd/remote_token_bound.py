"""Bound imported tokens by the producer's measured remote-token count.

The installed unidirectional connector assumes that every consumer prompt
token is already remote. First-sample reuse adds one token that is not remote.
This adapter changes only the remote match count. It does not change model
math, cache tensors, sampling, scheduler priority or memory budgets.
"""


def bounded_count(prompt_tokens,remote_tokens,locally_computed):
    return max(0,min(prompt_tokens,remote_tokens)-locally_computed)


def install(cfg,role):
    if role!='decode' or cfg.get('protocol')!='FIRST_TOKEN_REUSE_BOUNDED':return
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_scheduler import NixlPullConnectorScheduler
    original=NixlPullConnectorScheduler.get_num_new_matched_tokens
    def matched(self,request,num_computed_tokens):
        params=request.kv_transfer_params
        if params and params.get('do_remote_prefill') and 'remote_num_tokens' in params:
            count=bounded_count(request.num_prompt_tokens,int(params['remote_num_tokens']),num_computed_tokens)
            return count,count>0
        return original(self,request,num_computed_tokens)
    NixlPullConnectorScheduler.get_num_new_matched_tokens=matched

"""Test probe readiness before assigning tensor differences to model math."""
import time
from .common import ROOT,write,save
from .qualification import configuration
from .run import execute
from .numerical_campaign import difference


def main():
    names={}
    for label,extra in (('Standard',dict(reference_single=True,gpus=[1])),('Native',{})):
        name=f'NumericalSync_{label}_ShortAnswer_N8_BI0_v2';names[label]=name
        cfg=configuration(name,'Native','qualification8.json',state_capture=False,
            numerical_probe=True,numerical_probe_sync=True,correctness_only=True,**extra)
        assert execute(save(cfg)),name
    import torch
    torch.set_num_threads(1);rows=[]
    for index in range(8):
        rid=f'q{index}';teacher=torch.load(ROOT/'raw'/names['Standard']/f'numerical_single_{rid}.pt',weights_only=True)
        for role in ('prefill','decode'):
            value=torch.load(ROOT/'raw'/names['Native']/f'numerical_{role}_{rid}.pt',weights_only=True)
            rows.append(dict(request_id=rid,role=role,hidden=difference(value['hidden_last'],teacher['hidden_last']),
                KV=difference(value['KV_samples'],teacher['KV_samples']),
                reference_descriptor=teacher['batch_descriptor'],actual_descriptor=value['batch_descriptor'],
                reference_attention=teacher['attention_metadata'],actual_attention=value['attention_metadata']))
    write(ROOT/'analysis/BI0_synchronized_numerical_probe_v2.json',dict(rows=rows,evidence='CORRECTNESS_ONLY',
        scope='Device-wide synchronization precedes reads. Public query geometry and actual attention metadata are disclosed. '
        'No output is changed. This probe cannot provide latency measurements.'))
    write(ROOT/'state/numerical_sync_complete.json',dict(time=time.time(),complete=True))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'state/numerical_sync_hold.json',dict(time=time.time(),error=repr(error)));raise

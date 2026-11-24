"""Diagnostic-only binding of an unchanged target to its actual encoder exports."""


def prepare(store, template, target_request, encoder_rendered, bridge_key, arrival_time):
    import copy
    import torch
    target=copy.deepcopy(template)
    prompt=target[1][0];encoded=encoder_rendered[1][0]
    count=len(prompt['mm_kwargs']['image'])
    assert len(encoded['mm_kwargs']['image'])>=count
    for original,actual in zip(prompt['mm_kwargs']['image'],encoded['mm_kwargs']['image']):
        for field in ['pixel_values','image_grid_thw']:
            assert torch.equal(original[field].data,actual[field].data), 'Target processor bytes changed'
    prompt['mm_hashes']={'image':list(encoded['mm_hashes']['image'][:count])}
    prompt['arrival_time']=arrival_time
    store.save(bridge_key,target_request,target,'full')
    return {'x-campaign-metadata-key':bridge_key,'x-campaign-metadata-mode':'full'}


class Session:
    def __init__(self,session,headers):
        self.session,self.headers=session,headers

    def post(self,url,**kwargs):
        kwargs['headers']=dict(kwargs.get('headers') or {},**self.headers)
        return self.session.post(url,**kwargs)

"""CPU regression: native residual collectives may overwrite input storage."""
from types import SimpleNamespace
import torch
from libra_vl_bridge import install_captured_vl_context


class FakeDecoder:
    layers=[]
    def forward(self,ids,positions,batch,input_embeds=None,**kwargs):
        # The native first-layer residual aliases input_embeds; dp_scatter
        # writes the post-attention residual into that storage.
        input_embeds.add_(1)
        return input_embeds.clone()


def test_repeated_captures_are_immutable():
    model=SimpleNamespace(model=FakeDecoder())
    context=install_captured_vl_context(model)
    captured=torch.arange(8).reshape(2,4).float()
    context['inputs_embeds']=captured.clone()
    first=model.model.forward(torch.arange(2),None,None)
    second=model.model.forward(torch.arange(2),None,None)
    assert torch.equal(context['inputs_embeds'],captured), 'Captured source was mutated'
    assert torch.equal(first,second), 'Repeat depends on earlier forwards'


if __name__=='__main__':
    test_repeated_captures_are_immutable()
    print('PASS: repeated native forwards cannot mutate captured VL source')

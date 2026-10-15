"""Explicit straight-line execution. Modules and parameter storages stay shared."""
import linecache


def make_forward(passes):
    assert passes in [1,2,4]
    lines=['def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):',
        '    hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)']
    for recurrent in range(passes):
        lines.append('    residual = None')
        for layer in range(48):
            lines.append(f'    hidden_states, residual = self.layers[{layer}](positions, hidden_states, {recurrent}, residual)')
        lines.append('    hidden_states, _ = self.norm(hidden_states, residual)')
    lines.append('    return hidden_states')
    source='\n'.join(lines)+'\n';filename=f'<poc4_explicit_tied_R{passes}>'
    linecache.cache[filename]=(len(source),None,source.splitlines(keepends=True),filename)
    namespace={'__name__':__name__};exec(compile(source,filename,'exec'),namespace)
    return namespace['forward'],source


def install(passes):
    from vllm.model_executor.models import ouro
    forward,source=make_forward(passes)
    ouro.OuroModel.forward=forward
    return source

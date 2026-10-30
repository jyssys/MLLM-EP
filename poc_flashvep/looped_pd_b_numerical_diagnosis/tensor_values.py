"""Publish small exact tensor examples from saved real-operand diagnostics."""
import csv
import hashlib
import torch
from .common import ROOT,write


def main():
    examples=[]
    for model,run,contains in (
        ('ouro','teacher_ouro_sample3_original','down_proj'),
        ('ouro','teacher_ouro_sample3_down_norm_v3','layernorm'),
        ('huginn','teacher_huginn_sample92_norm_v3','norm_1'),
        ('huginn','teacher_huginn_independent_sample85_residual_v8','adapter'),
        ('huginn','teacher_huginn_closed_loop_sample56_residual_v8','adapter')):
        source=ROOT/'raw'/run/'isolated_same_operand_differences.pt'
        entries=torch.load(source,map_location='cpu',weights_only=True)
        row=next(r for r in entries if contains in r['key'][3])
        x,a,b=row['input'],row['one'],row['many']
        changed=(a!=b).nonzero()
        if a.ndim!=1:continue
        key=row['key']
        first=[]
        for index in changed.flatten().tolist()[:12]:
            first.append(dict(output_index=index,baseline=float(a[index]),other=float(b[index]),
                              baseline_BF16_bits=hex(int(a.view(torch.int16)[index])&0xffff),
                              other_BF16_bits=hex(int(b.view(torch.int16)[index])&0xffff)))
        examples.append(dict(model=model,source=str(source.relative_to(ROOT)),key=key,other_rows=row['rows'],
             common_input_sha256=hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
             input_shape=list(x.shape),first_input_values=x.flatten()[:12].float().tolist(),
             unequal_elements=len(changed),examples=first,evidence='MEASURED_SAVED_SAME_OPERAND_OPERATOR_VALUES'))
    write(ROOT/'configs/first_divergent_tensor_values.json',examples)
    flat=[dict(model=r['model'],phase=r['key'][0],step=r['key'][1],depth=r['key'][2],operation=r['key'][3],
               other_rows=r['other_rows'],input_sha256=r['common_input_sha256'],**entry) for r in examples for entry in r['examples']]
    with (ROOT/'first_divergent_tensor_values.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    print(examples)


if __name__=='__main__':main()

"""Freeze a P-only compiler intervention for independent correctness tests."""
import hashlib
import zipfile
from .common import ROOT,SOURCE,write,digest,preserve

def build():
    preserve();source=ROOT/'references/source_capsules/v2_rooted_eager_prefill_native_decode_v1.zip'
    target=source.with_name('v2_p_compiled_prefix_experiment_v1.zip');assert not target.exists()
    with zipfile.ZipFile(source) as archive:files={n:archive.read(n) for n in archive.namelist()}
    adapter='poc_flashvep/dense_ar_longbench400_pd/adapter.py'
    text=files[adapter].decode();old='                for layer in range(begin,end):local_h,local_r=module.layers[layer](local_positions,local_h,local_r)'
    assert text.count(old)==1
    text=text.replace(old,"                if self.role=='prefill' and self.cfg.get('P_compile_prefix'):\n"
        "                    local_h,local_r=self.compiled_prefix_segment(module,local_positions,local_h,local_r,begin,end)\n"
        "                else:\n"+old.replace('                for','                    for'))
    files[adapter]=text.encode()
    worker='poc_flashvep/dense_ar_longbench400_pd/standard_worker.py';text=files[worker].decode()
    old="        if cfg.get('token_stock_model'):";assert text.count(old)==1
    text=text.replace(old,"        if role=='prefill' and cfg.get('P_compile_prefix'):\n"
        "            from poc_flashvep.dense_ar_output_aware_v2.prefix_compiled_adapter import PrefixCompiledAdapter as Adapter\n"
        "        elif cfg.get('token_stock_model'):")
    old='from .scoped_compile_options import options';assert text.count(old)==1
    text=text.replace(old,'from poc_flashvep.dense_ar_output_aware_v2.prefix_compile_options import options')
    files[worker]=text.encode()
    for name in ('__init__.py','prefix_compiled_adapter.py','prefix_compile_options.py'):
        files['poc_flashvep/dense_ar_output_aware_v2/'+name]=(SOURCE/name).read_bytes()
    target.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,data in sorted(files.items()):
            compile(data,name,'exec') if name.endswith('.py') else None
            entry=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));entry.compress_type=zipfile.ZIP_DEFLATED;archive.writestr(entry,data)
    write(target.with_suffix('.json'),dict(SHA256=digest(target),base_SHA256=digest(source),
        files={n:hashlib.sha256(v).hexdigest() for n,v in files.items()},
        scope='Experimental only. P compile switch off preserves the rooted baseline. P switch on requires fresh strict state and mixed/variable/natural gates before performance. No installed runtime modification.'))
    print('EXPERIMENTAL_P_COMPILE_CAPSULE',digest(target),flush=True)

if __name__=='__main__':build()

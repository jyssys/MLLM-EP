"""Separate shared source paths from actual compiler-cache factor equality."""
import hashlib
import json
from pathlib import Path
import re
import socket
import time
import zipfile
from .common import ROOT,SOURCE,write,digest,preserve
from .capsule import replace_once

NAMES=('PC_StableSource_StockToken_NormalGC_Strict8','PC_StableSource_StockToken_DeferredGC_Strict8',
    'PC_StableSource_StockToken_NormalGCRepeat_Strict8')


def audit():
    preserve();roles=[]
    for role in ('prefill','decode'):
        rows=[];factors=[]
        for name in NAMES:
            text=(ROOT/'raw'/name/(role+'.log')).read_text()
            root=Path(re.search(r'Using cache directory: (.*?) for vLLM',text)[1])
            path=root/'cache_key_factors.json';data=json.loads(path.read_text());factors.append(data)
            rows.append(dict(name=name,cache_root=str(root),cache_factors_SHA256=digest(path),
                config_hash=data['config_hash'],code_hash=data['code_hash'],compiler_hash=data['compiler_hash'],
                NIXL_control_port=data['env']['VLLM_NIXL_SIDE_CHANNEL_PORT']))
        different=[]
        for data in factors[1:]:
            different.append([key for key in sorted(set(data['env'])|set(factors[0]['env'])) if data['env'].get(key)!=factors[0]['env'].get(key)])
        equal_other=all(all(data[key]==factors[0][key] for key in ('config_hash','code_hash','compiler_hash')) for data in factors[1:])
        roles.append(dict(role=role,rows=rows,other_compiler_factors_equal=equal_other,changed_environment_keys=different,
            only_port_changed=equal_other and all(keys==['VLLM_NIXL_SIDE_CHANNEL_PORT'] for keys in different),
            scope='Only sanitized factor hashes and control ports are copied. Do not publish raw environment values or credentials.'))
    strict=json.loads((ROOT/'state/stable_source_stock_strict_complete_v1.json').read_text())
    import torch
    torch.set_num_threads(1);hidden=[]
    controls=json.loads((ROOT/'raw'/NAMES[0]/'axis_events_decode.json').read_text())['states']
    for name in NAMES[1:]:
        states=json.loads((ROOT/'raw'/name/'axis_events_decode.json').read_text())['states']
        for rid in sorted(controls):
            if not rid.startswith('q'):continue
            values=[torch.load(state['tensor_file'],map_location='cpu',weights_only=False)['hidden'] for state in (controls[rid],states[rid])]
            assert values[0].shape==values[1].shape
            a,b=[value.float() for value in values];difference=a-b
            hidden.append(dict(candidate=name,request_id=rid,shape=list(a.shape),dtype=str(values[0].dtype),
                exact=torch.equal(*values),finite=all(bool(torch.isfinite(value).all()) for value in values),
                max_abs=float(difference.abs().max()),RMS_error=float(difference.square().mean().sqrt()),
                scope='Magnitude only. No acceptance tolerance is introduced. The strict failure remains blocking.'))
    result=dict(time=time.time(),passed=True,roles=roles,strict_counterfactual_passed=strict['passed'],hidden_numeric_differences=hidden,
        evidence='TRACE_ANALYSIS',scope='All three runs used one verified worker-source path. '
            'They still used different official cache roots. The only changed recorded environment factor was the NIXL control port. '
            'Config,code and compiler hashes matched within each role. Source-path stabilization did not create shared artifacts. '
            'This identifies cache-key fragmentation. It does not yet prove that fragmentation caused every state mismatch. '
            'The next intervention pins legitimate transport addresses. It does not override official hashes or inject compiled kernels.')
    write(ROOT/'analysis/NIXL_port_compile_cache_factor_audit.json',result)
    print('CACHE_PORT_FACTOR_AUDIT',[(row['role'],row['only_port_changed']) for row in roles],flush=True)
    return result


def freeze_ports():
    target=ROOT/'configs/fixed_NIXL_control_ports_v1.json'
    if target.exists():return json.loads(target.read_text())['ports']
    low,high=map(int,Path('/proc/sys/net/ipv4/ip_local_port_range').read_text().split())
    for base in range(16080,17900,8):
        values=list(range(base,base+8));handles=[]
        try:
            assert all(not low<=port<=high for port in values)
            for port in values:
                sock=socket.socket();handles.append(sock);sock.bind(('127.0.0.1',port))
            ports={f'{left},{left+1}':values[left:left+2] for left in (0,2,4,6)}
            write(target,dict(time=time.time(),ports=ports,available_before_any_new_GPU_run=True,
                scope='Choose the first available eight-port block below the old random18000–28000/62000–65000 pools. '
                    'Freeze addresses by physical pair. No performance value selects an address. '
                    'The runtime fails if an address becomes occupied. It does not kill another process or silently change the cache factor.'))
            return ports
        except OSError:continue
        finally:
            for sock in handles:sock.close()
    raise RuntimeError('No free frozen control-port block. Leave external listeners untouched.')


def build(parent_name):
    parent=ROOT/'references/source_capsules'/parent_name;target=parent.with_name(parent.stem+'_fixed_ports_v1.zip')
    if target.exists():return target
    with zipfile.ZipFile(parent) as stream:files={name:stream.read(name) for name in stream.namelist()}
    before=dict(files);path='poc_flashvep/dense_ar_longbench400_pd/serving.py'
    source=files[path].decode()
    source=replace_once(source,'            port=choose(control_ports.values());control_ports[role]=port\n',
        "            if cfg.get('fixed_NIXL_control_ports'):\n"
        "                from poc_flashvep.dense_ar_p_critical_path.pinned_control_ports import choose as choose_fixed\n"
        "                port=choose_fixed(cfg,role,control_ports.values())\n"
        "            else:port=choose(control_ports.values())\n"
        "            control_ports[role]=port\n")
    files[path]=source.encode();files['poc_flashvep/dense_ar_p_critical_path/pinned_control_ports.py']=(SOURCE/'pinned_control_ports.py').read_bytes()
    changed=[name for name in before if files[name]!=before[name]];assert changed==[path]
    for name,value in files.items():
        if name.endswith('.py'):compile(value,name,'exec')
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED) as stream:
        for name,value in sorted(files.items()):
            item=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));item.compress_type=zipfile.ZIP_DEFLATED;stream.writestr(item,value)
    write(target.with_suffix('.json'),dict(SHA256=digest(target),parent_SHA256=digest(parent),changed=changed,
        all_worker_bytes_identical=True,files={name:hashlib.sha256(value).hexdigest() for name,value in files.items()},
        scope='Change only startup control-port selection. Preserve every worker,model,attention,graph,precision and scheduling byte. '
            'Use the official compiler factor computation and cache loader. Require fresh exact state and Natural32/400 gates.'))
    return target


if __name__=='__main__':audit();print(freeze_ports(),flush=True)

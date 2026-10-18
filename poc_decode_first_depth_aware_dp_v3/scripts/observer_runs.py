"""Paired observer and graph checks on frozen calibration requests."""
import subprocess
from poc_decode_first_depth_aware_dp_v3.contract import *

def main():
    manifest=ROOT/'artifacts/observer_frozen.json'
    assert not manifest.exists()
    rows=[r for r in read(ROOT/'artifacts/discovery_frozen.json')['rows'] if r['split']=='calibration'][:8]
    write(manifest,dict(rows=rows,cap=256,criterion='First eight predeclared calibration requests.',
        modes=['graph_clean','graph_trace','eager_clean'],heldout=False))
    for mode in ['graph_clean','graph_trace','eager_clean']:
        command=[REFERENCE_PYTHON,'-m','poc_decode_first_depth_aware_dp_v3.scripts.gpu_session',
            '--gpus','7','--output',f'outputs/observer_{mode}_v1','--timeout','1800','--',PYTHON,
            '-m','poc_decode_first_depth_aware_dp_v3.scripts.probe','--manifest',str(manifest),'--cap','256']
        if mode!='eager_clean':command.append('--graphs')
        if mode=='graph_trace':command.append('--trace')
        subprocess.run(command,cwd=PROJECT,check=True)

if __name__=='__main__':main()

"""Keep the failed startup namespace and create compiler-path recovery cells."""
import json,time
from .common import *
from .prepare import save


def main():
    initialize();old=json.loads((ROOT/'state/native_initial_plan.json').read_text());names=[]
    for name in old['jobs']:
        cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
        cfg.update(name=name+'_pathv2',environment_revision='EXPLICIT_NATIVE_PYTHON_BIN_PATH')
        names.append(save(cfg))
    write(ROOT/'state/native_path_recovery_plan.json',dict(time=time.time(),jobs=names,
        reason='The first P engine failed before requests because FlashInfer could not find ninja. Only the Native Python bin path changes. The failed namespace and original plan remain unchanged.'))
    print(json.dumps(names))


if __name__=='__main__':main()

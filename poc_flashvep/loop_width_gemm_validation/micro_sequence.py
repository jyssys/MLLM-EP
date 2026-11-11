import argparse,subprocess
from .common import *


def main():
    initialize();p=argparse.ArgumentParser();p.add_argument('names',nargs='+');args=p.parse_args();results=[]
    for name in args.names:
        value=subprocess.run(['/home/esjung/.venvs/looped-pd-abc/bin/python','-u','-m',
            'poc_flashvep.loop_width_gemm_validation.run_micro',name],cwd=REPO)
        results.append(dict(name=name,returncode=value.returncode,time=time.time()))
        write(ROOT/'state/micro_progress.json',dict(results=results,closed=False))
        if value.returncode:break
    write(ROOT/'state/micro_progress.json',dict(results=results,closed=True))


if __name__=='__main__':main()

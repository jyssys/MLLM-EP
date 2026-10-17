"""Check only new report prose. The frozen source specification is excluded."""
import argparse
import subprocess
from poc_recurrent_asymmetry.contract import *


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    assert not args.output.exists()
    files=sorted(p for p in (ROOT/'reports').glob('*.md') if p.name!='spec.md')+[ROOT/'README.md']
    result=subprocess.run([PYTHON,'/home/esjung/.codex/skills/asd-ste100/scripts/ste-lint.py','--json',
        *map(str,files)],capture_output=True,text=True)
    record=json.loads(result.stdout)
    write(args.output,record)
    print('REPORT_CHECK',record['hard_count'],'hard findings,',record['count'],'total findings')
    assert result.returncode==0 and record['hard_count']==0


if __name__=='__main__':main()

"""Check new report structure. Preserve the source spec and technical depth labels."""
import subprocess
from poc_decode_first_depth_aware_dp.contract import *


def main():
    output=ROOT/'artifacts/writing_check.json'
    assert not output.exists()
    reports=[ROOT/'README.md',*sorted(p for p in (ROOT/'reports').glob('*.md') if p.name!='spec.md')]
    assert all(p.is_file() for p in reports)
    command=[REFERENCE_PYTHON,'/home/esjung/.codex/skills/asd-ste100/scripts/ste-lint.py',
        '--disable','synonym-rotation',*map(str,reports)]
    result=subprocess.run(command,capture_output=True,text=True)
    write(output,dict(files=[str(p.relative_to(ROOT)) for p in reports],return_code=result.returncode,
        output=result.stdout+result.stderr,
        lexical_exception='Fixed R4 denotes constant recurrent depth. Correct denotes answer validity. They are not synonymous actions.',
        manual_checks=['One main idea per sentence','Evidence categories remain separate',
            'Numbers, denominators, limitations, and CI bounds remain intact','No quality or speedup promotion from uncertain evidence'],
        excluded='Unchanged authoritative source specification and raw evidence.',
        scope='Structural heuristic check and prose review. Not certified dictionary compliance.'))
    print(result.stdout+result.stderr)
    assert result.returncode==0


if __name__=='__main__':main()

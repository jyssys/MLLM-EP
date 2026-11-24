"""Correct repeated option labels before performance. Keep original frozen files."""
from concurrent.futures import ProcessPoolExecutor
import json
import time
from .common import ROOT,write,digest,preserve
from .dataset_build import profile,keys


def original(row):
    return {k:row[k] for k in ('request_id','source','source_revision','original_id','metadata',
        'parquet','row_index','media_columns','stratum','order_key')}


def main():
    source=ROOT/'data/main_manifest.json';cal_source=ROOT/'data/calibration_manifest.json'
    frozen=json.loads(source.read_text());cal=json.loads(cal_source.read_text())
    rows=frozen['requests']+cal['requests']
    with ProcessPoolExecutor(max_workers=4) as pool:corrected=list(pool.map(profile,map(original,rows)))
    assert all(r['eligible'] for r in corrected),'Do not silently replace a previously selected request'
    assert [r['request_id'] for r in rows]==[r['request_id'] for r in corrected]
    final=dict(frozen,requests=corrected[:len(frozen['requests'])],prompt_template_version=2)
    cal_final=dict(cal,requests=corrected[len(frozen['requests']):],prompt_template_version=2)
    main_file=ROOT/'data/main_manifest_template_v2.json';cal_file=ROOT/'data/calibration_manifest_template_v2.json'
    write(main_file,final);write(cal_file,cal_final)
    changes=[dict(request_id=a['request_id'],old_tokens=a['total_input_tokens'],new_tokens=b['total_input_tokens'],
        old_prompt_SHA256=a['prompt_SHA256'],new_prompt_SHA256=b['prompt_SHA256'])
        for a,b in zip(rows,corrected) if a['prompt_SHA256']!=b['prompt_SHA256']]
    write(ROOT/'data/prompt_template_correction_manifest.json',dict(time=time.time(),changes=changes,
        reason='Retain each original multiple-choice option label exactly once. The first template duplicated B/C/D labels.',
        source_manifest_SHA256=digest(source),source_calibration_SHA256=digest(cal_source),
        final_main_SHA256=digest(main_file),final_calibration_SHA256=digest(cal_file),
        request_selection_changed=False,media_changed=False,reference_answers_changed=False,
        scope='Protocol correction before any Main400 serving performance. Original frozen files and calibration traces remain immutable.'))
    write(ROOT/'state/dataset_final_input_freeze.json',dict(time=time.time(),main_manifest=str(main_file),
        calibration_manifest=str(cal_file),main_SHA256=digest(main_file),calibration_SHA256=digest(cal_file),
        count=400,discovery=200,heldout=200,calibration=32,prompt_template_version=2,
        natural_output_status='NOT_MEASURED',status='INPUT_FROZEN'))
    preserve()


if __name__=='__main__':main()

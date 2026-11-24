"""Keep historical workload and independent-start families distinct."""
import csv
import json
from .common import ROOT,V2,PRIOR


def historical_runs():
    rows=[]
    for family,file in (
        ('Historical_Main400_Natural','c3_main_serving_v2_per_start.csv'),
        ('Historical_Heldout192_Natural','c3_heldout_serving_v2_per_start.csv'),
        ('Historical_Heldout192_EqualReplay','c3_heldout_replay_v2_per_start.csv')):
        for item in csv.DictReader((V2/'figures'/file).open()):
            if float(item['QPS'])!=2:continue
            rows.append(dict(root=V2,family=family,mode='Natural' if 'Natural' in family else 'EqualReplay',
                label=item['label'],start=int(item['start']),name=item['name'],
                label_scope='Frozen v2 best-of-family policy. NativeCommon0/1/2 are separate policies.'))
    for family,filename,scope in (
        ('Prior_Credit','c3_credit_retirement_confirmation_r1.json','Optimized means safe source retirement.'),
        ('Prior_Geometry','p_geometry_confirmation_v1.json','Optimized means Last8 or Hybrid iso8 geometry,not credit retirement.'),
        ('Prior_Warm','coordinator_hygiene_c3_confirmation_warm_regime_v10.json','Warm compiler regime. Optimized means geometry. GC control is a separate one-factor contrast.')):
        plan=json.loads((PRIOR/'jobs'/filename).read_text())
        for block in plan['blocks']:
            for label,name in block['arms'].items():
                folder=PRIOR/'raw'/name
                if not (folder/'dispatcher_result.json').exists():continue
                assert json.loads((folder/'dispatcher_result.json').read_text())['complete']
                rows.append(dict(root=PRIOR,family=family+'_'+block['mode'],mode=block['mode'],label=label,
                    start=block['start'],name=name,label_scope=scope))
    assert len({(row['family'],row['name']) for row in rows})==len(rows)
    return rows


def current_runs():
    """Register only finished timing jobs, never qualification or priming."""
    result=[]
    plans=(('Current_Main400_Natural','main400_native_relative_policy_confirmation_v2.json','Natural'),
        ('Current_Heldout192_EqualReplay','heldout192_native_relative_confirmation_v1.json','EqualReplay'))
    for family,filename,mode in plans:
        path=ROOT/'jobs'/filename
        if not path.exists():continue
        for block in json.loads(path.read_text())['blocks']:
            for label,name in block['arms'].items():
                folder=ROOT/'raw'/name
                receipt=folder/'dispatcher_result.json'
                if not receipt.exists():continue
                if not json.loads(receipt.read_text()).get('complete'):continue
                result.append(dict(root=ROOT,family=family,mode=mode,label=label,start=block['start'],name=name,
                    label_scope='Fresh warm-cache C3 policy control. Keep each workload family separate. '
                        'A raw-complete run is not automatically quality-qualified or decode-safe.'))
    return result


def analysis_runs():
    rows=historical_runs()+current_runs()
    assert len({(row['family'],row['name']) for row in rows})==len(rows)
    return rows

"""Run CPU result reconstruction after the declared GPU stages become terminal."""
import json
import subprocess
import time
from .common import ROOT, REPO, PYTHON, write, preserve


def call(module):
    subprocess.run([str(PYTHON), '-u', '-m',
        'poc_flashvep.dense_ar_longbench400_pd.' + module], cwd=REPO, check=True)


def main():
    preserve()
    stages = {
        'main': ([f'final_performance_lane{i}_complete_v1.json' for i in range(4)],
            ['final_confirmation_analysis', 'final_report_tables',
             'final_service_decomposition', 'final_serving_figures'], None),
        'bridge': (['controlled_continuous_bridge_complete_v1.json'],
            ['controlled_bridge_analysis', 'controlled_bridge_report', 'controlled_bridge_figures'],
            'controlled_continuous_I3_bridge_results_v1.json'),
        'READ': (['hybrid_READ_repair_complete_v1.json',
                  'hybrid_READ_repair_layer_control_complete_v1.json'],
            ['hybrid_read_repair_analysis', 'repair_report_figures'],
            'hybrid_READ_repair_results_v1.json'),
        'registration': (['request_registration_deferral_complete_v1.json'],
            ['registration_deferral_analysis'], 'request_registration_deferral_results_v1.json')}
    path = ROOT / 'state/CPU_terminal_postprocessing_progress_v1.json'
    results = json.loads(path.read_text())['results'] if path.exists() else {}
    while len(results) < len(stages):
        for label, (markers, modules, analysis) in stages.items():
            if label in results or not all((ROOT / 'state' / m).exists() for m in markers):
                continue
            done = []
            try:
                for index, module in enumerate(modules):
                    call(module)
                    done.append(module)
                    if index == 0 and analysis:
                        value = json.loads((ROOT / 'analysis' / analysis).read_text())
                        if not value['complete']:
                            raise RuntimeError('Terminal stage has invalid or missing qualified results. '
                                               'Preserve them for root review. Do not draw complete figures.')
                results[label] = dict(status='CPU_RECONSTRUCTED', modules=done)
            except Exception as error:
                results[label] = dict(status='ROOT_REVIEW_REQUIRED', modules=done, error=repr(error))
            write(path, dict(time=time.time(), results=results,
                scope='CPU reconstruction only. No GPU start,retry,parameter change,scientific GO,closeout or push.'))
            print('TERMINAL_CPU_STAGE', label, results[label], flush=True)
        if len(results) < len(stages):
            time.sleep(20)
    preserve()
    write(ROOT / 'state/CPU_terminal_postprocessing_complete_v1.json',
          dict(time=time.time(), complete=True, results=results,
               scope='Root must review errors,data,all safety gates and figures. Publication and GPU closeout remain manual.'))


if __name__ == '__main__':
    main()

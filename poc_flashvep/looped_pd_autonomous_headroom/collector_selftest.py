"""GPU collector validation only. This is not a model or serving benchmark."""
from collections import Counter
from .common import *


def main():
    cfg = json.loads(Path(os.environ['ARC_CONFIG']).read_text())
    folder = ROOT/'raw'/cfg['name']
    import torch
    torch.cuda.set_device(0)
    x = torch.ones((256, 256), device='cuda')
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            x.add_(1.)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        x.add_(1.)
    torch.cuda.synchronize()
    results = []
    for use_cpu in (False, True):
        for toggle in (False, True):
            for use_graph in (False, True):
                label = f'cpu{int(use_cpu)}_toggle{int(toggle)}_graph{int(use_graph)}'
                activities = [torch.profiler.ProfilerActivity.CUDA]
                if use_cpu:
                    activities.append(torch.profiler.ProfilerActivity.CPU)
                with torch.profiler.profile(activities=activities) as profiler:
                    if toggle:
                        profiler.toggle_collection_dynamic(False, [torch.profiler.ProfilerActivity.CUDA])
                        x.add_(1.)
                        torch.cuda.synchronize()
                        profiler.toggle_collection_dynamic(True, [torch.profiler.ProfilerActivity.CUDA])
                    for _ in range(100):
                        graph.replay() if use_graph else x.add_(1.)
                    torch.cuda.synchronize()
                    if toggle:
                        profiler.toggle_collection_dynamic(False, [torch.profiler.ProfilerActivity.CUDA])
                target = folder/(label+'.json')
                profiler.export_chrome_trace(str(target))
                trace = json.loads(target.read_text())
                counts = Counter(e.get('cat', 'metadata') for e in trace['traceEvents'])
                row = dict(case=label, activities=[a.name for a in activities], toggle=toggle,
                    graph=use_graph, category_counts=dict(counts),
                    measured_kernel_events=counts.get('kernel', 0),
                    scope='Instrumentation self-test with elementwise GPU work. No model, accuracy, capacity, or serving result.')
                results.append(row)
                write(folder/'collector_selftest_results.json', results)
    write(folder/'collector_environment.json', dict(torch_version=torch.__version__,
          cuda_version=torch.version.cuda, device=str(torch.cuda.get_device_properties(0)),
          scope='The model field in the dispatch template is not a loaded model.'))
    write(folder/'status.json', dict(status='COMPLETE'))


if __name__ == '__main__':
    main()

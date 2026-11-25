"""List transfer files and create relocated manifests without starting servers."""
import argparse
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys


RELATIVE_ROOT = Path('poc_flashvep/reports/mllm_epd_prefill_heavy_token_split_20261124')
ORIGINAL_REPO = Path('/home/esjung/MLLM-EP-github')
MANIFESTS = ('main_manifest_template_v2.json', 'main100_manifest_template_v2.json',
             'calibration_manifest_template_v2.json')
SCIENCE_COMMIT = '38e67f91f0099dc546d470c83ff03af638f7af16'


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(8 * 1024**2):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def relative_asset(value):
    path = Path(value)
    relative = path.relative_to(ORIGINAL_REPO)
    inner = relative.relative_to(RELATIVE_ROOT)
    if '..' in inner.parts or not (inner.parts[0] == 'media' or
                                  inner.parts[:2] == ('data', 'request_tensors')):
        raise ValueError('Unexpected replay asset: ' + value)
    return relative


def inventory(repo):
    repo = Path(repo).resolve()
    files = {}
    manifests = []
    request_ids = set()
    for name in MANIFESTS:
        path = repo / RELATIVE_ROOT / 'data' / name
        value = read(path)
        manifests.append(dict(path=str(path.relative_to(repo)), SHA256=digest(path),
                              requests=len(value['requests'])))
        for row in value['requests']:
            request_ids.add(row['request_id'])
            candidates = [(item['path'], item['SHA256'], 'image') for item in row['images']]
            candidates.append((row['token_metadata_file'], row['token_metadata_SHA256'], 'token_metadata'))
            for original, expected, kind in candidates:
                relative = relative_asset(original)
                source = repo / relative
                if str(relative) in files:
                    if files[str(relative)]['SHA256'] != expected:
                        raise ValueError('Conflicting asset hash: ' + str(relative))
                    continue
                if digest(source) != expected:
                    raise ValueError('Asset checksum failed: ' + str(source))
                files[str(relative)] = dict(path=str(relative), bytes=source.stat().st_size,
                                            SHA256=expected, kind=kind)
    return dict(schema=1, science_commit=SCIENCE_COMMIT, original_repo=str(ORIGINAL_REPO),
                manifests=manifests, unique_requests=len(request_ids), files=sorted(files.values(), key=lambda r: r['path']),
                transfer_bytes=sum(row['bytes'] for row in files.values()),
                scope='Only original Main400, Main100 and Calibration32 replay images and token metadata. '
                      'No model weights, raw tensor captures, process records or runtime environment.')


def verify(repo, catalog):
    repo = Path(repo).resolve()
    for item in catalog['files'] + catalog['manifests']:
        relative = Path(item['path'])
        if relative.is_absolute() or '..' in relative.parts or not relative.is_relative_to(RELATIVE_ROOT):
            raise ValueError('Invalid catalog path')
        target = repo / relative
        if not target.resolve().is_relative_to(repo):
            raise ValueError('Asset escapes the repository')
        if digest(target) != item['SHA256']:
            raise ValueError('Checksum failed: ' + str(target))
    return dict(status='ASSET_BYTES_VERIFIED', files=len(catalog['files']),
                manifests=len(catalog['manifests']), GPU_correctness='NOT_RUN', runtime_portability='NOT_VALIDATED')


def prepare(repo, catalog, output):
    """Write new paths only. Do not copy historical PID, GPU or budget state."""
    repo, output = Path(repo).resolve(), Path(output).resolve()
    if output == repo or output.is_relative_to(repo / RELATIVE_ROOT):
        raise ValueError('Use a new session directory outside the frozen campaign')
    checked = verify(repo, catalog)
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for item in catalog['manifests']:
        source = repo / item['path']
        original = read(source)
        value = copy.deepcopy(original)
        for row in value['requests']:
            for image in row['images']:
                image['path'] = str(repo / relative_asset(image['path']))
            row['token_metadata_file'] = str(repo / relative_asset(row['token_metadata_file']))
        target = output / source.name
        write_new(target, value)
        records.append(dict(original_path=item['path'], original_SHA256=item['SHA256'],
                            relocated_path=str(target), relocated_SHA256=digest(target)))
    result = dict(status='MANIFEST_PATHS_PREPARED_ONLY', verification=checked, manifests=records,
                  changed_fields=['requests[].images[].path', 'requests[].token_metadata_file'],
                  source_provenance_paths='Original parquet/PDF paths remain provenance, not transferred replay inputs.',
                  copied_process_ownership=False, copied_budget=False,
                  next_gate='Create fresh host resource ownership and runtime configuration. Then run N1 and N8 correctness.')
    write_new(output / 'relocation_receipt.json', result)
    return result


def environment():
    packages = {}
    for name in ('vllm', 'torch', 'torchvision', 'transformers', 'huggingface_hub',
                 'nixl', 'nixl-cu12', 'numpy', 'Pillow', 'aiohttp', 'uvicorn',
                 'safetensors', 'nvidia-cuda-cupti-cu12'):
        try:
            dist = importlib.metadata.distribution(name)
            packages[name] = dict(version=dist.version, root=str(dist.locate_file('')))
            if name == 'vllm':
                packages[name]['direct_url'] = json.loads(dist.read_text('direct_url.json') or 'null')
        except importlib.metadata.PackageNotFoundError:
            packages[name] = dict(status='NOT_INSTALLED_IN_THIS_INTERPRETER')
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=index,name,uuid,driver_version,memory.total',
                          '--format=csv'], capture_output=True, text=True, check=False)
    return dict(Python=sys.version, executable=sys.executable, prefix=sys.prefix, base_prefix=sys.base_prefix,
                sys_path=sys.path, platform=platform.platform(), libc=platform.libc_ver(), packages=packages,
                GPU_inventory=gpu.stdout, GPU_inventory_returncode=gpu.returncode,
                scope='Observed source-host environment. This record does not install packages or validate a new host.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    export = sub.add_parser('inventory')
    export.add_argument('--repo', type=Path, required=True)
    export.add_argument('--output', type=Path, required=True)
    env = sub.add_parser('environment')
    env.add_argument('--output', type=Path, required=True)
    for action in ('verify', 'prepare'):
        command = sub.add_parser(action)
        command.add_argument('--repo', type=Path, required=True)
        command.add_argument('--catalog', type=Path, required=True)
        if action == 'prepare':
            command.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'inventory':
        result = inventory(args.repo)
        args.output.mkdir(parents=True, exist_ok=False)
        write_new(args.output / 'replay_assets.json', result)
        with (args.output / 'replay_assets.rsync.txt').open('x') as stream:
            stream.writelines(row['path'] + '\n' for row in result['files'])
        print(json.dumps({key: result[key] for key in ('unique_requests', 'transfer_bytes')}))
    elif args.action == 'environment':
        write_new(args.output, environment())
        print('SOURCE_ENVIRONMENT_RECORDED')
    elif args.action == 'verify':
        print(json.dumps(verify(args.repo, read(args.catalog))))
    else:
        print(json.dumps(prepare(args.repo, read(args.catalog), args.output), ensure_ascii=False))


if __name__ == '__main__':
    main()

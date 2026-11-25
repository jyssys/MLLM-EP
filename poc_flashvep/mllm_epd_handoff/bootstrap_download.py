"""Download pinned public inputs and reconstruct frozen images without the old host."""
import argparse
import copy
import json
from pathlib import Path, PurePosixPath
import shutil
import time

from .assets import RELATIVE_ROOT, MANIFESTS, digest, read, relative_asset, verify, prepare, write_new

MODEL_ID = 'Qwen/Qwen2.5-VL-7B-Instruct'
MODEL_REVISION = 'cc594898137f460bfe9f0759e9844b3ce807cfb5'


def safe_name(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('Invalid upstream file name')
    return path.as_posix()


def parquet_name(row):
    marker = '/snapshots/' + row['source_revision'] + '/'
    if row['parquet'].count(marker) != 1:
        raise ValueError('The original Parquet locator has no pinned revision')
    return safe_name(row['parquet'].split(marker, 1)[1])


def request_files(row):
    names = [parquet_name(row)]
    if row.get('document'):
        names.append(safe_name('documents/' + row['metadata']['doc_id']))
    return {('dataset', row['source'], row['source_revision'], name) for name in names}


def selected(repo, scope):
    names = MANIFESTS if scope == 'main400' else MANIFESTS[1:]
    manifests, rows = [], {}
    for name in names:
        path = repo / RELATIVE_ROOT / 'data' / name
        value = read(path)
        manifests.append(dict(path=str(path.relative_to(repo)), SHA256=digest(path), requests=len(value['requests'])))
        for row in value['requests']:
            previous = rows.setdefault(row['request_id'], row)
            if previous['prompt_SHA256'] != row['prompt_SHA256']:
                raise ValueError('Conflicting frozen request')
    return manifests, rows


def make_plan(repo):
    manifests, rows = selected(repo, 'main400')
    files = {}
    for row in rows.values():
        receipt = read(repo / RELATIVE_ROOT / 'state' / ('download_' + row['source'].replace('/', '__') + '.json'))
        if receipt['revision'] != row['source_revision']:
            raise ValueError('Dataset revision changed')
        known = {item['name']: item for item in receipt['files']}
        for key in request_files(row):
            name = key[-1]
            item = known[name]
            files[key] = dict(repo_type=key[0], repo_id=key[1], revision=key[2], filename=name,
                              bytes=item['bytes'], SHA256=item['SHA256'])
    model = read(repo / RELATIVE_ROOT / 'state/model_download_complete.json')
    if model['revision'] != MODEL_REVISION:
        raise ValueError('Model revision changed')
    for item in model['files']:
        name = safe_name(str(Path(item['path']).relative_to(model['directory'])))
        key = ('model', MODEL_ID, MODEL_REVISION, name)
        files[key] = dict(repo_type=key[0], repo_id=key[1], revision=key[2], filename=name,
                          bytes=item['bytes'], SHA256=item['SHA256'])
    return dict(schema=1, manifests=manifests, files=[files[key] for key in sorted(files)],
                renderer=dict(PyMuPDF='2.1', MuPDF='1.28.6', Pillow='12.3.0', dpi=72, RGB=True, alpha=False, all_pages=True),
                scope='Frozen Main100/Main400 and Calibration32 only. No sample selection, Natural generation or GPU execution. '
                      'Token metadata comes from Git. Images and model come from pinned official source files.')


def fetch_one(item, cache, offline=False, downloader=None):
    if downloader is None:
        from huggingface_hub import hf_hub_download
        downloader = hf_hub_download
    name = safe_name(item['filename'])
    path = Path(downloader(repo_id=item['repo_id'], filename=name, repo_type=item['repo_type'],
                          revision=item['revision'], cache_dir=str(cache), local_files_only=offline))
    if path.stat().st_size != item['bytes'] or digest(path) != item['SHA256']:
        raise ValueError('Pinned source checksum failed: ' + item['repo_id'] + '/' + name)
    return path


def install_media(repo, row, material):
    expected, actual = row['images'], material['images']
    if len(expected) != len(actual) or material['question'] != row['question']:
        raise ValueError('Image order/count or question changed: ' + row['request_id'])
    if material['reference_answers'] != row['reference_answers']:
        raise ValueError('Reference answer changed')
    if row.get('document') and material['document']['document_SHA256'] != row['document']['document_SHA256']:
        raise ValueError('PDF source changed')
    for old, new in zip(expected, actual):
        for key in ('SHA256', 'pixel_SHA256', 'original_resolution'):
            if old[key] != new[key]:
                raise ValueError('Reconstructed image mismatch: ' + row['request_id'] + ' / ' + key)
        target = repo / relative_asset(old['path'])
        source = Path(new['path'])
        if digest(source) != old['SHA256']:
            raise ValueError('Generated image checksum failed')
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            with source.open('rb') as src, target.open('xb') as dst:
                shutil.copyfileobj(src, dst)
        if digest(target) != old['SHA256']:
            raise ValueError('Existing image differs. Do not overwrite it automatically.')


def reconstruct(repo, rows, fetched, workspace):
    from ..mllm_epd_prefill_heavy_token_split import dataset_build as builder
    previous_root, previous_snapshot = builder.ROOT, builder.snapshot
    directories = {}
    for key, path in fetched.items():
        if key[0] == 'dataset':
            directories[key[1]] = path.parents[len(PurePosixPath(key[3]).parts) - 1]
    try:
        builder.ROOT = workspace / 'generated'
        builder.snapshot = lambda source: dict(directory=str(directories[source]))
        for row in rows.values():
            local = copy.deepcopy(row)
            key = ('dataset', row['source'], row['source_revision'], parquet_name(row))
            local['parquet'] = str(fetched[key])
            material = builder.load_media(local)
            install_media(repo, row, material)
            print('IMAGE_REQUEST_HASH_PASS', row['request_id'], flush=True)
    finally:
        builder.ROOT, builder.snapshot = previous_root, previous_snapshot
        builder.media_table.cache_clear()


def bootstrap(repo, workspace, scope='main100', offline=False):
    repo, workspace = Path(repo).resolve(), Path(workspace).resolve()
    if workspace.is_relative_to(repo / RELATIVE_ROOT):
        raise ValueError('Use a workspace outside the frozen campaign')
    plan = read(Path(__file__).parent / 'download_plan.json')
    for item in plan['manifests']:
        if digest(repo / item['path']) != item['SHA256']:
            raise ValueError('Frozen manifest changed')
    manifests, rows = selected(repo, scope)
    needed = set().union(*(request_files(row) for row in rows.values()))
    workspace.mkdir(parents=True, exist_ok=True)
    fetched = {}
    for item in plan['files']:
        key = (item['repo_type'], item['repo_id'], item['revision'], item['filename'])
        if item['repo_type'] == 'model' or key in needed:
            print('PINNED_SOURCE_DOWNLOAD', item['repo_id'], item['filename'], flush=True)
            fetched[key] = fetch_one(item, workspace / 'hf_cache', offline)
    if not needed <= set(fetched):
        raise ValueError('Download plan does not cover selected requests')
    reconstruct(repo, rows, fetched, workspace)
    all_assets = read(Path(__file__).parent / 'transfer_20261125/replay_assets.json')
    names = {str(relative_asset(image['path'])) for row in rows.values() for image in row['images']}
    names.update(str(relative_asset(row['token_metadata_file'])) for row in rows.values())
    catalog = dict(files=[item for item in all_assets['files'] if item['path'] in names], manifests=manifests)
    verified = verify(repo, catalog)
    timestamp = str(time.time_ns())
    relocated = prepare(repo, catalog, workspace / ('manifests_' + timestamp))
    model_paths = [path for key, path in fetched.items() if key[0] == 'model']
    result = dict(status='PINNED_ASSETS_READY', scope=scope, requests=len(rows),
                  model_directory=str(model_paths[0].parent), model_revision=MODEL_REVISION,
                  verification=verified, relocated_manifests=relocated['manifests'],
                  old_server_contacted=False, GPU_correctness='NOT_RUN')
    receipt = workspace / ('download_ready_' + timestamp + '.json')
    write_new(receipt, result)
    print('DOWNLOAD_AND_RECONSTRUCTION_PASS', receipt, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--work-dir', type=Path)
    parser.add_argument('--scope', choices=('main100', 'main400'), default='main100')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--make-plan', type=Path)
    args = parser.parse_args()
    if args.make_plan:
        write_new(args.make_plan, make_plan(args.repo.resolve()))
        print('PINNED_DOWNLOAD_PLAN_CREATED')
    elif args.work_dir:
        bootstrap(args.repo, args.work_dir, args.scope, args.offline)
    else:
        parser.error('--work-dir or --make-plan is required')


if __name__ == '__main__':
    main()

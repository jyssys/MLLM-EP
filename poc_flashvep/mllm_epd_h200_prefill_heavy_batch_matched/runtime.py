#!/usr/bin/env python3
"""H200 setup verification, immutable source overlay and owned EPD processes."""
import csv
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

HEAD = '449c3e1873c65206435a6bb9411a889a62e55be7'
MODEL_REVISION = 'cc594898137f460bfe9f0759e9844b3ce807cfb5'
VLLM_WHEEL = '6ce4ca30616f0a35810391015622b197a7b8b267ed27f8716f0789db79ff578b'
PACKAGE = 'mllm_epd_prefill_heavy_token_split'
REPORT = Path('poc_flashvep/reports') / (PACKAGE + '_20261124')
CALIBRATION = 'calibration_manifest_template_v2.json'
MAIN100 = 'main100_manifest_template_v2.json'
PAUSE = Path('/home1/irteam/dsd_work/BURNER_PAUSE')
ROLES = [('encoder', 4, 18840, 19840), ('prefill', 5, 18841, 19841), ('decode', 6, 18842, 19842)]
PORTS = [port for _, _, http, side in ROLES for port in (http, side)] + [18847, 18848, 18849]
PINS = {'vllm': '0.26.0+cu129', 'torch': '2.11.0+cu129', 'torchvision': '0.26.0+cu129',
        'transformers': '5.16.1', 'huggingface_hub': '1.30.0', 'nixl': '1.4.1',
        'nixl-cu12': '1.4.1', 'numpy': '2.4.6', 'Pillow': '12.3.0',
        'aiohttp': '3.14.3', 'uvicorn': '0.52.4', 'safetensors': '0.8.0',
        'torchcodec': '0.16.0+cu129', 'nixl-cu13': '1.4.1'}


def require(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def git_check(repo):
    output = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True)
    require(output.strip() == HEAD, 'Checkout is not the exact pinned commit')
    subprocess.run(['git', '-C', str(repo), 'diff', '--exit-code', '--quiet', 'HEAD', '--'], check=True)


def verify_inputs(repo, receipt_path):
    receipt = read(receipt_path)
    require(receipt['status'] == 'PINNED_ASSETS_READY' and receipt['scope'] == 'main100'
            and receipt['requests'] == 132 and not receipt['old_server_contacted']
            and receipt['model_revision'] == MODEL_REVISION
            and receipt['verification']['status'] == 'ASSET_BYTES_VERIFIED',
            'Invalid Main100+Calibration32 bootstrap receipt')
    plan = read(repo / 'poc_flashvep/mllm_epd_handoff/download_plan.json')
    catalog = read(repo / 'poc_flashvep/mllm_epd_handoff/transfer_20261125/replay_assets.json')
    provenance_root = Path(catalog['original_repo'])
    model = Path(receipt['model_directory']).absolute()
    require(model.name == MODEL_REVISION and model.is_relative_to(receipt_path.parent / 'hf_cache'),
            'Unexpected model snapshot directory')
    expected_manifests = {Path(entry['path']).name: entry for entry in plan['manifests']}
    relocated = {Path(entry['original_path']).name: entry for entry in receipt['relocated_manifests']}
    require(set(relocated) == {CALIBRATION, MAIN100}, 'Unexpected relocated manifest scope')
    for name, count in [(MAIN100, 100), (CALIBRATION, 32)]:
        expected, entry = expected_manifests[name], relocated[name]
        original = repo / expected['path']
        path = Path(entry['relocated_path']).resolve(strict=True)
        require(path.is_relative_to(receipt_path.parent) and entry['original_path'] == expected['path']
                and sha(original) == entry['original_SHA256'] == expected['SHA256']
                and sha(path) == entry['relocated_SHA256'], f'Manifest integrity failure: {name}')
        canonical, changed = read(original), read(path)
        require(len(canonical['requests']) == len(changed['requests']) == count, f'Invalid row count: {name}')
        for before, after in zip(canonical['requests'], changed['requests'], strict=True):
            for original_image, image in zip(before['images'], after['images'], strict=True):
                asset = Path(image['path'])
                relative = Path(original_image['path']).relative_to(provenance_root)
                require(asset == repo / relative and asset.resolve().is_relative_to(repo / REPORT / 'media')
                        and sha(asset) == original_image['SHA256'], f'Image integrity failure: {asset}')
                image['path'] = original_image['path']
            token = Path(after['token_metadata_file'])
            relative = Path(before['token_metadata_file']).relative_to(provenance_root)
            require(token == repo / relative and token.resolve().is_relative_to(repo / REPORT / 'data/request_tensors')
                    and sha(token) == before['token_metadata_SHA256'], f'Token metadata integrity failure: {token}')
            after['token_metadata_file'] = before['token_metadata_file']
        require(canonical == changed, f'Relocated manifest changed frozen content: {name}')
    model_entries = [entry for entry in plan['files'] if entry['repo_type'] == 'model']
    require(bool(model_entries), 'No pinned model files')
    for entry in model_entries:
        artifact = model / entry['filename']
        require(entry['revision'] == MODEL_REVISION and artifact.stat().st_size == entry['bytes']
                and sha(artifact) == entry['SHA256'], f'Model integrity failure: {artifact}')
    return model, {name: Path(entry['relocated_path']) for name, entry in relocated.items()}


def runtime_inventory():
    require(sys.version_info[:3] == (3, 14, 6), 'Use the Python 3.14.6 runtime')
    installed = {name: importlib.metadata.version(name) for name in PINS}
    require(installed == PINS, f'Pinned runtime package mismatch: {installed}')
    info = json.loads(importlib.metadata.distribution('vllm').read_text('direct_url.json') or '{}')
    archive = info.get('archive_info', {})
    wheel_hash = archive.get('hashes', {}).get('sha256') or archive.get('hash', '').removeprefix('sha256=')
    require(wheel_hash == VLLM_WHEEL, 'Installed vLLM wheel needs the official cu129 SHA256 receipt')
    bin_dir = Path(sys.executable).parent
    runtime_tools = {name: str(bin_dir / name) for name in
                     ('ninja', 'x86_64-conda-linux-gnu-cc', 'x86_64-conda-linux-gnu-c++')}
    for name, path in runtime_tools.items():
        require(os.access(path, os.X_OK), f'{name} unavailable in the isolated runtime')
        subprocess.run([path, '--version'], check=True, capture_output=True)
    return {'python': sys.version, 'executable': sys.executable, 'packages': installed,
            'vllm_wheel_sha256': wheel_hash, 'runtime_tools': runtime_tools}


def nvidia(query):
    output = subprocess.check_output(['nvidia-smi', query, '--format=csv,noheader,nounits'], text=True)
    return [[part.strip() for part in row] for row in csv.reader(output.splitlines()) if row]


def verify_devices(expected):
    rows = nvidia('--query-gpu=index,uuid,name,driver_version,memory.total,pci.bus_id')
    selected = {int(row[0]): row[1] for row in rows if int(row[0]) in (4, 5, 6)}
    require(selected == expected, 'GPU indices do not match the authorized UUIDs')
    require(PAUSE.is_file(), f'Keepalive must be paused by the operator at {PAUSE}')
    occupied = nvidia('--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory')
    require(not any(row[0] in selected.values() for row in occupied), 'An authorized GPU is occupied')
    return rows, occupied


def verify_ports():
    sockets = []
    try:
        for port in PORTS:
            listener = socket.socket()
            sockets.append(listener)
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(('127.0.0.1', port))
    finally:
        for listener in sockets:
            listener.close()


COMMON = """from pathlib import Path
import os
import subprocess
from ..dense_ar_slo_layer_pd.common import append, digest, write

REPO = Path(os.environ['MLLM_EPD_PINNED_REPO'])
ROOT = Path(os.environ['MLLM_EPD_SESSION_ROOT'])
SOURCE = Path(__file__).parent
PYTHON = Path(os.environ['MLLM_EPD_SESSION_PYTHON'])
MODEL_ID = 'Qwen/Qwen2.5-VL-7B-Instruct'

def preserve():
    head = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    if head != '449c3e1873c65206435a6bb9411a889a62e55be7':
        raise RuntimeError('Pinned checkout HEAD changed')
    subprocess.run(['git', '-C', str(REPO), 'diff', '--quiet', 'HEAD', '--'], check=True)

def experiment_deadline():
    return float(os.environ['MLLM_EPD_SESSION_DEADLINE'])

def initialize():
    raise RuntimeError('Historical campaign initialization is disabled for this session')

def resources():
    raise RuntimeError('Historical GPU resource guard is disabled for this session')
"""


def prepare_overlay(repo, run, research=None):
    source = (research or repo) / 'poc_flashvep'
    package = run / 'source/poc_flashvep'
    package.mkdir(parents=True)
    shutil.copyfile(source / '__init__.py', package / '__init__.py')
    for entry in source.iterdir():
        if entry.is_dir() and entry.name not in (PACKAGE, 'reports', '__pycache__'):
            (package / entry.name).symlink_to(entry, target_is_directory=True)
    target = package / PACKAGE
    target.mkdir()
    for source_file in (source / PACKAGE).glob('*.py'):
        if source_file.name != 'common.py':
            shutil.copyfile(source_file, target / source_file.name)
    (target / 'common.py').write_text(COMMON)
    return package.parent


def module(name):
    return importlib.import_module(f'poc_flashvep.{PACKAGE}.{name}')


def http_json(port, resource, timeout=3):
    with urllib.request.urlopen(f'http://127.0.0.1:{port}/{resource}', timeout=timeout) as response:
        require(response.status == 200, f'HTTP readiness failed on port {port}')
        if resource != 'health':
            return json.load(response)
    return None


def pid_owned(pid, session, nonce):
    proc = Path('/proc') / str(pid)
    try:
        return proc.stat().st_uid == os.getuid() and os.getsid(pid) == session and (
            f'MLLM_EPD_RUN_NONCE={nonce}'.encode() in (proc / 'environ').read_bytes().split(b'\0'))
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return False


class OwnedProcesses:
    def __init__(self, run, env, source, deadline):
        self.run = run
        self.env = env
        self.source = source
        self.deadline = deadline
        self.processes = []

    def start(self, label, command, **additions):
        env = self.env | additions
        with (self.run / 'logs' / f'{label}.log').open('xb') as log:
            process = subprocess.Popen(command, cwd=self.source, env=env,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
        self.processes.append((label, process))
        write(self.run / 'state' / f'owned_{label}.json',
              {'command': command, 'pid': process.pid, 'sid': process.pid,
               'uid': os.getuid(), 'nonce': env['MLLM_EPD_RUN_NONCE'], 'environment': additions})
        return process

    def ready(self, process, port, label, *, role=None):
        limit = min(self.deadline, time.time() + (1500 if role else 30))
        while time.time() < limit:
            require(process.poll() is None, f'{label} exited during initialization')
            try:
                http_json(port, 'health')
                if role:
                    worker, = http_json(port, 'campaign/worker_info', timeout=30)
                    require(worker['role'] == role and pid_owned(worker['pid'], process.pid,
                            self.env['MLLM_EPD_RUN_NONCE']), f'Unverified worker identity for {label}')
                    write(self.run / 'state' / f'worker_{label}.json', worker)
                else:
                    value = http_json(port, 'health?identity=1')
                    require(value.get('name') == label and process.poll() is None,
                            f'Unverified proxy identity for {label}')
                return
            except (urllib.error.URLError, TimeoutError):
                time.sleep(1)
        raise TimeoutError(f'Initialization deadline expired for {label}')

    def live_group(self, pid):
        return [int(proc.name) for proc in Path('/proc').iterdir() if proc.name.isdecimal()
                and pid_owned(int(proc.name), pid, self.env['MLLM_EPD_RUN_NONCE'])]

    def signal_owned(self, label, process, sig):
        live_group = self.live_group(process.pid)
        if not live_group:
            return
        for pid in live_group:
            try:
                require(os.getpgid(pid) == process.pid, f'Owned process left its group: {label}')
            except ProcessLookupError:
                continue
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass

    def stop(self, label, process):
        for sig, delay in ((signal.SIGTERM, 40), (signal.SIGKILL, 15)):
            self.signal_owned(label, process, sig)
            limit = time.monotonic() + delay
            while time.monotonic() < limit:
                process.poll()
                if not self.live_group(process.pid) and process.poll() is not None:
                    write(self.run / 'state' / f'finished_{label}.json',
                          {'returncode': process.returncode, 'last_signal': sig.name})
                    return
                time.sleep(.3)
        raise RuntimeError(f'Owned process did not exit: {label}, group {process.pid}')

    def close(self):
        errors = []
        for label, process in reversed(self.processes):
            try:
                self.stop(label, process)
            except Exception as exc:
                errors.append(str(exc))
        if errors:
            raise RuntimeError('; '.join(errors))


def launch_models(group, tag, ec, uuid_by_gpu, media, token=False, common=True, prefill_kv_ttl=None):
    servers = module('servers')
    ready = []
    for role, gpu, port, side_port in ROLES:
        label = f'{tag}_{role}'
        encoder = role == 'encoder'
        command = servers.command(
            role, port, ec, side_port, observe=True, encoder_native_ops=encoder,
            compiled_encoder=encoder, encoder_sync_scheduling=encoder,
            mm_cache_gb=0 if encoder else 1, renderer_workers=2 if encoder else 1,
            ec_transport='cuda_ipc', token_split=token and not encoder,
            fixed_reduction=common and not encoder, deterministic_reductions=common and not encoder,
        )
        command[command.index('--allowed-local-media-path') + 1] = str(media)
        if prefill_kv_ttl is not None and not encoder:
            config_index = command.index('--kv-transfer-config') + 1
            config = json.loads(command[config_index])
            config.setdefault('kv_connector_extra_config', {})['kv_lease_duration'] = prefill_kv_ttl
            command[config_index] = json.dumps(config)
        process = group.start(
            label, command, CUDA_VISIBLE_DEVICES=uuid_by_gpu[gpu],
            VLLM_NIXL_SIDE_CHANNEL_PORT=str(side_port), MLLM_EPD_ROLE=role,
            MLLM_EPD_TOKEN_SPLIT='1' if token and not encoder else '0',
            VLLM_CACHE_ROOT=str(group.run / 'runtime_cache' / role),
            MLLM_EPD_EC_PATH=str(ec), MLLM_EPD_METADATA_PATH=str(ec / 'processed_prompts'),
        )
        ready.append((process, port, label, role))
    for process, port, label, role in ready:
        group.ready(process, port, label, role=role)
    occupied = nvidia('--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory')
    for (_, gpu, _, _), (process, _, label, _) in zip(ROLES, ready, strict=True):
        uuid = uuid_by_gpu[gpu]
        pids = [int(row[1]) for row in occupied if row[0] == uuid]
        require(pids and all(pid_owned(pid, process.pid, group.env['MLLM_EPD_RUN_NONCE']) for pid in pids),
                f'GPU ownership verification failed for {label} on {uuid}')
    write(group.run / 'state' / f'{tag}_gpu_processes.json', occupied)
    module('native_common_control').set_common_ipc_policy()


def launch_proxy(group, label, mode, port, token=False, batch_one=False, idle=False, fraction=.10, decode_cap=None):
    command = [sys.executable, '-m', f'poc_flashvep.{PACKAGE}.proxy', '--mode', mode,
               '--name', label, '--encoder', 'http://127.0.0.1:18840',
               '--prefill', 'http://127.0.0.1:18841',
               '--decoders', f'http://127.0.0.1:{18841 if mode == "e_pd" else 18842}',
               '--port', str(port), '--processor-handoff', 'grid_only', '--release-encoder-ipc']
    if token:
        command += ['--token-suffix-fraction', str(fraction), '--token-admission', 'decode_idle',
                    '--token-policy', 'idle_fallback' if idle else 'static']
    if batch_one:
        command += ['--decode-batch-one']
    if decode_cap not in (None, 'free'):
        command += ['--decode-cap', str(decode_cap)]
    process = group.start(label, command)
    group.ready(process, port, label)
    return f'http://127.0.0.1:{port}'

#!/usr/bin/env python3
"""Check the native training workflow end to end. Needs a GPU and the training
models; takes about an hour.

1. Parity: training_workflow.py and `yue2-train-run workflow` each take the
   same run through prepare, train and publish, in the same run folder, one
   after the other. Every file must be equal except logs, progress records,
   timings and peak readings; the published style must be byte-equal.
2. Resume: the Python workflow, running as it ships (its own script digests),
   is stopped during scores and, in a second run, during NAR conditioning.
   The native workflow resumes each run and publishes it.
3. Server: a run created, trained and published through yue2-server's
   training API, whose config names no Python.

python tools/check_native_workflow.py --runtime training-runtime.json
    --bin build/bin/Release --dataset RECORDINGS --out DIR [--parts parity,resume,server]

--runtime supplies the models and the Python and workspace the Python
workflow needs; --bin holds yue2-train-run, yue2-server and the native tools.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'tools'))
import check_native_fitting  # noqa: E402
import check_native_preparation  # noqa: E402
from training_workflow import process_identity  # noqa: E402

REQUEST = dict(window_seconds=8, rank=4, ar_steps=6, nar_steps=8, eval_every=2, recipe_version=5, family='lora')
STATE_VOLATILE = {'updated_at', 'free_disk_gib', 'progress', 'at', 'child_pid', 'log'}

# Makes the Python workflow and its scripts record the committed (LF) digests
# of the reference scripts, as the native side does.
SITECUSTOMIZE = '''
import hashlib, sys
from pathlib import Path
sys.path.insert(0, {tools!r})
import training_provenance
_real = training_provenance.digest
def digest(path):
    if Path(path).name in ('ar_dataset.py', 'prepare_nar_dataset.py'):
        return hashlib.sha256(Path(path).read_bytes().replace(b'\\r\\n', b'\\n')).hexdigest()
    return _real(path)
training_provenance.digest = digest
'''


def launch(command, run, env=None):
    """Start a workflow and commit its ownership the way yue2-server does."""
    number = 1
    while (run.parent / f'{run.name}-launch-{number}.log').exists(): number += 1
    output = run.parent / f'{run.name}-launch-{number}.log'
    with output.open('w') as stream:
        process = subprocess.Popen(command, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT)
    process.output = output
    (run / 'owner.json').write_text(json.dumps(dict(pid=process.pid, identity=process_identity(process.pid))))
    return process


def finish(process, run, expected):
    process.wait()
    output = process.output.read_text(encoding='utf-8', errors='replace')
    state = json.loads((run / 'state.json').read_text())
    if state.get('status') != expected:
        raise RuntimeError(f'{run.name}: {state.get("status")} instead of {expected}: {state.get("error", "")[-2000:]}\n{output[-2000:]}')
    return state


def python_workflow(runtime, run, action, env):
    return launch([runtime['python'], str(REPO / 'tools/training_workflow.py'), '--run', str(run), '--action', action], run, env)


def native_workflow(binary, run, action):
    return launch([str(binary), 'workflow', '--run', str(run), '--action', action], run)


def new_run(root, name, dataset, runtime):
    run = root / name; (run / 'data').mkdir(parents=True)
    (run / 'runtime.json').write_text(json.dumps(runtime))
    (run / 'request.json').write_text(json.dumps(dict(id=name, name=name, dataset=str(dataset), **REQUEST)))
    return run


def stable_state(path):
    # caches_released counts the released source.wav files, whose header
    # differs from libsndfile's (which records a write time); the files left
    # behind are compared instead.
    state = json.loads(path.read_text())
    return check_native_fitting.stable({k: v for k, v in state.items() if k not in STATE_VOLATILE | {'caches_released'}})


def parity(args, runtime, models, env):
    root = args.out / 'runs'; results = {}
    for implementation in ['python', 'native']:
        run = new_run(root, 'parity', args.out / 'dataset', runtime)
        for action, expected in [('prepare', 'ready'), ('train', 'review'), ('publish', 'published')]:
            process = python_workflow(runtime, run, action, env) if implementation == 'python' else native_workflow(args.train_run, run, action)
            finish(process, run, expected)
        run.rename(root / f'parity-{implementation}')
        (models / 'parity-F32-LoRA.gguf').rename(models / f'parity-{implementation}.gguf')
        (models / 'parity-F32-LoRA.training.json').rename(models / f'parity-{implementation}.training.json')
    problems, compared = check_native_fitting.compare(root / 'parity-python', root / 'parity-native')
    if stable_state(root / 'parity-python/state.json') != stable_state(root / 'parity-native/state.json'): problems.append('state.json differs')
    for name in ['gguf', 'training.json']:
        if (models / f'parity-python.{name}').read_bytes() != (models / f'parity-native.{name}').read_bytes(): problems.append(f'published {name} differs')
    results.update(compared_files=compared, problems=problems)
    return results


def stop_when(process, run, ready):
    while process.poll() is None:
        try:
            if ready(): (run / 'stop.request').write_text('stop\n'); break
        except (OSError, ValueError): pass
        time.sleep(.2)


def resume(args, runtime, models):
    root = args.out / 'runs'; results = {}
    scores = new_run(root, 'resume-scores', args.out / 'dataset', runtime)
    process = python_workflow(runtime, scores, 'prepare', None)
    stop_when(process, scores, lambda: len(list((scores / 'data/examples').glob('*/example.json'))) >= 2)
    finish(process, scores, 'stopped')
    conditioning = new_run(root, 'resume-conditioning', args.out / 'dataset', runtime)
    finish(python_workflow(runtime, conditioning, 'prepare', None), conditioning, 'ready')
    process = python_workflow(runtime, conditioning, 'train', None)
    stop_when(process, conditioning, lambda: json.loads((conditioning / 'nar-conditioning/progress.json').read_text()).get('current', 0) >= 1)
    finish(process, conditioning, 'stopped')
    for run in [scores, conditioning]:
        (run / 'stop.request').unlink(missing_ok=True)
        state = finish(native_workflow(args.train_run, run, 'resume'), run, 'ready' if run is scores else 'review')
        if run is scores: state = finish(native_workflow(args.train_run, run, 'train'), run, 'review')
        state = finish(native_workflow(args.train_run, run, 'publish'), run, 'published')
        normalizer = json.loads((run / 'data/dataset.json').read_text())['tools']['normalizer_sha256']
        source = json.loads((run / 'nar-conditioning/identity.json').read_text())['preparation_source_sha256']
        results[run.name] = dict(status=state['status'], normalizer_sha256=normalizer, preparation_source_sha256=source,
                                 published=Path(state['export']['adapter']).exists(),
                                 native_scores=(run / 'scores-2.log').exists(), native_conditioning=(run / 'nar-conditioning-2.log').exists())
    return results


def server(args, runtime_path):
    port = 8031; root = args.out / 'server'
    config = root / 'training-runtime.json'; root.mkdir()
    subprocess.run([str(args.train_run), 'setup', '--bin', str(args.bin), '--models-dir', str(args.models_dir),
                    '--run-root', str(root / 'runs'), '--out', str(config)], check=True)
    settings = json.loads(config.read_text())
    if any(key in settings for key in ['python', 'workspace']): raise RuntimeError('Native setup names Python')
    # Publish into the folder the server lists adapters from, not the models.
    settings['models_dir'] = str(root / 'adapters'); config.write_text(json.dumps(settings, indent=2))
    log = (root / 'server.log').open('w')
    process = subprocess.Popen([str(args.bin / ('yue2-server.exe' if os.name == 'nt' else 'yue2-server')), '--models-dir',
                                str(args.models_dir), '--adapters-dir', str(root / 'adapters'),
                                '--port', str(port), '--training-config', str(config)], stdout=log, stderr=subprocess.STDOUT)

    def api(path, body=None):
        request = urllib.request.Request(f'http://127.0.0.1:{port}{path}', method='POST' if body is not None else 'GET',
                                         data=json.dumps(body).encode() if body is not None else None, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=30) as response: return json.load(response)

    def wait(status):
        while True:
            run = next(r for r in api('/training/api/runs')['runs'] if r['id'] == 'served')
            if run['state'].get('status') == status and not run['active']: return run['state']
            if run['state'].get('status') == 'failed': raise RuntimeError(run['state'].get('error', '')[-3000:])
            time.sleep(2)
    try:
        for _ in range(60):
            try: api('/training/api'); break
            except OSError: time.sleep(1)
        api('/training/api/runs', dict(id='served', name='Served', dataset=str(args.out / 'dataset'), **{k: v for k, v in REQUEST.items() if k != 'recipe_version'}))
        wait('ready'); api('/training/api/runs/served/train', {}); wait('review')
        api('/training/api/runs/served/publish', {}); state = wait('published')
        loras = api('/loras')
        listed = any('served' in json.dumps(entry).lower() for entry in (loras if isinstance(loras, list) else loras.get('loras', loras.get('adapters', []))))
        return dict(status=state['status'], adapter=state['export']['adapter'], listed_in_loras=listed)
    finally:
        process.terminate(); process.wait(); log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runtime', type=Path, required=True, help='Training runtime with python and workspace, for the Python workflow')
    parser.add_argument('--bin', type=Path, required=True, help='Native tools, including yue2-train-run and yue2-server')
    parser.add_argument('--dataset', type=Path, required=True, help='Recordings with TXT captions')
    parser.add_argument('--out', type=Path, required=True, help='New folder')
    parser.add_argument('--models-dir', type=Path, default=REPO / 'models', help='Installed models, for the server part')
    parser.add_argument('--parts', default='parity,resume,server')
    args = parser.parse_args()
    if args.out.exists(): parser.error('Output exists; choose a new folder')
    args.out = args.out.resolve(); args.bin = args.bin.resolve()
    args.train_run = args.bin / ('yue2-train-run.exe' if os.name == 'nt' else 'yue2-train-run')
    check_native_preparation.make_dataset(args.dataset, args.out / 'dataset', 3, 20)
    models = args.out / 'models'; models.mkdir()
    runtime = json.loads(args.runtime.read_text())
    runtime.update(bin=str(args.bin), run_root=str(args.out / 'runs'), models_dir=str(models))
    patch = args.out / 'lf-digests'; patch.mkdir()
    (patch / 'sitecustomize.py').write_text(SITECUSTOMIZE.format(tools=str(REPO / 'tools')))
    env = dict(os.environ, PYTHONPATH=str(patch))
    results = {}
    parts = args.parts.split(',')
    if 'parity' in parts: results['parity'] = parity(args, runtime, models, env); print('parity', json.dumps(results['parity']), flush=True)
    if 'resume' in parts: results['resume'] = resume(args, runtime, models); print('resume', json.dumps(results['resume']), flush=True)
    if 'server' in parts: results['server'] = server(args, args.runtime); print('server', json.dumps(results['server']), flush=True)
    (args.out / 'results.json').write_text(json.dumps(results, indent=2))
    ok = (not results.get('parity', {}).get('problems')) and all(r['status'] == 'published' and r['published'] for r in results.get('resume', {}).values()) \
        and results.get('server', {}).get('status', 'published') == 'published'
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

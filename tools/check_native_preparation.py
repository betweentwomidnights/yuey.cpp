#!/usr/bin/env python3
"""Check that yue2-train-run prepares a run exactly as the Python does.

Each implementation prepares the same dataset in the same run folder, one
after the other: Workflow.prepare() (semantic caches, scored AR examples, VAE
targets), then frozen-AR conditioning with cache and rolling storage, then a
rolling regeneration. Every file must be equal, except logs, timings and the
source.wav header, whose samples must be equal. Needs a GPU and the training
models; runs take minutes.

python tools/check_native_preparation.py --runtime training-runtime.json
    --train-run build/bin/Release/yue2-train-run.exe --dataset RECORDINGS
    --ar-adapter AR_ADAPTER.gguf --out DIR [--tracks 3] [--clip-seconds 20] [--window-seconds 8]

The first --tracks recordings of --dataset are cut to --clip-seconds (one is
written as 24-bit FLAC, the others as float WAV) with their captions. The Python's own source digests (which
follow the checkout's line endings) are replaced by digests of the committed
LF bytes, which the native side records.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'tools'))
import training_workflow  # noqa: E402

# Runs one Python preparation script with LF digests of the reference scripts.
WRAPPER = '''
import hashlib, importlib, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import training_provenance
real = training_provenance.digest
def digest(path):
    if Path(path).name in ('ar_dataset.py', 'prepare_nar_dataset.py'):
        return hashlib.sha256(Path(path).read_bytes().replace(b'\\r\\n', b'\\n')).hexdigest()
    return real(path)
module = importlib.import_module(sys.argv[2])
module.digest = digest
if sys.argv[3] == 'materialize':
    import json, os
    # train_nar_run.py, which calls materialize(), pins TF32 off first.
    os.environ['GGML_CUDA_TF32'] = '0'
    rows = json.loads(Path(sys.argv[5]).read_text())
    print(module.materialize(sys.argv[4], rows, *sys.argv[6:10]))
else:
    sys.argv = [sys.argv[2] + '.py', *sys.argv[3:]]
    module.main()
'''

NATIVE = {'prepare_ar_dataset.py': lambda a: ['prepare-' + a[a.index('--phase') + 1], '--config', a[a.index('--config') + 1]],
          'prepare_nar_targets.py': lambda a: ['prepare-targets', *a],
          'prepare_nar_dataset.py': lambda a: ['prepare-conditioning', *a]}


def lf_digest(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def make_dataset(source, out, tracks, seconds):
    out.mkdir(parents=True)
    audio = sorted(p for p in source.iterdir() if p.suffix.lower() in ['.wav', '.flac'] and p.with_suffix('.txt').exists())[:tracks]
    for index, path in enumerate(audio):
        data, rate = sf.read(path, dtype='float32', always_2d=True, frames=int(seconds * sf.info(path).samplerate))
        target = out / (path.stem + ('.flac' if index == 1 else '.wav'))
        sf.write(target, data, rate, subtype='PCM_24' if index == 1 else 'FLOAT')
        shutil.copyfile(path.with_suffix('.txt'), target.with_suffix('.txt'))


def run(command, cwd, log):
    with open(log, 'w', encoding='utf-8') as output:
        result = subprocess.run(command, cwd=cwd, stdout=output, stderr=subprocess.STDOUT)
    if result.returncode: raise RuntimeError(log.read_text(encoding='utf-8', errors='replace')[-4000:])


def prepare(implementation, run_folder, args, runtime, wrapper):
    """One implementation through review, both conditioning modes and a regeneration."""
    tools = REPO / 'tools'

    def command(script, script_args):
        script_args = [str(a) for a in script_args]
        if implementation == 'native': return [str(args.train_run), *NATIVE[script](script_args)]
        return [sys.executable, str(wrapper), str(tools), script[:-3], *script_args]

    def patched(self, script, script_args, stage, progress=None):
        number = 1
        while (self.run / f'{stage}-{number}.log').exists(): number += 1
        run(command(script, script_args), self.workspace, self.run / f'{stage}-{number}.log')
    original = training_workflow.Workflow.command
    training_workflow.Workflow.command = patched
    try:
        workflow = training_workflow.Workflow(run_folder)
        workflow.prepare()
    finally:
        training_workflow.Workflow.command = original
    binary = Path(runtime['bin']) / ('yue2-train.exe' if os.name == 'nt' else 'yue2-train')
    for storage in ['cache', 'rolling']:
        run(command('prepare_nar_dataset.py', ['--config', run_folder / 'ar-job.json', '--targets', run_folder / 'vae-targets/index.json',
                                               '--ar-adapter', args.ar_adapter, '--binary', binary, '--out', run_folder / f'nar-{storage}',
                                               '--conditioning-device', runtime.get('device', 'cuda'), '--conditioning-batch', '0',
                                               '--stop-file', run_folder / 'stop.request', '--storage', storage]),
            REPO, run_folder / f'nar-{storage}.log')
    manifest = run_folder / 'nar-rolling/dataset.json'
    rows = json.loads(manifest.read_text())['rows'][:2]
    (run_folder / 'materialize-rows.json').write_text(json.dumps(rows))
    models = runtime['models']; adapter = str(Path(args.ar_adapter).resolve())
    if implementation == 'native':
        command_line = [str(args.train_run), 'materialize', '--manifest', str(manifest), '--rows', str(run_folder / 'materialize-rows.json'),
                        '--binary', str(binary), '--base', models['base'], '--common', models['companion'], '--adapter', adapter]
    else:
        command_line = [sys.executable, str(wrapper), str(tools), 'prepare_nar_dataset', 'materialize', str(manifest),
                        str(run_folder / 'materialize-rows.json'), str(binary), models['base'], models['companion'], adapter]
    run(command_line, REPO, run_folder / 'materialize-check.log')
    return all(Path(r['cache']).exists() for r in rows)


def compare(python, native):
    """Relative paths whose contents differ, and files only one side has."""
    files = lambda root: {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    left, right = files(python), files(native)
    problems = [f'only python: {p}' for p in sorted(left - right)] + [f'only native: {p}' for p in sorted(right - left)]
    compared = 0
    for name in sorted(left & right):
        a, b = python / name, native / name
        if name.endswith('.log') or name in ['state.json', 'materialize-rows.json'] or re.fullmatch(r'[^/]+\.command\.json', name):
            continue
        compared += 1
        if name.endswith('command.json'):
            strip = lambda p: {k: v for k, v in json.loads(p.read_text()).items() if k != 'seconds'}
            same = strip(a) == strip(b)
        elif name.endswith('source.wav'):
            (x, rate_x), (y, rate_y) = sf.read(a, dtype='float32'), sf.read(b, dtype='float32')
            same = rate_x == rate_y and np.array_equal(x, y)
        else:
            same = a.read_bytes() == b.read_bytes()
        if not same: problems.append(f'differs: {name}')
    return problems, compared


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--train-run', type=Path, required=True, help='yue2-train-run')
    parser.add_argument('--dataset', type=Path, required=True, help='Recordings with TXT captions')
    parser.add_argument('--ar-adapter', type=Path, required=True, help='Any AR adapter for the base, to condition with')
    parser.add_argument('--out', type=Path, required=True, help='New folder')
    parser.add_argument('--tracks', type=int, default=3)
    parser.add_argument('--clip-seconds', type=float, default=20)
    parser.add_argument('--window-seconds', type=int, default=8)
    args = parser.parse_args()
    if args.out.exists(): parser.error('Output exists; choose a new folder')
    args.out = args.out.resolve(); args.train_run = args.train_run.resolve()
    constants = dict(re.findall(r'const char \* const (\w+) = "([0-9a-f]{64})"', (REPO / 'tools/training/preparation.cpp').read_text()))
    pinned = {'normalizer_sha256': lf_digest(REPO / 'tools/ar_dataset.py'),
              'conditioning_sha256': lf_digest(REPO / 'tools/prepare_nar_dataset.py')}
    if constants != pinned: raise SystemExit(f'Native reference digests are stale: {constants} != {pinned}')
    make_dataset(args.dataset, args.out / 'dataset', args.tracks, args.clip_seconds)
    wrapper = args.out / 'python_preparation.py'; wrapper.write_text(WRAPPER)
    runtime = json.loads(args.runtime.read_text())
    runtime.update(run_root=str(args.out / 'runs'), models_dir=str(args.out / 'models'))
    results = {}
    for implementation in ['python', 'native']:
        folder = args.out / 'runs/parity'; (folder / 'data').mkdir(parents=True)
        (folder / 'runtime.json').write_text(json.dumps(runtime))
        (folder / 'request.json').write_text(json.dumps(dict(id='parity', name='Parity', dataset=str(args.out / 'dataset'),
            window_seconds=args.window_seconds, rank=4, recipe_version=5)))
        regenerated = prepare(implementation, folder, args, runtime, wrapper)
        folder.rename(args.out / f'runs/{implementation}')
        results[implementation] = dict(regenerated=regenerated)
        print(implementation, 'prepared', flush=True)
    problems, compared = compare(args.out / 'runs/python', args.out / 'runs/native')
    examples = json.loads((args.out / 'runs/native/data/dataset.json').read_text())['rows']
    results.update(compared_files=compared, examples=len(examples), eligible=sum(r['eligible'] for r in examples), problems=problems)
    (args.out / 'results.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return 0 if not problems and all(results[i]['regenerated'] for i in ['python', 'native']) else 1


if __name__ == '__main__':
    sys.exit(main())

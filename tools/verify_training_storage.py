#!/usr/bin/env python3
"""Check that recipe 5 storage modes cannot change training.

A cached-mode run through yue2-server is the baseline. The AR fit is repeated
with an online teacher (`online`); the same frozen AR and VAE targets are
re-prepared and fitted with each NAR conditioning mode (`rolling`,
`resident`). Final optimizer state, RNG and every history row must equal the
baseline; for NAR, the adapter and every evaluation too. Use a small dataset:
this is an equivalence check, not a training recipe.

python tools/verify_training_storage.py --runtime training-runtime.json --out DIR
    [--dataset FOLDER] [--modes online,rolling,resident] [--port 8019]
    [--window-seconds 4 --rank 4 --ar-steps 6 --nar-steps 12 --eval-every 4]

Results also report preparation/fit time and the median update for each mode
beside the cached baseline, so a real dataset at 30-second windows and
rank 32 measures what each mode costs on this machine.

Without --dataset, two short synthetic piano recordings with captions are
written to DIR/dataset.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request

import gguf
import numpy as np

REPO = Path(__file__).resolve().parents[1]


def synthesize(folder):
    """Two 8-second D minor chord progressions at 88 bpm, 48 kHz stereo."""
    import soundfile as sf
    folder.mkdir(parents=True)
    rate, beat = 48000, 60 / 88
    chords = [[62, 65, 69], [58, 62, 65], [60, 64, 67], [57, 61, 64]]
    for index in range(2):
        audio = np.zeros(int(8 * rate))
        for n in range(int(8 / beat)):
            start = int(n * beat * rate)
            t = np.arange(int(beat * rate * 1.5)) / rate
            for note in chords[(n // 2 + index) % 4]:
                frequency = 440 * 2 ** ((note - 69) / 12)
                tone = sum(np.sin(2 * math.pi * frequency * k * t) / k for k in range(1, 5)) * np.exp(-3 * t)
                end = min(len(audio), start + len(t))
                audio[start:end] += .08 * tone[:end - start]
        sf.write(folder / f'piano-{index}.wav', np.column_stack([audio, audio]), rate, subtype='FLOAT')
        (folder / f'piano-{index}.txt').write_text('orchestral piano, intricate chords, 88 bpm, D minor\n', encoding='utf-8')


def tensors(path):
    reader = gguf.GGUFReader(str(path))
    return {t.name: hashlib.sha256(t.data.tobytes()).hexdigest() for t in reader.tensors}


def rng(path):
    return gguf.GGUFReader(str(path)).get_field('yue2.training.rng').contents()


def latest_state(folder):
    return max(folder.glob('native*/step-*.state.gguf'), key=lambda p: int(p.name.split('-')[1].split('.')[0]))


def history(folder):
    rows = []
    for path in sorted(folder.glob('native*/history.jsonl'), key=lambda p: int(p.parent.name.split('-')[-1])):
        rows += [{k: v for k, v in json.loads(line).items() if not k.endswith('seconds')}
                 for line in path.read_text().splitlines()]
    return rows


def evaluations(folder):
    selection = json.loads((folder / 'selection.json').read_text())
    return selection['selected']['step'], [{k: v for k, v in m.items() if k != 'segment'} for m in selection['evaluations']]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runtime', type=Path, required=True, help='training-runtime.json from setup_training.py')
    parser.add_argument('--out', type=Path, required=True, help='New folder for runs and results')
    parser.add_argument('--dataset', type=Path, help='WAV/FLAC recordings with TXT captions (default: synthetic)')
    parser.add_argument('--modes', default='online,rolling,resident')
    parser.add_argument('--port', type=int, default=8019)
    parser.add_argument('--rolling-window', type=int, default=3, help='Small windows force many segments')
    parser.add_argument('--window-seconds', type=int, default=4)
    parser.add_argument('--rank', type=int, default=4)
    parser.add_argument('--ar-steps', type=int, default=6)
    parser.add_argument('--nar-steps', type=int, default=12)
    parser.add_argument('--eval-every', type=int, default=4)
    args = parser.parse_args()
    requested = args.modes.split(',')
    if not set(requested) <= {'online', 'rolling', 'resident'}: parser.error('modes are online, rolling and/or resident')
    teacher_modes = [m for m in requested if m == 'online']; modes = [m for m in requested if m != 'online']
    if args.out.exists(): parser.error('Output exists; choose a new folder')
    os.environ['GGML_CUDA_TF32'] = '0'
    runtime = json.loads(args.runtime.read_text())
    suffix = '.exe' if os.name == 'nt' else ''
    binary = Path(runtime['bin'])
    args.out.mkdir(parents=True)
    dataset = args.dataset or args.out / 'dataset'
    if not args.dataset: synthesize(dataset)
    config = dict(runtime, run_root=str((args.out / 'runs').resolve()), models_dir=str((args.out / 'models').resolve()))
    (args.out / 'runs').mkdir(); (args.out / 'models').mkdir()
    (args.out / 'training-runtime.json').write_text(json.dumps(config, indent=2))

    def api(path, body=None):
        request = urllib.request.Request(f'http://127.0.0.1:{args.port}{path}', method='POST' if body is not None else 'GET',
                                         data=json.dumps(body).encode() if body is not None else None,
                                         headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=30) as response: return json.load(response)

    run = args.out / 'runs/storage'

    def wait(predicate, what):
        while True:
            try:
                state = json.loads((run / 'state.json').read_text())
                if state.get('status') == 'failed': raise RuntimeError(what + ': ' + str(state.get('error'))[-3000:])
                if predicate(state): return state
            except (OSError, json.JSONDecodeError): pass
            time.sleep(2)

    server = subprocess.Popen([str(binary / ('yue2-server' + suffix)), '--models-dir', runtime['models_dir'],
                               '--training-config', str(args.out / 'training-runtime.json'), '--port', str(args.port)],
                              stdout=(args.out / 'server.log').open('w'), stderr=subprocess.STDOUT)
    try:
        for _ in range(120):
            try: api('/training/api'); break
            except OSError: time.sleep(1)
        api('/training/api/runs', dict(id='storage', name='Storage verify', trigger='storage_verify',
            dataset=str(dataset.resolve()).replace('\\', '/'), window_seconds=args.window_seconds,
            ar_steps=args.ar_steps, nar_steps=args.nar_steps, eval_every=args.eval_every, rank=args.rank,
            family='lora', keep_caches=True))
        wait(lambda s: s.get('status') == 'ready', 'prepare')
        api('/training/api/runs/storage/train', {})
        wait(lambda s: s.get('status') == 'review', 'cached baseline')
    finally:
        server.terminate(); server.wait()
    baseline = latest_state(run / 'nar-run')

    def update_median(folder):
        values = [json.loads(line) for path in folder.glob('native*/history.jsonl') for line in path.read_text().splitlines()]
        return round(float(np.median([v['seconds'] for v in values if 'loss' in v])), 3)

    def fit_seconds(folder):
        return round(sum(json.loads(p.read_text())['seconds'] for p in folder.glob('fit-native-*.command.json')), 1)
    conditioning = run / 'nar-conditioning'
    results = dict(cached=dict(nar_update_seconds=update_median(run / 'nar-run'), nar_fit_seconds=fit_seconds(run / 'nar-run'),
                               conditioning_gib=round(sum(p.stat().st_size for p in conditioning.rglob('*.gguf')) / 2**30, 3)))
    print('cached', results['cached'], flush=True)
    if 'online' in teacher_modes:
        # The workflow releases the teacher cache after AR selection, so the
        # cached fit is the baseline and the online fit repeats it in full.
        lora = run / 'data/lora'
        command = json.loads(next(lora.glob('fit-native-*.command.json')).read_text())['command']
        for key in ['--teacher-cache', '--teacher-identity', '--teacher-base-sha256', '--teacher-companion-sha256',
                    '--out', '--steps', '--stop-file']:
            if key in command: i = command.index(key); del command[i:i + 2]
        out = args.out / 'ar-online'
        subprocess.run(command + ['--teacher-online', '--out', str(out), '--steps', str(args.ar_steps)], cwd=REPO, check=True,
                       stdout=(args.out / 'ar-online.log').open('w'), stderr=subprocess.STDOUT)
        cached_ar = lora / 'native-1' / f'step-{args.ar_steps}.state.gguf'
        final = out / f'step-{args.ar_steps}.state.gguf'
        strip = lambda path: [{k: v for k, v in json.loads(l).items() if not k.endswith('seconds')} for l in path.read_text().splitlines()]
        updates = lambda path: round(float(np.median([json.loads(l)['seconds'] for l in path.read_text().splitlines()])), 3)
        results['ar-cached'] = dict(ar_update_seconds=updates(lora / 'native-1/history.jsonl'))
        results['ar-online'] = dict(ar_update_seconds=updates(out / 'history.jsonl'),
                                    state=tensors(final) == tensors(cached_ar), rng=rng(final) == rng(cached_ar),
                                    history=strip(out / 'history.jsonl') == strip(lora / 'native-1/history.jsonl'))
        results['ar-online']['identical'] = all(results['ar-online'][k] for k in ['state', 'rng', 'history'])
        print('ar-online', results['ar-online'], flush=True)
    job = json.loads((run / 'nar-job.json').read_text())
    ar = json.loads((run / 'data/lora/selection.json').read_text())['adapter']
    for mode in modes:
        folder = args.out / mode; folder.mkdir()
        started = time.monotonic()
        subprocess.run([sys.executable, str(REPO / 'tools/prepare_nar_dataset.py'), '--config', str(run / 'ar-job.json'),
                        '--targets', str(run / 'vae-targets/index.json'), '--ar-adapter', ar,
                        '--binary', str(binary / ('yue2-train' + suffix)), '--out', str(folder / 'nar-conditioning'),
                        '--conditioning-device', runtime.get('device', 'cuda'), '--conditioning-batch', '0',
                        '--storage', mode], cwd=REPO, check=True)
        mode_job = json.loads(json.dumps(job))
        mode_job.update(output=str(folder / 'nar-run'), training_dataset=str(folder / 'nar-conditioning/dataset.json'))
        mode_job['training']['conditioning_storage'] = mode
        if mode == 'rolling': mode_job['training']['rolling_window_updates'] = args.rolling_window
        (folder / 'nar-job.json').write_text(json.dumps(mode_job))
        preparation = round(time.monotonic() - started, 1)
        for phase in ['probe', 'fit']:
            subprocess.run([sys.executable, str(REPO / 'tools/train_nar_run.py'), '--config', str(folder / 'nar-job.json'),
                            '--phase', phase, *(['--segment', 'native-1'] if phase == 'fit' else [])], cwd=REPO, check=True)
        final = latest_state(folder / 'nar-run')
        adapter = final.with_name(final.name.replace('.state', ''))
        results[mode] = dict(
            preparation_seconds=preparation, nar_fit_seconds=fit_seconds(folder / 'nar-run'),
            nar_update_seconds=update_median(folder / 'nar-run'), segments=len(list((folder / 'nar-run').glob('native-*'))),
            caches_left=len(list((folder / 'nar-conditioning').rglob('sample.gguf'))),
            state=tensors(final) == tensors(baseline),
            adapter=tensors(adapter) == tensors(baseline.with_name(baseline.name.replace('.state', ''))),
            rng=rng(final) == rng(baseline), history=history(folder / 'nar-run') == history(run / 'nar-run'),
            evaluations=evaluations(folder / 'nar-run') == evaluations(run / 'nar-run'))
        results[mode]['identical'] = all(results[mode][k] for k in ['state', 'adapter', 'rng', 'history', 'evaluations'])
        print(mode, results[mode], flush=True)
    (args.out / 'results.json').write_text(json.dumps(results, indent=2))
    checked = [results[m] for m in modes] + [results['ar-online']] * bool(teacher_modes)
    return 0 if all(r['identical'] and not r.get('caches_left') for r in checked) else 1


if __name__ == '__main__':
    sys.exit(main())

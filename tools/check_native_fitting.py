#!/usr/bin/env python3
"""Check that yue2-train-run prepares and fits a run exactly as the Python does.

Each implementation runs Workflow.prepare() and Workflow.train() on the same
clips in the same run folder, one after the other: semantic caches, scores,
VAE targets, AR probe and fit with heldout selection, teacher release, NAR
conditioning, NAR probe and fit. Two runs cover the storage modes: a LoRA with
cached teachers and cached conditioning, and a DoRA with an online teacher and
rolling conditioning. Every file must be equal except logs, progress and
process records, timings, peak-memory readings and the source.wav header.
Needs a GPU and the training models; takes tens of minutes.

python tools/check_native_fitting.py --runtime training-runtime.json
    --train-run build/bin/Release/yue2-train-run.exe --dataset RECORDINGS --out DIR
"""
import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'tools'))
import check_native_preparation as preparation  # noqa: E402
import training_workflow  # noqa: E402

NATIVE = dict(preparation.NATIVE, **{'train_ar_run.py': lambda a: ['fit-ar', *a], 'train_nar_run.py': lambda a: ['fit-nar', *a]})

RUNS = {'lora-cached': dict(family='lora', teacher_storage='cache', conditioning_storage='cache'),
        'dora-online-rolling': dict(family='dora', teacher_storage='online', conditioning_storage='rolling')}

# Keys whose values are timings or machine readings, at any depth.
VOLATILE = re.compile(r'.*(seconds|_peak_|peak_mib|pid).*')


def stable(value):
    if isinstance(value, dict): return {k: stable(v) for k, v in value.items() if not VOLATILE.fullmatch(k)}
    if isinstance(value, list): return [stable(v) for v in value]
    return value


def train(implementation, run_folder, args, wrapper):
    def command(script, script_args):
        script_args = [str(a) for a in script_args]
        if implementation == 'native': return [str(args.train_run), *NATIVE[script](script_args)]
        return [sys.executable, str(wrapper), str(REPO / 'tools'), script[:-3], *script_args]

    def patched(self, script, script_args, stage, progress=None):
        self.check_stop()
        number = 1
        while (self.run / f'{stage}-{number}.log').exists(): number += 1
        preparation.run(command(script, script_args), self.workspace, self.run / f'{stage}-{number}.log')
    original = training_workflow.Workflow.command
    training_workflow.Workflow.command = patched
    try:
        training_workflow.Workflow(run_folder).prepare()
        training_workflow.Workflow(run_folder).train()
    finally:
        training_workflow.Workflow.command = original


def compare(python, native):
    files = lambda root: {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    left, right = files(python), files(native)
    problems = [f'only python: {p}' for p in sorted(left - right)] + [f'only native: {p}' for p in sorted(right - left)]
    compared = 0
    for name in sorted(left & right):
        a, b = python / name, native / name
        base = name.rsplit('/', 1)[-1]
        if base.endswith('.log') or base in ['state.json', 'progress.json', 'process.json', 'owner.json'] or re.fullmatch(r'[^/]+\.command\.json', name):
            continue
        compared += 1
        if base.endswith('.jsonl'):
            same = [stable(json.loads(l)) for l in a.read_text().splitlines()] == [stable(json.loads(l)) for l in b.read_text().splitlines()]
        elif base.endswith('command.json') or base in ['run.json', 'selection.json']:
            same = stable(json.loads(a.read_text())) == stable(json.loads(b.read_text()))
        elif base == 'source.wav':
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
    parser.add_argument('--out', type=Path, required=True, help='New folder')
    parser.add_argument('--runs', default=','.join(RUNS))
    parser.add_argument('--tracks', type=int, default=3)
    parser.add_argument('--clip-seconds', type=float, default=20)
    parser.add_argument('--window-seconds', type=int, default=8)
    args = parser.parse_args()
    if args.out.exists(): parser.error('Output exists; choose a new folder')
    args.out = args.out.resolve(); args.train_run = args.train_run.resolve()
    preparation.make_dataset(args.dataset, args.out / 'dataset', args.tracks, args.clip_seconds)
    wrapper = args.out / 'python_workflow.py'; wrapper.write_text(preparation.WRAPPER)
    runtime = json.loads(args.runtime.read_text())
    runtime.update(run_root=str(args.out / 'runs'), models_dir=str(args.out / 'models'))
    results = {}
    for run in args.runs.split(','):
        for implementation in ['python', 'native']:
            folder = args.out / 'runs' / run; folder.mkdir(parents=True)
            (folder / 'runtime.json').write_text(json.dumps(runtime))
            (folder / 'request.json').write_text(json.dumps(dict(id=run, name=run, dataset=str(args.out / 'dataset'),
                window_seconds=args.window_seconds, rank=4, ar_steps=6, nar_steps=8, eval_every=2, recipe_version=5, **RUNS[run])))
            train(implementation, folder, args, wrapper)
            folder.rename(args.out / 'runs' / f'{run}-{implementation}')
            print(run, implementation, 'trained', flush=True)
        problems, compared = compare(args.out / 'runs' / f'{run}-python', args.out / 'runs' / f'{run}-native')
        state = json.loads((args.out / 'runs' / f'{run}-native/state.json').read_text())
        results[run] = dict(compared_files=compared, status=state['status'], problems=problems,
                            ar_step=state['ar_selection']['selected']['step'], nar_step=state['nar_selection']['selected']['step'])
        print(run, json.dumps(results[run], indent=2), flush=True)
    (args.out / 'results.json').write_text(json.dumps(results, indent=2))
    return 0 if all(not r['problems'] and r['status'] == 'review' for r in results.values()) else 1


if __name__ == '__main__':
    sys.exit(main())

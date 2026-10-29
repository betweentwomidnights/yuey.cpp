#!/usr/bin/env python3
"""Check that yue2-train-run publishes and releases a run exactly as the Python does.

For each reviewed run under RUNS (named NAME-native, as check_native_fitting.py
leaves them), the run is moved back to the path its state records, published
by publish_training_adapter.publish() and by `yue2-train-run publish` into the
same models folder, and the GGUF and sidecar compared byte for byte, along
with a second publication of an already-published run. Then
Workflow.release_caches() and `yue2-train-run release-caches` each run on a
copy of the run, and the files left behind and the bytes released compared.
Needs no GPU.

python tools/check_native_publication.py --train-run build/bin/Release/yue2-train-run.exe
    --runs outputs/native-fit/runs
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'tools'))
import publish_training_adapter  # noqa: E402
import training_workflow  # noqa: E402


def native(binary, *args):
    result = subprocess.run([str(binary), *map(str, args)], capture_output=True, text=True)
    if result.returncode: raise RuntimeError(result.stderr)
    return json.loads(result.stdout)


def remaining(run):
    return sorted(p.relative_to(run).as_posix() for p in run.rglob('*') if p.is_file() and p.name != 'state.json')


def check(binary, folder, problems):
    name = folder.name[:-len('-native')]
    run = folder.parent / name
    folder.rename(run)
    try:
        runtime = json.loads((run / 'runtime.json').read_text()); request = json.loads((run / 'request.json').read_text())
        state = json.loads((run / 'state.json').read_text())
        models = Path(runtime['models_dir']).resolve(); models.mkdir(parents=True, exist_ok=True)
        target = models / (request['id'] + '-F32-LoRA.gguf'); sidecar = target.with_suffix('.training.json')
        expected = publish_training_adapter.publish(run.resolve(), runtime, request, state)
        python_bytes = target.read_bytes(), sidecar.read_bytes()
        target.unlink(); sidecar.unlink()
        result = native(binary, 'publish', '--run', run)
        if (target.read_bytes(), sidecar.read_bytes()) != python_bytes: problems.append(f'{name}: published bytes differ')
        if result != json.loads(json.dumps(expected)): problems.append(f'{name}: publish result differs')
        if native(binary, 'publish', '--run', run) != result: problems.append(f'{name}: republish differs')
        backup = run.parent / (name + '-backup'); shutil.copytree(run, backup)
        workflow = training_workflow.Workflow(run); workflow.release_caches()
        released = workflow.state['caches_released']['gib'] * 2**30
        python_left = remaining(run)
        shutil.rmtree(run); backup.rename(run)
        native_released = native(binary, 'release-caches', '--run', run)['released_bytes']
        if remaining(run) != python_left: problems.append(f'{name}: released files differ')
        if native_released != round(released): problems.append(f'{name}: released bytes {native_released} != {released}')
        return dict(adapter_bytes=len(python_bytes[0]), released_bytes=native_released, files_left=len(python_left))
    finally:
        run.rename(folder)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--train-run', type=Path, required=True)
    parser.add_argument('--runs', type=Path, required=True)
    args = parser.parse_args()
    problems = []; results = {}
    for folder in sorted(args.runs.glob('*-native')):
        results[folder.name] = check(args.train_run.resolve(), folder.resolve(), problems)
    results['problems'] = problems
    print(json.dumps(results, indent=2))
    return 1 if problems or len(results) < 2 else 0


if __name__ == '__main__':
    sys.exit(main())

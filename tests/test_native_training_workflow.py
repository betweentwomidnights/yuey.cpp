"""yue2-train-run must equal the Python training workflow it replaces.

The Python functions are the reference until the native workflow replaces
them. Values compare with their JSON types (an int is not a float) and
rejections compare by message. Fuzzed cases run in one native process.

YUE2_TRAIN_RUN_BINARY=build/bin/Release/yue2-train-run.exe python -m unittest tests.test_native_training_workflow

With YUE2_TRAINING_PARITY_ROOT set to a folder of earlier work (outputs/),
every prepared run under it is also reviewed, every dataset it names is
inventoried, and every transcription it holds is re-engraved and compared,
including with the normalized.abc preparation wrote.
"""
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

import gguf
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parents[1] / 'tools'))
import ar_dataset  # noqa: E402
import training_provenance  # noqa: E402
import training_workflow as workflow  # noqa: E402

BINARY = os.environ.get('YUE2_TRAIN_RUN_BINARY')
ROOT = os.environ.get('YUE2_TRAINING_PARITY_ROOT')


def native(*args):
    result = subprocess.run([BINARY, *map(str, args)], capture_output=True, text=True, encoding='utf-8')
    if result.returncode:
        raise ValueError(result.stderr.strip().removeprefix('error: '))
    return json.loads(result.stdout)


def same(a, b):
    """Equal values with equal JSON types: 1, 1.0 and True all differ."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(map(same, a, b))
    return a == b


def reference(fn, case):
    """The Python result as JSON would carry it, or ('error', message)."""
    try:
        if fn == 'repr': value = repr(case['value'])
        elif fn == 'caption': value = ar_dataset.caption_metadata(case['text'])
        elif fn == 'strip': value = case['text'].strip()
        elif fn == 'windows': value = ar_dataset.window_starts(case['total'], case['frames'], case['bpm'])
        elif fn == 'chord': value = ar_dataset.chord_name(case['text'])
        elif fn == 'pitch': value = ar_dataset.abc_pitch(case['pitch'])
        elif fn == 'settings': value = workflow.settings(case['request'])
        elif fn == 'score':
            abc, stats = ar_dataset.score_from_events(case['events'], case['seconds'], case['bpm'], case['key'])
            value = dict(abc=abc, stats=stats)
        elif fn == 'estimate':
            value = workflow.ar_resource_estimate(case['eligible'], case['options'], case['device'], case['model_bytes'])
        elif fn == 'conditioning': value = workflow.conditioning_bytes(case['eligible'], case['storage'], case['scope'])
        elif fn == 'allowance': value = workflow.checkpoint_allowance(case['options'], case['experts'])
        elif fn == 'storage':
            value = workflow.storage_options(case['eligible'], case['options'], dict(nar=case['nar'], ar=case['ar']))
        else: raise AssertionError(fn)
    except ValueError as error:
        return 'error', str(error)
    except (KeyError, TypeError, IndexError, AttributeError):
        return 'error', None
    return 'ok', json.loads(json.dumps(value))


class Parity:
    def __init__(self):
        self.cases = []

    def add(self, fn, **case):
        # The native side reads the JSON both sides are given.
        self.cases.append(json.loads(json.dumps(dict(fn=fn, **case))))

    def run(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'cases.json'
            path.write_text(json.dumps(self.cases), encoding='utf-8')
            return native('parity', '--cases', path)

    def mismatches(self):
        failures = []; self.results = self.run(); self.outcomes = {}
        for case, result in zip(self.cases, self.results):
            kind, value = reference(case['fn'], {k: v for k, v in case.items() if k != 'fn'})
            if kind == 'error':
                matched = 'error' in result and (value is None or result['error'] == value)
            else:
                matched = 'ok' in result and same(value, result['ok'])
            key = (case['fn'], kind); self.outcomes[key] = self.outcomes.get(key, 0) + 1
            if not matched:
                failures.append(dict(case=case, python=[kind, value], native=result))
        return failures


CAPTION_PARTS = ['dark electronica', 'neurofunk', 'lofi piano', '88 bpm', '120BPM', '72.5 bpm', '7.5bpm', '400 bpm',
                 '19 bpm', '1200 bpm', 'B♭ major', 'f# minor', 'C major', 'eb MINOR', 'G♯ Minor', 'A  major',
                 'Bb', 'minority report', 'x120 bpm', '120 bpmx', 'Cmajor', 'D\tminor', 'é120 bpm', '　soft　',
                 'ambient.', '', ' ', 'F#m', '12.bpm', '1.2.3 bpm', '90 bpm', 'E major', 'tempo 100 bpm in C major']


def random_caption(rng):
    parts = rng.sample(CAPTION_PARTS, rng.randint(1, 6))
    separators = [', ', ',', ' , ', ',\n', '; ']
    return rng.choice(['', ' ', '\n']) + ''.join(p + rng.choice(separators) for p in parts[:-1]) + parts[-1]


def valid_caption(rng):
    style = rng.sample(['dark electronica', 'neurofunk', 'lofi piano', 'étude', 'ambient.', 'tape hiss', 'x120 bpm'],
                       rng.randint(1, 3))
    tempo = rng.choice(['88 bpm', '120BPM', '72.5 bpm', '20 bpm', '400 bpm', '96 bpm', f'{rng.uniform(20, 400):.3f} bpm'])
    key = rng.choice(['B♭ major', 'f# minor', 'C major', 'eb MINOR', 'G♯ Minor', 'A  major', 'D\tminor'])
    parts = style + [tempo, key]; rng.shuffle(parts)
    return rng.choice([', ', ',', ' , ', ',\n']).join(parts)


SETTING_VALUES = dict(
    window_seconds=[30, 2, 60, 1, 61, 30.0, 30.5, '30', True, None], rank=[32, 2, 1, 128, 129, 16.0, 'x'],
    ar_steps=[800, 1, 0, 10000, 10001], nar_steps=[400, 1, 10000.0], ar_lr=[2e-5, 1e-7, 1e-8, .001, .0011, 1, 0, 'a'],
    nar_lr=[3e-5, 5e-4, 1e-3], kl_weight=[.2, 0, 5, 5.0001, -0.1, 1], seed=[831001, 0, 4294967295, 4294967296, -1, 7.0],
    eval_every=[100, 1, 1000, 1001], family=['lora', 'dora', 'LoRA', 1, None],
    trigger=['koan_style', 'a-b', 'two words', '', 'x' * 81, 'x' * 80, 5],
    keep_caches=[True, False, 0, 'true'], recipe_version=[1, 2, 3, 4, 5, 6, 0, 5.0, '5', True],
    evaluation_scope=['heldout', 'full', 'none', None], teacher_storage=['cache', 'online', 'f16', None],
    conditioning_storage=['cache', 'rolling', 'resident', 'zip'])


def random_request(rng, valid=False):
    request = dict(id=rng.choice(['koan', 'run-1', 'a_b']))
    for key, values in SETTING_VALUES.items():
        if rng.random() < (0.3 if valid else 0.4):
            request[key] = values[0] if valid else rng.choice(values)
    if valid:
        request['recipe_version'] = rng.choice([1, 2, 3, 4, 5])
        if rng.random() < .3: request['kl_weight'] = rng.choice([0, 0.0, 0.5])
        if rng.random() < .3: request['rank'] = rng.choice([2, 16, 64, 128])
        if rng.random() < .3: request['teacher_storage'] = 'online'; request['recipe_version'] = 5
        if rng.random() < .3: request['conditioning_storage'] = rng.choice(['rolling', 'resident']); request['recipe_version'] = 5
        if request['recipe_version'] < 5:
            request.pop('teacher_storage', None); request.pop('conditioning_storage', None)
    return request


def random_rows(rng):
    return [dict(split=rng.choice(['train', 'train', 'validation']), frames=rng.choice([750, 500, 1500, 1, rng.randint(1, 4000)]),
                 prefix_tokens=rng.randint(0, 3000)) for _ in range(rng.randint(0, 60))]


def random_events(rng, seconds):
    """Mostly valid transcriptions; one in ten may hold an invalid note."""
    events = []; invalid = rng.random() < .1
    chords = ['C:maj', 'A:min7', 'E:maj/3', 'B:min7/b7', 'G:min/2', 'F#:hdim7', 'Bb:sus4(b7)', 'N', '', 'C', 'D:weird', None]
    for _ in range(rng.randint(0, 60)):
        start = rng.choice([rng.uniform(-0.5, seconds + 0.5), round(rng.uniform(0, seconds), 2), rng.uniform(0, seconds), 0.0])
        event = dict(time_seconds=start, notes=[])
        if rng.random() < .6: event['chord'] = rng.choice(chords)
        if rng.random() < .03: event.update(meter_numerator=rng.choice([3, 4, 6, 7]), meter_denominator=rng.choice([4, 8]))
        for _ in range(rng.randint(0, 3)):
            end = start + rng.choice([rng.uniform(0, 2), 0.0, 1 / 48, round(rng.uniform(0, 1), 3)])
            note = dict(pitch=rng.randint(30, 100), end_time_seconds=end, track=rng.choice([0, 1]))
            if invalid and rng.random() < .1:
                note.update(rng.choice([dict(pitch=128), dict(track=2), dict(end_time_seconds=start - .01), dict(pitch=-1)]))
            event['notes'].append(note)
        events.append(event)
    return events


def write_audio(path, seconds, rate=48000, channels=2, seed=0, subtype='FLOAT'):
    noise = np.random.default_rng(seed).standard_normal((int(seconds * rate), channels)) * .01
    sf.write(path, noise, rate, subtype=subtype)


def write_base(path):
    """A tiny GGUF with tensors named like each expert's."""
    writer = gguf.GGUFWriter(str(path), 'yue2')
    for name, count in [('model.embed_tokens.weight', 7), ('model.norm.weight', 3), ('model.layers.0.self_attn.q_proj.weight', 11),
                        ('model.layers.0.nar_mlp.up_proj.weight', 13), ('latent_pos_embed.pe', 5), ('vae2llm.weight', 2),
                        ('llm2vae.bias', 4), ('time_embedder.linear.weight', 6), ('lm_head.weight', 9)]:
        writer.add_tensor(name, np.arange(count, dtype=np.float32))
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()


def python_review(run):
    """What Workflow.prepare() records for dataset review, minus free disk."""
    request = json.loads((run / 'request.json').read_text()); runtime = json.loads((run / 'runtime.json').read_text())
    o = workflow.settings(request); models = runtime['models']
    data = json.loads((run / 'data/dataset.json').read_text())
    eligible = [r for r in data['rows'] if r.get('eligible', True)]
    weights = sum(Path(models[k]).stat().st_size for k in ['base', 'companion'])
    estimate = workflow.ar_resource_estimate(eligible, o, runtime.get('device', 'cuda'), weights)
    storage = o.get('conditioning_storage', 'cache'); scope = o.get('evaluation_scope', 'full')
    conditioning = workflow.conditioning_bytes(eligible, storage, scope)
    peak = max(estimate['estimated_teacher_disk_gib'] * 2**30, conditioning) + workflow.checkpoint_allowance(o)
    return json.loads(json.dumps(dict(
        examples=len(eligible), excluded=len(data['rows']) - len(eligible),
        train_examples=sum(r['split'] == 'train' for r in eligible), validation_examples=sum(r['split'] == 'validation' for r in eligible),
        estimated_conditioning_gib=conditioning / 2**30, estimated_peak_disk_gib=peak / 2**30,
        storage_options=workflow.storage_options(eligible, o, workflow.expert_bytes(models['base'])), **estimate)))


def audio_subtype(index):
    return ['FLOAT', 'PCM_16', 'PCM_24', 'PCM_32', 'DOUBLE', 'FLOAT'][index % 6]


@unittest.skipUnless(BINARY, 'Set YUE2_TRAIN_RUN_BINARY to built yue2-train-run')
class NativeWorkflowParity(unittest.TestCase):
    def assertParity(self, parity, accepted=()):
        failures = parity.mismatches()
        self.assertFalse(failures, json.dumps(failures[:5], indent=1, ensure_ascii=False))
        # Most cases must take the success path, not only rejections.
        for fn in accepted:
            ok, rejected = parity.outcomes.get((fn, 'ok'), 0), parity.outcomes.get((fn, 'error'), 0)
            self.assertGreater(ok, rejected / 2, (fn, ok, rejected))

    def test_pure_functions_match_on_fuzzed_cases(self):
        rng = random.Random(20261028); parity = Parity()
        for _ in range(3000): parity.add('caption', text=random_caption(rng))
        for _ in range(3000): parity.add('caption', text=valid_caption(rng))
        for text in CAPTION_PARTS: parity.add('strip', text=text); parity.add('caption', text=text + ', piano')
        for _ in range(2000):
            frames = rng.choice([750, 500, 100, rng.randint(1, 2000)])
            parity.add('windows', total=rng.choice([frames - 1, frames, rng.randint(frames, 20000)]), frames=frames,
                       bpm=rng.choice([88.0, 120.0, 72.5, rng.uniform(20, 400), 174.0]))
        roots = ['C', 'C#', 'Db', 'E', 'Bb', 'F#', 'X', '']
        kinds = ['maj', 'min', 'dim', 'aug', '7', 'maj7', 'min7', 'dim7', 'hdim7', 'sus4', 'sus2', 'maj6', 'min6', 'sus4(b7)',
                 'minmaj7', 'add9']
        for root in roots:
            for kind in kinds:
                for inversion in ['', '/2', '/b3', '/3', '/5', '/b7', '/7', '/9']: parity.add('chord', text=f'{root}:{kind}{inversion}')
        for text in ['N', '', 'C', 'G7', 'C:maj/']: parity.add('chord', text=text)
        for pitch in range(-1, 129): parity.add('pitch', pitch=pitch)
        for _ in range(3000): parity.add('settings', request=random_request(rng))
        for _ in range(3000): parity.add('settings', request=random_request(rng, valid=True))
        for value in [30.0, 2e-5, 3e-5, .2, 1e16, 1e15, 1e-4, 1e-5, 0.1 + 0.2, -0.0, 5e-324, 1.7976931348623157e308, 123456789.125]:
            parity.add('repr', value=value)
        for _ in range(2000): parity.add('repr', value=rng.choice([rng.uniform(-1e6, 1e6), 10 ** rng.uniform(-30, 30), rng.random()]))
        self.assertParity(parity, accepted=['caption', 'settings', 'windows', 'chord', 'pitch', 'repr'])

    def test_scores_match_on_fuzzed_events(self):
        rng = random.Random(831001); parity = Parity()
        for _ in range(3000):
            seconds = rng.choice([30, 4, 30.0, 3.2, rng.uniform(1, 60)])
            parity.add('score', events=random_events(rng, seconds), seconds=seconds,
                       bpm=rng.choice([88.0, 120.0, 150.0, rng.uniform(20, 400)]),
                       key=rng.choice(['C', 'Bb', 'F#m', 'Dm', 'C', 'H', 'cm']))
        self.assertParity(parity, accepted=['score'])

    def test_estimates_match_on_fuzzed_datasets(self):
        rng = random.Random(7); parity = Parity()
        for _ in range(1500):
            rows = random_rows(rng); options = workflow.settings(random_request(rng, valid=True))
            parity.add('estimate', eligible=rows, options=options, device=rng.choice(['cuda', 'cpu']),
                       model_bytes=rng.choice([0, 7_900_000_000, rng.randint(0, 10**10)]))
            for storage in ['cache', 'rolling', 'resident']:
                parity.add('conditioning', eligible=rows, storage=storage, scope=rng.choice(['heldout', 'full']))
            parity.add('allowance', options=options, experts=rng.choice([1, 2]))
            parity.add('storage', eligible=rows, options=options, nar=rng.randint(0, 3 * 2**30), ar=rng.randint(0, 4 * 2**30))
        self.assertParity(parity, accepted=['estimate', 'conditioning', 'allowance', 'storage'])

    def test_inventory_matches_on_a_mixed_dataset(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            captions = {'b-Track.wav': '﻿neurofunk, 88 bpm, B♭ major\r\n', 'A_track.flac': 'lofi piano\r\n72.5 BPM\r\nf# minor',
                        'c.WAV': 'ambient,   120bpm , C major  ', 'déjà vu.wav': 'jazz, 96 bpm, Eb minor',
                        'e.flac': 'house, 124 bpm, A minor', '_under.wav': 'dub, 70 bpm, G major'}
            for index, (name, caption) in enumerate(captions.items()):
                subtype = 'PCM_16' if name.endswith('.flac') else audio_subtype(index)
                write_audio(folder / name, 6 + index * .37, rate=[48000, 44100, 32000][index % 3], channels=1 + index % 2,
                            seed=index, subtype=subtype)
                (folder / name).with_suffix('.txt').write_bytes(caption.encode('utf-8'))
            (folder / 'notes.md').write_text('ignored')
            for seed in [831001, 1, 2**32 - 1]:
                expected = json.loads(json.dumps(workflow.inventory(folder, 5, seed)))
                self.assertTrue(same(expected, native('inventory', '--dataset', folder, '--window-seconds', 5, '--seed', seed)))
            for setup, message in [(lambda f: (f / 'c.txt').unlink(), 'Missing caption: c.txt'),
                                   (lambda f: (f / 'c.txt').write_text('ambient, C major'), 'Caption needs explicit tempo and key'),
                                   (lambda f: (f / 'c.txt').write_text('120 bpm, C major'), 'Caption has no style description: c.WAV'),
                                   (lambda f: (f / 'e.flac').write_bytes((f / 'A_track.flac').read_bytes()), 'Duplicate recording: e.flac')]:
                with self.subTest(message=message), tempfile.TemporaryDirectory() as copy:
                    copy = Path(copy)
                    for path in folder.iterdir(): (copy / path.name).write_bytes(path.read_bytes())
                    setup(copy)
                    with self.assertRaisesRegex(ValueError, message): workflow.inventory(copy, 5, 1)
                    with self.assertRaisesRegex(ValueError, message): native('inventory', '--dataset', copy, '--window-seconds', 5, '--seed', 1)
            with self.assertRaisesRegex(ValueError, 'Recording shorter than training window: A_track.flac'):
                native('inventory', '--dataset', folder, '--window-seconds', 7, '--seed', 1)

    def test_review_matches_a_prepared_run(self):
        rng = random.Random(3)
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder); write_base(folder / 'base.gguf'); (folder / 'companion.gguf').write_bytes(b'\0' * 12345)
            for index in range(20):
                run = folder / f'run-{index}'; (run / 'data').mkdir(parents=True)
                request = random_request(rng, valid=True); (run / 'request.json').write_text(json.dumps(request))
                (run / 'runtime.json').write_text(json.dumps(dict(device=rng.choice(['cuda', 'cpu']),
                    models=dict(base=str(folder / 'base.gguf'), companion=str(folder / 'companion.gguf')))))
                rows = random_rows(rng)
                for row in rows:
                    if rng.random() < .2: row['eligible'] = False
                (run / 'data/dataset.json').write_text(json.dumps(dict(rows=rows)))
                self.assertTrue(same(python_review(run), native('review', '--run', run)), run)

    def test_capabilities_match(self):
        self.assertTrue(same(json.loads(json.dumps(workflow.capabilities())), native('capabilities')))
        with tempfile.TemporaryDirectory() as folder:
            runtime = Path(folder) / 'training-runtime.json'
            runtime.write_text(json.dumps(dict(bin=str(Path(BINARY).resolve().parent))))
            self.assertTrue(same(json.loads(json.dumps(workflow.capabilities(runtime))), native('capabilities', '--runtime', runtime)))

    def test_runtime_identity_matches(self):
        expected = json.loads(json.dumps(training_provenance.native_runtime_identity(BINARY)))
        self.assertTrue(same(expected, native('identity', '--executable', BINARY)))
        with tempfile.TemporaryDirectory() as folder:
            files = [Path(folder) / name for name in ['a.bin', 'empty', 'B.txt']]
            for index, path in enumerate(files): path.write_bytes(os.urandom(index * 1_000_003))
            self.assertEqual(native('sha256', *files), {str(p): training_provenance.digest(p) for p in files})

    @unittest.skipUnless(ROOT, 'Set YUE2_TRAINING_PARITY_ROOT to compare earlier runs')
    def test_earlier_runs_match(self):
        root = Path(ROOT); runs = sorted(p.parent for p in root.glob('*/runs/*/request.json'))
        datasets = {}
        for run in runs:
            with self.subTest(review=str(run)):
                if not (run / 'data/dataset.json').exists() or not (run / 'runtime.json').exists(): continue
                models = json.loads((run / 'runtime.json').read_text())['models']
                if not all(Path(models[k]).exists() for k in ['base', 'companion']): continue
                self.assertTrue(same(python_review(run), native('review', '--run', run)))
            request = json.loads((run / 'request.json').read_text()); o = workflow.settings(request)
            datasets[(request['dataset'], o['window_seconds'], o['seed'])] = run
        for (dataset, seconds, seed), run in sorted(datasets.items()):
            with self.subTest(inventory=dataset):
                if not Path(dataset).is_dir(): continue
                expected = json.loads(json.dumps(workflow.inventory(dataset, seconds, seed)))
                self.assertTrue(same(expected, native('inventory', '--dataset', dataset, '--window-seconds', seconds, '--seed', seed)))
        parity = Parity(); folders = []
        for record_path in sorted(root.glob('**/examples/*/example.json')):
            folder = record_path.parent; events = folder / 'events.json'
            if not events.exists(): continue
            record = json.loads(record_path.read_text())
            # Preparation passes its integer window_seconds.
            seconds = record['frames'] // 25 if record['frames'] % 25 == 0 else record['frames'] / 25
            parity.add('score', events=json.loads(events.read_text())['events'], seconds=seconds, bpm=record['bpm'], key=record['key'])
            folders.append(folder)
        self.assertGreater(len(parity.cases), 0)
        self.assertParity(parity)
        # Scores written before chord inversions were engraved differ from
        # both implementations; count them rather than fail.
        written = [(folder / 'normalized.abc', result) for folder, result in zip(folders, parity.results)
                   if (folder / 'normalized.abc').exists() and 'ok' in result]
        current = sum(result['ok']['abc'] == path.read_text() for path, result in written)
        print(f'{len(runs)} runs, {len(datasets)} datasets, {len(folders)} transcriptions; '
              f'{current} of {len(written)} written scores equal today\'s engraving')


if __name__ == '__main__':
    unittest.main()

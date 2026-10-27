#!/usr/bin/env python3
"""Bind scored audio targets to one frozen AR adapter for native NAR fitting.

This consumes completed AR preparation and CPU VAE targets. It does not choose
an AR checkpoint or start training. Frozen K/V caches are separate for each AR
and numerical runtime; source recordings remain read-only.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from prepare_ar_dataset import save, wait_for_inference
from yue2_training_audit import digest, native_runtime_identity


def match_targets(rows, targets):
    """Require a one-to-one join; never substitute a nearby offset/other split."""
    def key(row): return row['source_sha256'], row['start_frame'], row['frames']
    by_key = {}
    for row in targets:
        if key(row) in by_key: raise ValueError('Duplicate VAE target core')
        by_key[key(row)] = row
    result = []
    for row in rows:
        target = by_key.pop(key(row), None)
        if target is None or target['split'] != row['split']:
            raise ValueError('VAE target missing or split differs')
        result.append((row, target))
    if by_key: raise ValueError('Unexpected VAE target cores')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True, help='Completed AR job JSON')
    parser.add_argument('--targets', type=Path, required=True, help='Completed real VAE index.json')
    parser.add_argument('--ar-adapter', type=Path, required=True, help='Listening-selected frozen AR')
    parser.add_argument('--binary', type=Path, required=True, help='Native yue2-train')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--conditioning-device', choices=['cpu','cuda'], default='cpu')
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--auxiliary-dataset', type=Path,
                        help='Explicit generated replay/retention examples, with source provenance')
    parser.add_argument('--auxiliary-targets', type=Path,
                        help='Corresponding completed VAE index')
    args = parser.parse_args()
    if args.threads < 1: parser.error('threads must be positive')
    if bool(args.auxiliary_dataset) != bool(args.auxiliary_targets):
        parser.error('auxiliary dataset and target index must be supplied together')
    os.environ['GGML_CUDA_TF32'] = '0'
    config = json.loads(args.config.read_text())
    dataset_path = Path(config['training_dataset'])
    data = json.loads(dataset_path.read_text())
    targets = json.loads(args.targets.read_text())
    if not data['completed'] or not targets['completed']:
        raise ValueError('AR preparation and VAE targets must be complete')
    rows = [r for r in data['rows'] if r.get('eligible',True) and r.get('kind','real')=='real']
    if not rows: raise ValueError('Dataset has no eligible real examples')
    pairs = match_targets(rows, targets['rows'])
    auxiliary_identity = None
    if args.auxiliary_dataset:
        auxiliary = json.loads(args.auxiliary_dataset.read_text())
        auxiliary_targets = json.loads(args.auxiliary_targets.read_text())
        if not auxiliary['completed'] or not auxiliary_targets['completed']:
            raise ValueError('Auxiliary preparation is incomplete')
        for key in ['vae_sha256','encoder_source_sha256','torch_version','posterior','precision','halo_frames']:
            if targets['identity'][key] != auxiliary_targets['identity'][key]:
                raise ValueError('Auxiliary VAE numerical identity differs')
        extra = [r for r in auxiliary['rows'] if r.get('eligible',True)]
        if not extra or any(r.get('kind','real')=='real' or not r.get('provenance') for r in extra):
            raise ValueError('Auxiliary examples need explicit generated-target provenance')
        if any(r['frames'] != data['frames'] for r in extra):
            raise ValueError('Auxiliary frame count differs')
        pairs += match_targets(extra, auxiliary_targets['rows'])
        keys=[(r['source_sha256'],r['start_frame'],r['frames']) for r,_ in pairs]
        if len(set(keys)) != len(keys):raise ValueError('Duplicate real/auxiliary target core')
        auxiliary_identity = dict(dataset_sha256=digest(args.auxiliary_dataset),
             targets_sha256=digest(args.auxiliary_targets),vae_identity=auxiliary_targets['identity'])
    args.out.mkdir(parents=True, exist_ok=True)
    base = Path(config['models']['base']).resolve()
    common = Path(config['models']['companion']).resolve()
    adapter = args.ar_adapter.resolve()
    binary = args.binary.resolve()
    identity = dict(format='yue2-nar-cache-job-v1', ar_dataset_sha256=digest(dataset_path),
                    target_index_sha256=digest(args.targets), vae_identity=targets['identity'],
                    base_sha256=digest(base), companion_sha256=digest(common),
                    ar_adapter_sha256=digest(adapter), runtime=native_runtime_identity(binary),
                    device=args.conditioning_device, threads=args.threads, tf32=False,
                    preparation_source_sha256=digest(Path(__file__)),
                    prefix_policy='Exact AR training prefix and semantic IDs',
                    replay_policy='Real targets plus explicitly sourced generated replay/retention' if auxiliary_identity else 'Real targets only',
                    auxiliary=auxiliary_identity)
    job_path = args.out/'identity.json'
    if job_path.exists() and json.loads(job_path.read_text()) != identity:
        raise ValueError('NAR cache identity changed; use another output directory')
    save(job_path, identity)
    prepared = []
    for index, (row, target) in enumerate(pairs):
        for path, expected in [(Path(row['prefix']),row['prefix_sha256']),
                               (Path(row['codes']),row['codes_sha256']),
                               (Path(target['means']),target['means_sha256'])]:
            if digest(path) != expected: raise ValueError('NAR input bytes changed')
        if Path(target['means']).stat().st_size != row['frames']*64*4:
            raise ValueError('VAE target frame count differs')
        folder = args.out/(row['source_sha256'][:12]+f"-{row['start_frame']:07d}-{row['frames']}")
        folder.mkdir(exist_ok=True)
        cache = folder/'sample.gguf'; metadata = folder/'cache.json'
        expected = dict(job=identity,prefix_sha256=row['prefix_sha256'],codes_sha256=row['codes_sha256'],
                        means_sha256=target['means_sha256'],frames=row['frames'])
        save(args.out/'progress.json',dict(status='running',stage='frozen-ar-conditioning',
             current=index,total=len(pairs),track=row['track'],start_frame=row['start_frame']))
        if cache.exists():
            previous = json.loads(metadata.read_text())
            if previous['identity'] != expected or digest(cache) != previous['cache_sha256']:
                raise ValueError('Frozen NAR cache changed')
        else:
            if args.conditioning_device=='cuda': wait_for_inference(config.get('inference_urls',[]))
            command = list(map(str,[binary,'--prepare','--model',base,'--prefix',row['prefix'],
                '--tokens',row['codes'],'--latents',target['means'],'--cache-out',cache,
                '--conditioning-lora',common,'--conditioning-lora',adapter,
                '--device',args.conditioning_device,'--threads',args.threads]))
            log = folder/'prepare.log'; start=time.monotonic()
            with log.open('w') as stream:
                result=subprocess.run(command,cwd=config['workspace'],stdout=stream,stderr=subprocess.STDOUT)
            save(folder/'command.json',dict(command=command,returncode=result.returncode,seconds=time.monotonic()-start))
            if result.returncode: raise RuntimeError(log.read_text()[-4000:])
            save(metadata,dict(identity=expected,cache_sha256=digest(cache)))
        prepared.append(dict(row,cache=str(cache.resolve()),cache_sha256=digest(cache),
                             means=target['means'],means_sha256=target['means_sha256']))
        # A partial manifest is never eligible for native fitting.
        save(args.out/'dataset.json',dict(format='yue2-nar-prepared-dataset-v1',completed=False,
             frames=data['frames'],identity=identity,rows=prepared))
        print('CACHE',index+1,len(pairs),row['track'],row['start_frame'],flush=True)
    # Recheck every cache before marking the dataset ready for the trainer.
    if digest(dataset_path)!=identity['ar_dataset_sha256'] or digest(args.targets)!=identity['target_index_sha256']:
        raise ValueError('Input manifest changed during preparation')
    if auxiliary_identity and (digest(args.auxiliary_dataset)!=auxiliary_identity['dataset_sha256'] or
                               digest(args.auxiliary_targets)!=auxiliary_identity['targets_sha256']):
        raise ValueError('Auxiliary manifest changed during preparation')
    for row in prepared:
        for key in ['cache','prefix','codes','means']:
            if digest(Path(row[key])) != row[key+'_sha256']:
                raise ValueError('Input or cache changed during preparation')
    save(args.out/'dataset.json',dict(format='yue2-nar-prepared-dataset-v1',completed=True,
         frames=data['frames'],identity=identity,rows=prepared))
    save(args.out/'progress.json',dict(status='completed',stage='frozen-ar-conditioning',current=len(pairs),total=len(pairs)))


if __name__=='__main__':main()

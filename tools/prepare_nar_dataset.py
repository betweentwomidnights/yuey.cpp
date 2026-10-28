#!/usr/bin/env python3
"""Bind scored audio targets to one frozen AR adapter for native NAR fitting.

This consumes completed AR preparation and CPU VAE targets. It does not choose
an AR checkpoint or start training. Frozen K/V caches are separate for each AR
and numerical runtime; source recordings remain read-only.
"""
import argparse
import hashlib
import gguf
import numpy as np
import json
import os
from pathlib import Path
import subprocess
import time

from prepare_ar_dataset import save, wait_for_inference
from training_provenance import digest, native_runtime_identity


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


def verify_cache(cache, key, row, target, base, common, adapter, kv_type):
    """Check native output provenance and actual aligned input payloads."""
    reader=gguf.GGUFReader(str(cache))
    def field(name):
        value=reader.get_field(name)
        if value is None:raise ValueError('Cache metadata missing: '+name)
        return value.contents()
    if field('general.architecture')!='yue2_nar_sample' or field('yue2.training.cache.identity')!=key:
        raise ValueError('Frozen NAR cache identity differs')
    if Path(field('yue2.training.model')).resolve()!=base.resolve():raise ValueError('Cache base differs')
    expected=''.join(str(p.resolve())+'=1.000000\n' for p in [common,adapter])
    if field('yue2.training.conditioning_adapters')!=expected or field('yue2.training.fixed_adapters')!='':
        raise ValueError('Cache frozen adapters differ')
    payload={t.name:t for t in reader.tensors}
    if set(payload)!={'prefix','codes','clean','keys','values'}:raise ValueError('Unexpected cache payload')
    for name,path in [('prefix',row['prefix']),('codes',row['codes']),('clean',target['means'])]:
        expected_type=gguf.GGMLQuantizationType.F32 if name=='clean' else gguf.GGMLQuantizationType.I32
        if payload[name].tensor_type!=expected_type or payload[name].data.tobytes()!=Path(path).read_bytes():
            raise ValueError('Cache input payload differs: '+name)
    tokens=Path(row['prefix']).stat().st_size//4+row['frames']+1
    if field('yue2.training.ar_tokens')!=tokens:raise ValueError('Cache conditioning length differs')
    for name in ['keys','values']:
        tensor=payload[name];kind=gguf.GGMLQuantizationType.F16 if kv_type=='f16' else gguf.GGMLQuantizationType.F32
        if tensor.tensor_type!=kind or list(tensor.shape)!=[128,tokens,8,28]:raise ValueError('Cache conditioning shape/type differs')
        values=tensor.data.view(np.float16) if kv_type=='f16' else tensor.data
        if not np.isfinite(values).all():raise ValueError('Nonfinite conditioning')


def batch_command(binary, plan, base, common, adapter, device, threads, kv_type, batch, stop_file=None):
    command=list(map(str,[binary,'--prepare-batch',plan,'--conditioning-only','--model',base,
        '--conditioning-lora',common,'--conditioning-lora',adapter,'--device',device,
        '--threads',threads,'--cache-kv-type',kv_type,'--conditioning-batch',batch]))
    if stop_file:command+=['--stop-file',str(stop_file)]
    return command


def run_batch(command, workspace, log_path, on_event):
    """Run native batch preparation, handing each completed cache to on_event."""
    with log_path.open('a',encoding='utf-8') as log:
        process=subprocess.Popen(command,cwd=workspace,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
        try:
            for line in process.stdout:
                log.write(line);log.flush()
                if line.startswith('CACHE_EVENT '):
                    on_event(json.loads(line[len('CACHE_EVENT '):]))
                    print(line.rstrip(),flush=True)
        except BaseException:
            # Drain the pipe before waiting: an unwatched child can block
            # on stdout and never reach its next stop-file check.
            for line in process.stdout:log.write(line)
            raise
        finally:process.wait()
    if process.returncode:raise RuntimeError(log_path.read_text()[-4000:])


def native_facts(event):
    """The recorded cache facts a rolling NAR fit binds its identity to."""
    return dict(cache_bytes=event['bytes'],cache_fnv=event['fnv'],cache_tokens=event['tokens'],
                cache_frames=event['frames'],cache_half=event['half'])


def materialize(manifest, rows, binary, base, common, adapter, stop_file=None):
    """Regenerate absent rolling caches. Each must be byte-identical to the one
    recorded by the preparation pass; any difference fails the run."""
    identity=json.loads(Path(manifest).read_text())['identity']
    pending=[r for r in rows if not Path(r['cache']).exists()]
    if not pending:return 0
    out=Path(manifest).parent;plan=out/'materialize.json'
    save(plan,[dict(prefix=r['prefix'],tokens=r['codes'],latents=r['means'],
                    **{'cache-out':r['cache'],'cache-identity':r['cache_identity']}) for r in pending])
    by_path={str(Path(r['cache']).resolve()):r for r in pending}
    def check(event):
        row=by_path[str(Path(event['path']).resolve())]
        if native_facts(event)!={k:row[k] for k in native_facts(event)} or digest(Path(row['cache']))!=row['cache_sha256']:
            Path(row['cache']).unlink(missing_ok=True)
            raise ValueError('Regenerated NAR cache differs from its recorded bytes: '+row['cache'])
    run_batch(batch_command(binary,plan,base,common,adapter,identity['device'],identity['threads'],
                            identity['kv_type'],identity['prefill_batch'],stop_file),
              Path(binary).parent,out/'materialize.log',check)
    missing=[r['cache'] for r in pending if not Path(r['cache']).exists()]
    if missing and not (stop_file and Path(stop_file).exists()):raise ValueError('Native preparation did not regenerate '+missing[0])
    return len(pending)-len(missing)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True, help='Completed AR job JSON')
    parser.add_argument('--targets', type=Path, required=True, help='Completed real VAE index.json')
    parser.add_argument('--ar-adapter', type=Path, required=True, help='Listening-selected frozen AR')
    parser.add_argument('--binary', type=Path, required=True, help='Native yue2-train')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--conditioning-device', choices=['cpu','cuda'], default='cuda')
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--cache-kv-type', choices=['f16','f32'], default='f16')
    parser.add_argument('--conditioning-batch', type=int, default=0)
    parser.add_argument('--stop-file', type=Path)
    parser.add_argument('--storage', choices=['cache','rolling','resident'], default='cache',
                        help='cache keeps every K/V file; rolling records each one, then deletes it '
                             'for later byte-identical regeneration; resident writes none')
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
        for key in ['vae_sha256','encoder_source_sha256','torch_version','posterior','precision','halo_frames',
                    'native_runtime','device','threads','core_frames','encoder_halo_frames','resampler','precision_environment']:
            if targets['identity'].get(key) != auxiliary_targets['identity'].get(key):
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
                    conditioning_only=True, kv_type=args.cache_kv_type, prefill_batch=args.conditioning_batch, flash_attention=False,
                    preparation_source_sha256=digest(Path(__file__)),
                    prefix_policy='Exact AR training prefix and semantic IDs',
                    replay_policy='Real targets plus explicitly sourced generated replay/retention' if auxiliary_identity else 'Real targets only',
                    auxiliary=auxiliary_identity)
    if args.storage!='cache':identity['storage']=args.storage
    if args.storage!='cache' and args.cache_kv_type!='f16':raise ValueError('Rolling and resident conditioning use F16 K/V')
    job_path = args.out/'identity.json'
    if job_path.exists() and json.loads(job_path.read_text()) != identity:
        raise ValueError('NAR cache identity changed; use another output directory')
    save(job_path, identity)
    prepared = []; pending = []; entries = []
    for index, (row, target) in enumerate(pairs):
        for path, expected_hash in [(Path(row['prefix']),row['prefix_sha256']),
                               (Path(row['codes']),row['codes_sha256']),
                               (Path(target['means']),target['means_sha256'])]:
            if digest(path) != expected_hash: raise ValueError('NAR input bytes changed')
        if Path(target['means']).stat().st_size != row['frames']*64*4:
            raise ValueError('VAE target frame count differs')
        folder = args.out/(row['source_sha256'][:12]+f"-{row['start_frame']:07d}-{row['frames']}")
        folder.mkdir(exist_ok=True)
        cache = folder/'sample.gguf'; metadata = folder/'cache.json'
        expected = dict(job=identity,prefix_sha256=row['prefix_sha256'],codes_sha256=row['codes_sha256'],
                        means_sha256=target['means_sha256'],frames=row['frames'])
        key = hashlib.sha256(json.dumps(expected,sort_keys=True).encode()).hexdigest()
        entries.append((row,target,cache,metadata,expected,key))
        if args.storage=='resident':continue
        if args.storage=='rolling' and not cache.exists() and metadata.exists():
            previous=json.loads(metadata.read_text())
            if previous.get('identity')!=expected or 'native' not in previous:raise ValueError('Rolling NAR cache record changed')
            continue
        if not cache.exists():
            pending.append(dict(prefix=row['prefix'],tokens=row['codes'],latents=target['means'],
                                **{'cache-out':str(cache.resolve()),'cache-identity':key}))
        else:
            verify_cache(cache,key,row,target,base,common,adapter,args.cache_kv_type)
            if metadata.exists():
                previous=json.loads(metadata.read_text())
                if previous['identity']!=expected or digest(cache)!=previous['cache_sha256']:
                    raise ValueError('Frozen NAR cache changed')
            else:
                # After a coordinator crash there is no trusted payload checksum.
                # Preserve the unregistered file and recompute instead of blessing
                # potentially corrupted K/V bytes with a newly invented checksum.
                number=1
                while (folder/f'sample.unregistered-{number}.gguf').exists():number+=1
                quarantine=folder/f'sample.unregistered-{number}.gguf'
                cache.rename(quarantine)
                save(quarantine.with_suffix('.json'),dict(reason='Missing committed cache checksum',identity=expected,sha256=digest(quarantine)))
                pending.append(dict(prefix=row['prefix'],tokens=row['codes'],latents=target['means'],
                                    **{'cache-out':str(cache.resolve()),'cache-identity':key}))
    if pending:
        if args.conditioning_device=='cuda':wait_for_inference(config.get('inference_urls',[]))
        plan=args.out/'pending.json';save(plan,pending)
        command=batch_command(binary,plan,base,common,adapter,args.conditioning_device,args.threads,
                              args.cache_kv_type,args.conditioning_batch,args.stop_file)
        start=time.monotonic();completed=len(entries)-len(pending)
        def committed(event):
            nonlocal completed
            item=pending[event['index']]
            entry=next(e for e in entries if e[2].resolve()==Path(item['cache-out']).resolve())
            row,target,cache,metadata,expected,key=entry
            verify_cache(cache,key,row,target,base,common,adapter,args.cache_kv_type)
            record=dict(identity=expected,cache_sha256=digest(cache))
            if args.storage=='rolling':
                # Keep only what proves a later regeneration is byte-identical.
                record['native']=native_facts(event);save(metadata,record);cache.unlink()
            else:save(metadata,record)
            completed+=1
            save(args.out/'progress.json',dict(status='running',stage='frozen-ar-conditioning',current=completed,total=len(entries)))
        try:run_batch(command,config['workspace'],args.out/'prepare.log',committed)
        finally:save(args.out/'command.json',dict(command=command,seconds=time.monotonic()-start))
    for row,target,cache,metadata,expected,key in entries:
        if args.storage=='resident':
            # No cache file exists; the path only names the window for the fitter.
            prepared.append(dict(row,cache=str(cache.resolve()),means=target['means'],means_sha256=target['means_sha256']))
            continue
        if args.storage=='rolling':
            if not metadata.exists():
                if args.stop_file and args.stop_file.exists():
                    save(args.out/'progress.json',dict(status='stopped',stage='frozen-ar-conditioning',current=len(prepared),total=len(entries)))
                    return
                raise ValueError('Native preparation did not record all caches')
            record=json.loads(metadata.read_text())
            if record.get('identity')!=expected or 'native' not in record:raise ValueError('Rolling NAR cache record changed')
            prepared.append(dict(row,cache=str(cache.resolve()),cache_sha256=record['cache_sha256'],cache_identity=key,
                                 means=target['means'],means_sha256=target['means_sha256'],**record['native']))
            continue
        if not cache.exists():
            if args.stop_file and args.stop_file.exists():
                save(args.out/'progress.json',dict(status='stopped',stage='frozen-ar-conditioning',current=len(prepared),total=len(entries)))
                return
            raise ValueError('Native preparation did not produce all caches')
        cache_hash=digest(cache)
        if not metadata.exists() or json.loads(metadata.read_text())!=dict(identity=expected,cache_sha256=cache_hash):
            raise ValueError('Cache has no matching committed checksum')
        prepared.append(dict(row,cache=str(cache.resolve()),cache_sha256=cache_hash,
                             means=target['means'],means_sha256=target['means_sha256']))
    # Recheck every cache before marking the dataset ready for the trainer.
    if digest(dataset_path)!=identity['ar_dataset_sha256'] or digest(args.targets)!=identity['target_index_sha256']:
        raise ValueError('Input manifest changed during preparation')
    if auxiliary_identity and (digest(args.auxiliary_dataset)!=auxiliary_identity['dataset_sha256'] or
                               digest(args.auxiliary_targets)!=auxiliary_identity['targets_sha256']):
        raise ValueError('Auxiliary manifest changed during preparation')
    for row in prepared:
        for key in ['cache','prefix','codes','means'] if args.storage=='cache' else ['prefix','codes','means']:
            if digest(Path(row[key])) != row[key+'_sha256']:
                raise ValueError('Input or cache changed during preparation')
    save(args.out/'dataset.json',dict(format='yue2-nar-prepared-dataset-v1',completed=True,
         frames=data['frames'],identity=identity,rows=prepared))
    save(args.out/'progress.json',dict(status='completed',stage='frozen-ar-conditioning',current=len(pairs),total=len(pairs)))


if __name__=='__main__':main()

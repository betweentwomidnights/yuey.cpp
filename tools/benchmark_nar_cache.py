#!/usr/bin/env python3
"""Manual real-weight cache benchmark and numerical compatibility checks.

Use a completed AR job and VAE target index. Writes only a new benchmark folder.
CPU/CUDA rounding differences are reported, never described as exact parity.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import gguf
import numpy as np


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8'); temporary.replace(path)


def match_targets(rows, targets):
    """Each example's target: same recording, start and length, same split."""
    key = lambda row: (row['source_sha256'], row['start_frame'], row['frames'])
    by_key = {key(row): row for row in targets}
    if len(by_key) != len(targets): raise ValueError('Duplicate VAE target core')
    pairs = []
    for row in rows:
        target = by_key.pop(key(row), None)
        if target is None or target['split'] != row['split']: raise ValueError('VAE target missing or split differs')
        pairs.append((row, target))
    if by_key: raise ValueError('Unexpected VAE target cores')
    return pairs


def tensors(path):
    reader=gguf.GGUFReader(str(path));result={}
    for tensor in reader.tensors:
        array=tensor.data
        if tensor.tensor_type==gguf.GGMLQuantizationType.F16:array=array.view(np.float16).astype(np.float32)
        result[tensor.name]=np.array(array,copy=True)
    return result


def compare(a,b):
    return {key:dict(exact=np.array_equal(value,b[key]),relative_l2=float(np.linalg.norm(value.astype(np.float64)-b[key])/max(1e-30,np.linalg.norm(value.astype(np.float64))))) for key,value in a.items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['config','targets','ar-adapter','binary','out']:parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--full-cuda',action='store_true',help='Also allocate the legacy full CUDA model; requires more VRAM')
    args=parser.parse_args()
    if args.out.exists():parser.error('Output must be a new folder')
    args.out.mkdir(parents=True);os.environ['GGML_CUDA_TF32']='0'
    config=json.loads(args.config.read_text());data=json.loads(Path(config['training_dataset']).read_text())
    targets=json.loads(args.targets.read_text());pairs=match_targets([r for r in data['rows'] if r.get('eligible',True) and r.get('kind','real')=='real'],targets['rows'])
    row,target=max(pairs,key=lambda pair:Path(pair[0]['prefix']).stat().st_size)
    base=list(map(str,[args.binary,'--model',config['models']['base'],'--conditioning-lora',config['models']['companion'],
                      '--conditioning-lora',args.ar_adapter,'--threads',6]))
    inputs=dict(prefix=row['prefix'],tokens=row['codes'],latents=target['means']);records=[]
    def execute(label,device,selected=False,half=False):
        path=args.out/(label+'.gguf');command=base+['--prepare','--device',device,'--cache-out',str(path)]
        for key,value in inputs.items():command+=['--'+key,str(value)]
        if selected:command+=['--conditioning-only']
        if half:command+=['--cache-kv-type','f16']
        start=time.monotonic()
        with (args.out/(label+'.log')).open('w') as log:
            result=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            save(args.out/'active.json',dict(name=label,pid=result.pid))
            result.wait()
        record=dict(name=label,command=command,seconds=time.monotonic()-start,returncode=result.returncode,bytes=path.stat().st_size if path.exists() else None)
        records.append(record);save(args.out/'results.json',dict(records=records))
        if result.returncode:raise RuntimeError('Cache benchmark failed: '+label)
        return tensors(path)
    cpu=execute('cpu-full','cpu');selected=execute('cpu-selected','cpu',True)
    assert all(x['exact'] for x in compare(cpu,selected).values())
    cuda=execute('cuda-selected','cuda',True);half=execute('cuda-compact','cuda',True,True)
    assert all(x['exact'] for x in compare(cuda,half).values())
    comparisons=dict(cpu_selected_vs_full=compare(cpu,selected),compact_vs_f32=compare(cuda,half),cpu_vs_cuda=compare(cpu,cuda))
    if args.full_cuda:
        full=execute('cuda-full','cuda');comparisons['cuda_selected_vs_full']=compare(full,cuda)
        assert all(x['exact'] for x in comparisons['cuda_selected_vs_full'].values())
    plan=[dict(inputs,**{'cache-out':str((args.out/f'batch-{i}.gguf').resolve()),'cache-identity':f'benchmark-{i}'}) for i in range(2)]
    save(args.out/'batch.json',plan);command=base+['--conditioning-only','--cache-kv-type','f16','--device','cuda','--prepare-batch',str(args.out/'batch.json')]
    start=time.monotonic()
    with (args.out/'batch.log').open('w') as log:
        result=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        save(args.out/'active.json',dict(name='resident-pair',pid=result.pid))
        result.wait()
    elapsed=time.monotonic()-start
    if result.returncode:raise RuntimeError('Resident batch failed')
    for i in range(2):
        actual=tensors(args.out/f'batch-{i}.gguf')
        assert all(np.array_equal(v,actual[k]) for k,v in half.items())
    records.append(dict(name='resident-pair',seconds=elapsed,command=command,returncode=result.returncode))
    save(args.out/'results.json',dict(records=records,comparisons=comparisons,prefix_tokens=Path(row['prefix']).stat().st_size//4,frames=row['frames']))
    save(args.out/'complete.json',dict(success=True))
    print(json.dumps(dict(records=records,comparisons=comparisons),indent=2))


if __name__=='__main__':main()

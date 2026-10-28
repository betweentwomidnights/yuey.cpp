#!/usr/bin/env python3
"""Prepare posterior-mean audio targets with a resident native GGML encoder.

Python validates the manifest and records provenance; audio reading, resampling
and all model computation run in yue2-vae-encode. PyTorch is a developer oracle
only and is never imported by this production preparation path.
"""
import argparse
import hashlib
import math
import json
from pathlib import Path
import os
import subprocess
import time
import numpy as np
import soundfile as sf
from prepare_ar_dataset import save
from training_provenance import digest,native_runtime_identity

def context_bounds(start,frames,total,halo):
    if start<0 or frames<1 or start+frames>total or halo<0:raise ValueError('Invalid target core')
    begin=max(0,start-halo);end=min(total,start+frames+halo)
    return begin,end,start-begin


def source_frame_count(row,info,include_generated):
    trim=row.get('native_decoder_trim_samples',0)
    if trim:
        # Native decode deliberately returns latent_frames*1920-64 samples.
        # Its final posterior bin is valid for generated replay, but it must
        # never silently relax the real-source 30-second coverage requirement.
        if not include_generated or row.get('kind','real')=='real' or trim!=64 or info.samplerate!=48000:
            raise ValueError('Invalid native decoder edge declaration')
        return (info.frames+trim)//1920
    return math.floor(info.duration*25)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--vae',type=Path,required=True)
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--core-frames',type=int,default=128)
    parser.add_argument('--halo-frames',type=int,default=50)
    parser.add_argument('--stop-file',type=Path)
    parser.add_argument('--include-generated',action='store_true')
    args=parser.parse_args()
    if args.threads<1 or args.core_frames<1 or args.halo_frames<0:parser.error('Invalid VAE preparation profile')
    os.environ['GGML_CUDA_TF32']='0'
    data=json.loads(args.dataset.read_text());args.out.mkdir(parents=True,exist_ok=True)
    if not data['completed']:raise ValueError('Dataset preparation is incomplete')
    rows=[r for r in data['rows'] if r.get('eligible',True) and (args.include_generated or r.get('kind','real')=='real')]
    if not rows:raise ValueError('No eligible VAE source rows')
    identity=dict(format='native-vae-target-v1',vae_sha256=digest(args.vae),native_runtime=native_runtime_identity(args.binary.resolve()),
                  device=args.device,threads=args.threads,core_frames=args.core_frames,encoder_halo_frames=16,
                  posterior='mean',precision='GGML F32 weights/activations from folded F16/F32 VAE',halo_frames=args.halo_frames,
                  resampler='native per-channel centered polyphase Kaiser5 FIR; unchanged amplitude; zero-extended context',
                  boundary_policy='Real source context up to halo; native encoder padding at actual recording edges',
                  precision_environment={'GGML_CUDA_TF32':'0'})
    if args.include_generated:identity['target_source_policy']='Explicit real/generated waveform posterior means'
    index_path=args.out/'index.json'
    if index_path.exists() and json.loads(index_path.read_text())['identity']!=identity:
        raise ValueError('VAE target profile changed; use a new output folder. Existing targets are preserved.')
    prepared=[];pending=[];checked={};entries=[]
    for row in rows:
        source=Path(row['source']);info=sf.info(source)
        if row['source_sha256'] not in checked:
            if digest(source)!=row['source_sha256']:raise ValueError('Source changed')
            checked[row['source_sha256']]=source
        begin,end,offset=context_bounds(row['start_frame'],row['frames'],source_frame_count(row,info,args.include_generated),args.halo_frames)
        folder=args.out/(row['source_sha256'][:12]+f"-{row['start_frame']:07d}-{row['frames']}");folder.mkdir(exist_ok=True)
        target=folder/'means.f32';metadata=folder/'target.json'
        expected=dict(source_sha256=row['source_sha256'],start_frame=row['start_frame'],frames=row['frames'],context_begin_frame=begin,context_end_frame=end,**identity)
        if row.get('native_decoder_trim_samples'):expected['native_decoder_trim_samples']=row['native_decoder_trim_samples']
        if target.exists() and metadata.exists():
            previous=json.loads(metadata.read_text())
            if previous['identity']!=expected or digest(target)!=previous['means_sha256']:raise ValueError('VAE target identity changed; use a new output folder')
        else:
            if target.exists():
                number=1
                while target.with_name(f'means.unregistered-{number}.f32').exists():number+=1
                target.rename(target.with_name(f'means.unregistered-{number}.f32'))
            key=hashlib.sha256(json.dumps(expected,sort_keys=True).encode()).hexdigest()
            pending.append(dict(audio=str(source.resolve()),begin_frame=begin,end_frame=end,core_offset=offset,frames=row['frames'],
                                out=str(target.resolve()),identity=key,decoder_trim_samples=row.get('native_decoder_trim_samples',0)))
        entries.append((row,target,metadata,expected))
    def record(entry):
        row,target,metadata,expected=entry
        values=np.fromfile(target,dtype='<f4')
        if values.size!=row['frames']*64 or not np.isfinite(values).all():raise ValueError('Invalid native VAE target payload')
        if not metadata.exists():save(metadata,dict(identity=expected,means_sha256=digest(target),shape=[row['frames'],64],latent_rms=float(np.sqrt(np.mean(values.astype(np.float64)**2)))))
        previous=json.loads(metadata.read_text())
        if previous['identity']!=expected or digest(target)!=previous['means_sha256']:raise ValueError('VAE target changed')
        return dict(source_sha256=row['source_sha256'],track=row['track'],start_frame=row['start_frame'],frames=row['frames'],split=row['split'],means=str(target.resolve()),means_sha256=previous['means_sha256'])
    # Write a manifest even before the first event; partial data is never ready.
    save(index_path,dict(completed=False,identity=identity,rows=[]))
    if pending:
        plan=args.out/'pending.json';save(plan,pending)
        command=list(map(str,[args.binary.resolve(),'--model',args.vae.resolve(),'--plan',plan.resolve(),'--device',args.device,
                              '--threads',args.threads,'--core-frames',args.core_frames]))
        if args.stop_file:command+=['--stop-file',str(args.stop_file.resolve())]
        completed=len(entries)-len(pending);start=time.monotonic()
        save(args.out/'progress.json',dict(stage='vae-targets-native',backend=args.device,current=completed,total=len(rows)))
        by_path={str(target.resolve()):(row,target,metadata,expected) for row,target,metadata,expected in entries}
        with (args.out/'encode.log').open('a',encoding='utf-8') as log:
            process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
            try:
                for line in process.stdout:
                    log.write(line);log.flush()
                    if line.startswith('VAE_EVENT '):
                        event=json.loads(line[len('VAE_EVENT '):]);item=pending[event['index']]
                        if event['identity']!=item['identity'] or event['frames']!=item['frames']:raise ValueError('Native VAE event identity differs')
                        entry=by_path[item['out']];record(entry);completed+=1
                        save(args.out/'progress.json',dict(stage='vae-targets-native',backend=args.device,current=completed,total=len(rows),track=entry[0]['track']))
                        print('VAE',completed,len(rows),entry[0]['track'],entry[0]['start_frame'],flush=True)
            except BaseException:
                for line in process.stdout:log.write(line)
                raise
            finally:process.wait()
        save(args.out/'command.json',dict(command=command,returncode=process.returncode,seconds=time.monotonic()-start))
        if process.returncode:raise RuntimeError((args.out/'encode.log').read_text()[-5000:])
    for entry in entries:
        if not entry[1].exists():
            if args.stop_file and args.stop_file.exists():return
            raise ValueError('Native VAE output missing')
        if not entry[2].exists():raise ValueError('Native VAE output has no committed provenance')
        prepared.append(record(entry))
        save(index_path,dict(completed=False,identity=identity,rows=prepared))
    for expected,source in checked.items():
        if digest(source)!=expected:raise ValueError('Source changed during VAE encoding')
    save(index_path,dict(completed=True,identity=identity,rows=prepared))
    save(args.out/'progress.json',dict(stage='vae-targets-native-complete',backend=args.device,current=len(rows),total=len(rows)))


if __name__=='__main__':main()

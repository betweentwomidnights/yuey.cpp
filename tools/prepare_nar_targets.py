#!/usr/bin/env python3
"""Cache CPU VAE posterior means independently of the future AR/NAR choice."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch

from encode_yue2_vae_targets import Encoder
from prepare_ar_dataset import save
from yue2_training_audit import digest


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
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--halo-frames',type=int,default=50)
    parser.add_argument('--stop-file',type=Path)
    parser.add_argument('--include-generated',action='store_true',
                        help='Also encode explicitly sourced generated replay/retention rows')
    args=parser.parse_args();data=json.loads(args.dataset.read_text());args.out.mkdir(parents=True,exist_ok=True)
    if not data['completed']:raise ValueError('Dataset preparation is incomplete')
    rows=[r for r in data['rows'] if r.get('eligible',True) and
          (args.include_generated or r.get('kind','real')=='real')]
    if not rows:raise ValueError('No eligible VAE source rows')
    torch.set_num_threads(args.threads);encoder=Encoder(args.vae)
    identity=dict(vae_sha256=digest(args.vae),encoder_source_sha256=digest(Path(__file__).with_name('encode_yue2_vae_targets.py')),
                  torch_version=str(torch.__version__),posterior='mean',precision='CPU F32 from folded F16/F32 VAE',
                  halo_frames=args.halo_frames,boundary_policy='Real source context up to halo; native encoder boundary padding at actual recording edges')
    if args.include_generated:identity['target_source_policy']='Explicitly sourced real or generated waveform; posterior means, not original synthesis latents'
    checked={};prepared=[]
    for i,row in enumerate(rows):
        if args.stop_file and args.stop_file.exists():return
        source=Path(row['source']);info=sf.info(source);stat=source.stat()
        if row['source_sha256'] not in checked:
            if digest(source)!=row['source_sha256']:raise ValueError('Source changed')
            checked[row['source_sha256']]=(stat.st_size,stat.st_mtime_ns)
        elif checked[row['source_sha256']]!=(stat.st_size,stat.st_mtime_ns):raise ValueError('Source changed during encoding')
        begin,end,offset=context_bounds(row['start_frame'],row['frames'],source_frame_count(row,info,args.include_generated),args.halo_frames)
        folder=args.out/(row['source_sha256'][:12]+f"-{row['start_frame']:07d}-{row['frames']}");folder.mkdir(exist_ok=True)
        target=folder/'means.f32';metadata=folder/'target.json'
        expected=dict(source_sha256=row['source_sha256'],start_frame=row['start_frame'],frames=row['frames'],context_begin_frame=begin,context_end_frame=end,**identity)
        if row.get('native_decoder_trim_samples'):expected['native_decoder_trim_samples']=row['native_decoder_trim_samples']
        if target.exists():
            previous=json.loads(metadata.read_text())
            if previous['identity']!=expected or digest(target)!=previous['means_sha256']:raise ValueError('VAE target identity changed')
        else:
            save(args.out/'progress.json',dict(stage='vae-targets',current=i,total=len(rows),track=row['track']))
            with sf.SoundFile(source) as stream:
                first=round(begin/25*info.samplerate);count=round((end-begin)/25*info.samplerate)
                if row.get('native_decoder_trim_samples'):count=min(count,info.frames-first)
                stream.seek(first);audio=stream.read(count,dtype='float32',always_2d=True)
            if len(audio)!=count:raise ValueError('Source context read is incomplete')
            if info.samplerate!=48000:
                divisor=math.gcd(info.samplerate,48000);audio=resample_poly(audio,48000//divisor,info.samplerate//divisor,axis=0)
            latent=encoder.encode(audio)[offset:offset+row['frames']]
            if latent.shape!=(row['frames'],64) or not np.isfinite(latent).all():raise ValueError('Invalid VAE target shape/values')
            temporary=target.with_suffix('.f32.partial');latent.astype('<f4').tofile(temporary);temporary.rename(target)
            save(metadata,dict(identity=expected,means_sha256=digest(target),shape=list(latent.shape),latent_rms=float(np.sqrt(np.mean(latent.astype(np.float64)**2)))))
        prepared.append(dict(source_sha256=row['source_sha256'],track=row['track'],start_frame=row['start_frame'],frames=row['frames'],split=row['split'],means=str(target),means_sha256=digest(target)))
        save(args.out/'index.json',dict(completed=i+1==len(rows),identity=identity,rows=prepared))
        print('VAE',i+1,len(rows),row['track'],row['start_frame'],flush=True)
    save(args.out/'progress.json',dict(stage='vae-targets-complete',current=len(rows),total=len(rows)))


if __name__=='__main__':main()

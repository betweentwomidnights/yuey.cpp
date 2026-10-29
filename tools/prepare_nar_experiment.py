#!/usr/bin/env python3
"""Prepare bounded, whole-song-held-out native NAR experiments.

Full-track semantic extraction precedes cropping. VAE targets have 2 seconds of
surrounding source context on each side. Source audio and captions stay read-only.
This helper performs local preprocessing only; it does not upload audio.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch

from encode_yue2_vae_targets import Encoder
from yue2_training_audit import digest, native_runtime_identity


def fractional_window_starts(duration, frames, fractions):
    """Plan non-overlapping target cores on the 25 Hz grid, with VAE context."""
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('invalid source duration')
    if not fractions or any(not math.isfinite(f) or not 0 <= f <= 1 for f in fractions):
        raise ValueError('window fractions must be finite and in [0,1]')
    if len(set(fractions)) != len(fractions):
        raise ValueError('duplicate window fraction')
    first, last = 50, math.floor(duration * 25) - frames - 50
    if last < first:
        raise ValueError('source is too short for a core and two VAE halos')
    starts = [round(first + fraction * (last-first)) for fraction in sorted(fractions)]
    if any(b-a < frames for a,b in zip(starts, starts[1:])):
        raise ValueError('training window cores overlap; reduce fractions or window length')
    return starts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=["features","cache"])
    parser.add_argument("--dataset",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--track",action="append",required=True,help="exact WAV filename; training track")
    parser.add_argument("--holdout",action="append",required=True,help="exact WAV filename; whole-song holdout")
    parser.add_argument("--bin",type=Path,required=True)
    parser.add_argument("--vae",type=Path,required=True)
    parser.add_argument("--semantic-model",type=Path,required=True)
    parser.add_argument("--model",type=Path)
    parser.add_argument("--tokenizer",type=Path)
    parser.add_argument("--fixed-lora",type=Path,action="append",default=[])
    parser.add_argument("--frames",type=int,default=256)
    parser.add_argument("--offset",type=float,default=20)
    parser.add_argument("--window-fraction",type=float,action='append',default=[],
                        help='repeat for training windows across usable song length; holdouts retain --offset')
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--conditioning-device",default="cpu")
    args=parser.parse_args()
    if args.frames<2 or args.frames>1024 or not math.isfinite(args.offset) or args.offset<2:
        parser.error("frames must be in [2,1024]; offset must be at least 2 seconds")
    if set(args.track)&set(args.holdout):parser.error("a track cannot appear in both splits")
    names=args.track+args.holdout
    if len(set(names))!=len(names):parser.error("duplicate track")
    args.out.mkdir(parents=True,exist_ok=True)
    config={"preprocessing_version":1,"vae_halo_frames":50,"frames":args.frames,"offset":args.offset,
            "vae":digest(args.vae),"head":digest(args.semantic_model),"tracks":[]}
    executable=lambda name:args.bin/(name+('.exe' if os.name=='nt' else ''))
    if args.window_fraction:
        config.update(preprocessing_version=2, training_window_fractions=sorted(args.window_fraction),
                      semantic_runtime=native_runtime_identity(executable('yue2-semantic-tokenize')),
                      semantic_device=args.device, encoder_source_sha256=digest(Path(__file__).with_name('encode_yue2_vae_targets.py')),
                      torch_version=str(torch.__version__), windows=[])
    seen_hashes={}
    for name in names:
        if Path(name).name!=name:parser.error("track names must be filenames inside the dataset")
        path=args.dataset/name
        caption=path.with_suffix('.txt')
        text=caption.read_text(encoding='utf-8-sig').strip() if caption.exists() else 'instrumental electronic music'
        fingerprint=digest(path)
        if fingerprint in seen_hashes:
            parser.error('duplicate source audio: '+name+' and '+seen_hashes[fingerprint])
        seen_hashes[fingerprint]=name
        item={"source":str(path.resolve()),"sha256":fingerprint,"caption":text,
              "split":"validation" if name in args.holdout else "train"}
        config["tracks"].append(item)
        if args.window_fraction:
            duration=sf.info(path).duration
            starts=(fractional_window_starts(duration,args.frames,args.window_fraction)
                    if item['split']=='train' else [round(args.offset*25)])
            if any(s<50 or s+args.frames+50>math.floor(duration*25) for s in starts):
                parser.error('insufficient source context: '+name)
            for start in starts:
                config['windows'].append(dict(item,start_frame=start,offset=start/25,
                    directory=f'{fingerprint[:12]}/at-{start:07d}'))
    manifest=args.out/"features.json"
    if manifest.exists() and json.loads(manifest.read_text())!=config:parser.error("feature configuration changed; use a new output directory")
    manifest.write_text(json.dumps(config,indent=2),encoding="utf-8")
    windows=config.get('windows',[dict(item,start_frame=round(args.offset*25),offset=args.offset,
                                     directory=item['sha256'][:12]) for item in config['tracks']])
    def execute(command,target,log):
        if target.exists():return
        with log.open('w',encoding='utf-8') as stream:
            subprocess.run([str(x) for x in command],check=True,stdout=stream,stderr=subprocess.STDOUT)
    if args.phase=='features':
        torch.set_num_threads(4); encoder=Encoder(args.vae)
        for item in config['tracks']:
            path=Path(item['source']); directory=args.out/item['sha256'][:12];directory.mkdir(exist_ok=True)
            full=directory/'full.i32'
            execute([executable('yue2-semantic-tokenize'),'--model',args.semantic_model,'--audio',path,'--out',full,'--device',args.device],full,directory/'tokenize.log')
            codes=np.fromfile(full,dtype='<i4')
            if not len(codes) or np.any((codes<0)|(codes>=32768)):
                raise ValueError('invalid full-track semantic stream: '+path.name)
            print(item['split'],path.name,'full-track tokens prepared',flush=True)
        for item in windows:
            path=Path(item['source']); directory=args.out/item['directory'];directory.mkdir(parents=True,exist_ok=True)
            codes=np.fromfile(args.out/item['sha256'][:12]/'full.i32',dtype='<i4');start=item['start_frame']
            if len(codes)<start+args.frames:raise ValueError('track is too short: '+path.name)
            codes[start:start+args.frames].tofile(directory/'codes.i32')
            means=directory/'means.f32'
            if not means.exists():
                # Read from a 25-Hz-aligned offset, retaining encoder context.
                begin=(start-50)/25; count=(args.frames+100)/25
                info=sf.info(path)
                with sf.SoundFile(path) as audio:
                    audio.seek(round(begin*info.samplerate))
                    samples=audio.read(round(count*info.samplerate),dtype='float32',always_2d=True)
                if len(samples)<round(count*info.samplerate):raise ValueError('insufficient VAE context: '+path.name)
                if info.samplerate!=48000:
                    divisor=math.gcd(info.samplerate,48000)
                    samples=resample_poly(samples,48000//divisor,info.samplerate//divisor,axis=0)
                latent=encoder.encode(samples)[50:50+args.frames]
                if latent.shape!=(args.frames,64):raise ValueError('VAE target alignment failed')
                latent.astype('<f4').tofile(means)
                core=samples[96000:96000+args.frames*1920]
                sf.write(directory/'source.wav',core,48000,subtype='FLOAT')
            (directory/'caption.txt').write_text(item['caption'],encoding='utf-8')
            print(item['split'],path.name,'at',start/25,'seconds prepared',flush=True)
    else:
        if not args.model or not args.tokenizer:parser.error('cache requires --model and --tokenizer')
        identity={'features':config,'base_model_sha256':digest(args.model),'tokenizer_sha256':digest(args.tokenizer),
                  'fixed_loras':[digest(p) for p in args.fixed_lora],
                  'conditioning_device':args.conditioning_device,'cache_builder_sha256':digest(executable('yue2-train')),
                  'cache_runtime':native_runtime_identity(executable('yue2-train'))}
        import hashlib
        identity['run_identity']=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        recorded=args.out/'cache-config.json'
        if recorded.exists() and json.loads(recorded.read_text())!=identity:parser.error('cache identity changed; use a new output directory')
        recorded.write_text(json.dumps(identity,indent=2),encoding='utf-8')
        samples=[]
        for item in windows:
            directory=args.out/item['directory']; target=directory/'sample.gguf'
            command=[executable('yue2-train'),'--prepare','--model',args.model,'--tokenizer',args.tokenizer,
                     '--tokens',directory/'codes.i32','--latents',directory/'means.f32','--cache-out',target,
                     '--prompt',(directory/'caption.txt').read_text(encoding='utf-8'),'--device',args.conditioning_device]
            for adapter in args.fixed_lora:command+=['--fixed-lora',adapter]
            execute(command,target,directory/'cache.log')
            samples.append(dict(item,cache=str(target.resolve()),cache_sha256=digest(target),directory=str(directory.resolve())))
            print(item['split'],Path(item['source']).name,'at',item['offset'],'seconds cached',flush=True)
        (args.out/'samples.json').write_text(json.dumps(samples,indent=2),encoding='utf-8')
        identity['sample_sha256']=[s['cache_sha256'] for s in samples]
        identity.pop('run_identity')
        identity['run_identity']=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        (args.out/'cache-identity.json').write_text(json.dumps(identity,indent=2),encoding='utf-8')


if __name__=='__main__':main()

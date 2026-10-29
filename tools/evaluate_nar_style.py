#!/usr/bin/env python3
"""Reference-free NAR style gate: fresh scores, user audio and text-only generation.

No training runs here. Dataset recordings and captions are never generation inputs.
The source recording is transcribed to content; listener guides are not model inputs.
"""
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import time

import numpy as np
import soundfile as sf

from yue2_training_audit import digest, native_runtime_identity


def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    temporary.replace(path)


def score(title, meter, bpm, key, chords, bars):
    units=6 if meter=='3/4' else 8
    return (f'X:1\nT:{title}\nM:{meter}\nL:1/8\nQ:1/4={bpm}\n'
            'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"\n'
            'V: Ins clef=treble name="Ins Melody" snm="Inst."\n'
            f'K:{key}\n% verse\nV: Vocal\n'+
            '|'.join(f'"{c}"z{units}' for c in chords)+'|\nV: Ins\n'+'|'.join(bars)+'|\n')


def melody_guide(bars, units, bpm):
    """Dry harmonic melody for the listener only, from our restricted authored ABC."""
    rate=48000;seconds_per_unit=30/bpm
    data=np.zeros(round(len(bars)*units*seconds_per_unit*rate),dtype=np.float64)
    natural={'C':60,'D':62,'E':64,'F':65,'G':67,'A':69,'B':71}
    at=0.
    for bar in bars:
        total=0
        for accidental,pitch,octave,count in re.findall(r'([\^_=]?)([A-Ga-gz])([,\x27]*)(\d*)',bar):
            count=int(count or 1);total+=count;duration=count*seconds_per_unit
            if pitch!='z':
                midi=natural[pitch.upper()]+(12 if pitch.islower() else 0)+12*(octave.count("'")-octave.count(','))
                midi+=1 if accidental=='^' else -1 if accidental=='_' else 0
                n=round(duration*rate);t=np.arange(n)/rate
                frequency=440*2**((midi-69)/12)
                envelope=np.minimum(t/.008,1)*np.minimum((duration-t)/.035,1)*np.exp(-2*t/max(duration,.1))
                tone=sum(g*np.sin(2*np.pi*frequency*h*t) for h,g in [(1,1),(2,.22),(3,.09)])
                start=round(at*rate);end=min(start+n,len(data));data[start:end]+=.24*tone[:end-start]*envelope[:end-start]
            at+=duration
        if total!=units:raise ValueError(f'authored bar duration {total} != {units}: {bar}')
    return np.column_stack([data,data]),rate


class Gate:
    def __init__(self,args):
        self.args=args;self.out=args.out.resolve();self.bin=args.bin.resolve()
        self.env=dict(os.environ,GGML_CUDA_TF32='0')
        self.extension='.exe' if os.name=='nt' else ''

    def exe(self,name):return self.bin/(name+self.extension)

    def execute(self,name,command):
        command=list(map(str,command));logs=self.out/'logs';logs.mkdir(parents=True,exist_ok=True)
        attempt=1
        while (logs/f'{name}-{attempt}.log').exists():attempt+=1
        path=logs/f'{name}-{attempt}.log';start=time.monotonic()
        print('START',name,flush=True)
        with path.open('w',encoding='utf-8') as stream:
            result=subprocess.run(command,env=self.env,stdout=stream,stderr=subprocess.STDOUT)
        write_json(path.with_suffix('.command.json'),dict(command=command,returncode=result.returncode,
            elapsed_seconds=time.monotonic()-start,environment={'GGML_CUDA_TF32':'0'}))
        print('END',name,result.returncode,f'{time.monotonic()-start:.1f}s',flush=True)
        if result.returncode:raise RuntimeError(path.read_text(encoding='utf-8',errors='replace')[-5000:])

    def common(self):
        a=self.args
        result=[self.exe('yue2-generate'),'--model',a.model,'--vae',a.vae,'--tokenizer',a.tokenizer,
                '--instrumental','--device',a.device,'--ode-steps',a.steps]
        for path in a.fixed_lora:result+=['--lora',path]
        return result

    def prepare(self):
        a=self.args;self.out.mkdir(parents=True,exist_ok=True)
        os.environ['GGML_CUDA_TF32']='0' # Match recorded identities to child precision.
        identity=dict(version=1,source_sha256=digest(a.source_audio),
            source_info=dict(samplerate=sf.info(a.source_audio).samplerate,channels=sf.info(a.source_audio).channels,
                             seconds=sf.info(a.source_audio).duration),
            models={str(p.resolve()):digest(p) for p in [a.model,a.vae,a.tokenizer,a.transcription_model,
                                                      *a.fixed_lora,a.reference,a.candidate]},
            generation_runtime=native_runtime_identity(self.exe('yue2-generate')),
            transcription_runtime=native_runtime_identity(self.exe('yue2-transcribe')),
            evaluator_sha256=digest(Path(__file__)),device=a.device,steps=a.steps,seeds=[831001,123],
            neutral_caption='Instrumental music.',strength_probe=2.0,
            source_provenance='User-provided piano/orchestral recording; no assertion that its underlying composition was absent from training.')
        config=self.out/'identity.json'
        if config.exists() and json.loads(config.read_text())!=identity:
            raise ValueError('configuration/runtime changed; use a fresh output directory')
        write_json(config,identity)
        inputs=self.out/'inputs';inputs.mkdir(exist_ok=True)
        source=inputs/'source-16bar';source.mkdir(exist_ok=True)
        copy=source/'source.wav'
        if not copy.exists():shutil.copy2(a.source_audio,copy)
        if digest(copy)!=identity['source_sha256']:raise ValueError('source copy mismatch')
        audio,rate=sf.read(copy,dtype='float64',always_2d=True)
        if not np.isfinite(audio).all() or len(audio)/rate<4:raise ValueError('invalid source audio')
        abc=source/'score.abc'
        if not abc.exists():
            self.execute('transcribe-source',[self.exe('yue2-transcribe'),'--model',a.transcription_model,
                '--audio',copy,'--output',abc,'--midi',source/'transcription.mid','--events',source/'events.json',
                '--full','--window-seconds','30','--overlap-seconds','20','--lookahead-seconds','10','--device',a.device])
        cases=[dict(id='source-16bar',mode='source-to-score',caption='Instrumental music.',
                    frames=round(len(audio)/rate*25),seeds=[831001,123],score=str(abc),source=str(copy),
                    notes='Full user-provided passage, transcribed to ABC; no source latents or source semantic tokens supplied.')]
        authored=[('score-waltz','Fresh waltz theme','3/4',90,'C',
                   ['C','F','G','C','Am','G','F','C'],
                   ['C2EGcG','A2FAcA','G2DGBG','E2CGc2','A2cedc','B2GBdB','cBAGFE','D2G2C2']),
                  ('score-syncopation','Fresh syncopated theme','4/4',120,'Am',
                   ['Am','Am','Dm','E7','Am','F','E7','Am'],
                   ['A,2E2zAE2','AcEAz2E2','D2A2zDF2','E2B2^G2E2','A2z2e2c2','F2A2cBAG','E2^G2B2d2','c2BAE2A,2'])]
        for name,title,meter,bpm,key,chords,bars in authored:
            folder=inputs/name;folder.mkdir(exist_ok=True)
            text=score(title,meter,bpm,key,chords,bars);path=folder/'score.abc'
            if path.exists() and path.read_text(encoding='utf-8')!=text:raise ValueError('authored score changed')
            path.write_text(text,encoding='utf-8')
            guide,sr=melody_guide(bars,6 if meter=='3/4' else 8,bpm)
            if not (folder/'guide.wav').exists():sf.write(folder/'guide.wav',guide,sr,subtype='FLOAT')
            checked=folder/'validated.abc'
            if not checked.exists():self.execute('validate-'+name,self.common()+['--prompt','Instrumental music.',
                '--abc',path,'--plan-only','--score-output',checked])
            cases.append(dict(id=name,mode='new-authored-score',caption='Instrumental music.',frames=400,
                seeds=[831001],score=str(path),guide=str(folder/'guide.wav'),
                notes='Newly authored eight-bar score. Dry melody guide is for the listener only and is never a model input.'))
        cases += [dict(id='generate-01',mode='standard-generation',caption='Instrumental music.',frames=400,seeds=[831001],
                       bpm=120,key='D minor',meter='4/4',notes='No supplied audio or score; fresh model-generated plan.'),
                  dict(id='generate-02',mode='standard-generation',caption='Instrumental music with changing dynamics.',frames=400,seeds=[831001],
                       bpm=90,key='C major',meter='3/4',notes='No supplied audio or score; fresh model-generated plan.')]
        for case in cases:
            if 'score' in case:case['score_sha256']=digest(Path(case['score']))
        manifest=dict(date='2026-10-25',cases=cases,no_dataset_recordings_or_captions_as_generation_inputs=True,
                      learned_style_not_assumed=True,fixed_semantic_budgets=True,
                      roles={'base':'Fixed companions only','reference':'Original four-track step48',
                             'candidate':'Expanded twelve-track step192','candidate-x2':'Expanded adapter strength 2; diagnostic only'})
        if (self.out/'cases.json').exists() and json.loads((self.out/'cases.json').read_text())!=manifest:
            raise ValueError('case manifest changed')
        write_json(self.out/'cases.json',manifest)

    def render(self):
        a=self.args;manifest=json.loads((self.out/'cases.json').read_text())
        os.environ['GGML_CUDA_TF32']='0'
        identity=json.loads((self.out/'identity.json').read_text())
        if identity['evaluator_sha256']!=digest(Path(__file__)) or identity['device']!=a.device or identity['steps']!=a.steps:
            raise ValueError('evaluator or execution configuration changed')
        if identity['generation_runtime']!=native_runtime_identity(self.exe('yue2-generate')):
            raise ValueError('generation runtime changed')
        for path in [a.model,a.vae,a.tokenizer,a.transcription_model,*a.fixed_lora,a.reference,a.candidate]:
            if identity['models'].get(str(path.resolve()))!=digest(path):raise ValueError('model/adapter identity changed')
        if identity['source_sha256']!=digest(a.source_audio):raise ValueError('source recording changed')
        for case in manifest['cases']:
            for seed in case['seeds']:
                baseline_codes=baseline_score=None
                roles=['base','reference','candidate']
                if seed==831001 and case['id'] in ('source-16bar','generate-01'):roles+=['candidate-x2']
                for role in roles:
                    folder=self.out/'raw'/case['id']/str(seed)/role;folder.mkdir(parents=True,exist_ok=True)
                    target=folder/'render.wav';ids=folder/'codes.i32';used=folder/'used.abc';done=folder/'done.json'
                    if not done.exists():
                        if target.exists() or ids.exists():raise ValueError('incomplete render exists: '+str(target))
                        command=self.common()+['--prompt',case['caption'],'--seed',seed,'--out',target,
                            '--score-output',used,'--semantic-output',ids,
                            '--semantic-min-tokens',case['frames'],'--semantic-max-tokens',case['frames']]
                        if case['mode']=='standard-generation':
                            command+=['--bars','8','--bpm',case['bpm'],'--key',case['key'],'--meter',case['meter'],
                                      '--abc-max-tokens','2048']
                        else:
                            if digest(Path(case['score']))!=case['score_sha256']:raise ValueError('score changed')
                            command+=['--abc',case['score']]
                        if role=='reference':command+=['--lora',a.reference]
                        elif role=='candidate':command+=['--lora',a.candidate]
                        elif role=='candidate-x2':command+=['--lora',str(a.candidate)+'=2']
                        self.execute(f'{case["id"]}-{seed}-{role}',command)
                        audio,sr=sf.read(target,dtype='float64',always_2d=True)
                        codes=np.fromfile(ids,dtype='<i4')
                        if not np.isfinite(audio).all() or abs(len(audio)/sr-case['frames']/25)>.1:
                            raise ValueError('invalid rendered audio')
                        if len(codes)!=case['frames'] or np.any((codes<0)|(codes>=32768)):raise ValueError('invalid codes')
                        write_json(done,dict(role=role,wav_sha256=digest(target),tokens_sha256=digest(ids),
                            used_score_sha256=digest(used),seconds=len(audio)/sr,
                            rms_db=float(20*np.log10(max(np.sqrt(np.mean(audio**2)),1e-12))),peak=float(np.abs(audio).max())))
                    record=json.loads(done.read_text())
                    if digest(target)!=record['wav_sha256'] or digest(ids)!=record['tokens_sha256']:
                        raise ValueError('render artifact changed')
                    if baseline_codes is None:baseline_codes,baseline_score=record['tokens_sha256'],record['used_score_sha256']
                    elif (baseline_codes,baseline_score)!=(record['tokens_sha256'],record['used_score_sha256']):
                        raise ValueError('NAR adapter changed upstream semantic tokens or score')
        write_json(self.out/'renders-complete.json',dict(groups=6,primary_renders=18,diagnostic_renders=2,
                   used_scores_and_semantic_tokens_match_within_groups=True))

    def pack(self):
        manifest=json.loads((self.out/'cases.json').read_text());assert (self.out/'renders-complete.json').exists()
        listening=self.out/'listening';listening.mkdir(exist_ok=False)
        pages={};key=[];ratings=[];rng=random.Random(20261025)
        permutations=list(itertools.permutations(['base','reference','candidate']));rng.shuffle(permutations)
        group_index=0;max_level_difference=0.;max_peak=0.
        def normalize(paths,stems):
            nonlocal max_level_difference,max_peak
            waves=[];rates=[]
            for path in paths:
                x,sr=sf.read(path,dtype='float64',always_2d=True);rms=float(np.sqrt(np.mean(x*x)))
                if not np.isfinite(x).all() or rms<1e-5:raise ValueError('silent/nonfinite preview input')
                waves.append(x*10**(-18/20)/rms);rates.append(sr)
            gain=min(1.,.98/max(float(np.abs(x).max()) for x in waves));levels=[];outputs=[]
            for wave,sr,stem in zip(waves,rates,stems):
                path=listening/(stem+'.wav');sf.write(path,wave*gain,sr,subtype='PCM_24')
                actual,_=sf.read(path,dtype='float64',always_2d=True)
                peak=float(np.abs(actual).max());assert peak<=.980001
                max_peak=max(max_peak,peak);levels.append(20*np.log10(np.sqrt(np.mean(actual**2))))
                outputs.append(path)
            max_level_difference=max(max_level_difference,max(levels)-min(levels))
            assert max(levels)-min(levels)<.0001
            return outputs
        for case in manifest['cases']:
            for seed in case['seeds']:
                group=f'{case["id"]}-{seed}';order=permutations[group_index];group_index+=1
                paths=[self.out/'raw'/case['id']/str(seed)/role/'render.wav' for role in order]
                stems=[group+'-'+label for label in 'ABC']
                source=case.get('source',case.get('guide'))
                if source:paths.append(Path(source));stems.append(group+'-content-reference')
                files=normalize(paths,stems)
                name='source-audio' if case['mode']=='source-to-score' else 'fresh-scores' if case['mode']=='new-authored-score' else 'standard-generation'
                body=pages.setdefault(name,[])
                body += [f'## {case["id"]}, seed {seed}',case['notes'],f'Caption: `{case["caption"]}`.']
                if source:
                    label='User source recording' if 'source' in case else 'Dry melody guide — listener only'
                    body.append(f'**{label}**\n\n![Content reference]({files[-1].as_posix()})')
                for label,path in zip('ABC',files[:3]):body.append(f'**{label}**\n\n![{label}]({path.as_posix()})')
                if 'score' in case:body.append(f'[Input ABC score]({Path(case["score"]).as_posix()})')
                else:body.append(f'[Model-generated plan]({(self.out/"raw"/case["id"]/str(seed)/"base/used.abc").as_posix()})')
                key.append(dict(group=group,**dict(zip('ABC',order))))
                ratings.append(dict(group=group,style_preference=None,content_preference=None,fidelity_preference=None,
                                    notes='',content_axis='source/score adherence' if source or 'score' in case else 'musical coherence and prompt adherence'))
        assert group_index==6
        header=('# Dataset-style capability test\n\nThree models are hidden behind A/B/C. Order changes across groups and is counterbalanced. '
                'All share the same used score and AR semantic tokens within a group. Audio is RMS matched with shared peak protection.\n\n'
                '**Judge separately:** Which resembles the Koan dataset\'s instrumentation, bass/sound-design textures and production? '
                'Which follows the content best? Which has the best audio fidelity? Ties and no style shift are valid answers. '
                'Cleaner cymbals alone are not evidence of learning the dataset\'s style.\n\n')
        for name,body in pages.items():(self.out/(name+'.md')).write_text(header+'\n\n'.join(body),encoding='utf-8')
        stress=['# Adapter-strength diagnostic','Labeled diagnostic, separate from the primary blind comparison. '
                'Does increasing the expanded adapter strength produce a stylistic shift, only change fidelity, or damage content?']
        for case_id in ('source-16bar','generate-01'):
            labels=['base','reference','candidate','candidate-x2']
            paths=[self.out/'raw'/case_id/'831001'/role/'render.wav' for role in labels]
            files=normalize(paths,[case_id+'-strength-'+label for label in labels]);stress.append('## '+case_id)
            for label,path in zip(labels,files):stress.append(f'**{manifest["roles"][label]}**\n\n![{label}]({path.as_posix()})')
        (self.out/'strength-diagnostic.md').write_text('\n\n'.join(stress),encoding='utf-8')
        write_json(self.out/'answer-key.json',key);write_json(self.out/'ratings.json',ratings)
        index=['# YuE2 dataset-style milestone',
               'No additional training was done. The candidates are fixed v9, original four-track step48 and expanded twelve-track step192. '
               'No dataset audio or dataset captions are supplied at generation time.',
               'Start with the full user-provided piano/orchestral source. Style resemblance, content adherence and fidelity are separate judgments. '
               'Read the strength diagnostic after the blind comparisons.',
               '- [Full source-audio transformation](source-audio.md)',
               '- [New independently authored scores](fresh-scores.md)',
               '- [Standard generation without supplied audio or scores](standard-generation.md)',
               '- [Labeled adapter-strength diagnostic](strength-diagnostic.md)',
               f'The supplied recording is {sf.info(self.args.source_audio).duration:.2f} seconds and is used in full. Authored and standard-generation probes are 16 seconds. '
               'Explicit semantic budgets bound the comparisons; natural endings are not evaluated. '
               'These are score-mediated remixes; no source-latent warm start is used. The supplied recording\'s underlying composition provenance is not established.',
               'Prefer natural-language notes or fill ratings.json. Keep style/fidelity preferences separate; unblind only after rating.']
        # Use absolute links in the app's local Markdown viewer.
        text='\n\n'.join(index)
        for name in [*pages,'strength-diagnostic']:text=text.replace(f']({name}.md)',f']({(self.out/(name+".md")).as_posix()})')
        (self.out/'listening.md').write_text(text+'\n',encoding='utf-8')
        write_json(self.out/'results.json',dict(date='2026-10-25',status='style gate rendered; human style judgments pending',
            groups=6,primary_renders=18,strength_diagnostic_renders=2,no_new_training=True,
            no_dataset_recordings_or_captions_as_inference_inputs=True,
            no_supplied_score_or_audio_for_standard_generation=True,
            used_scores_and_semantic_tokens_match_within_groups=True,
            max_preview_peak=max_peak,max_group_rms_difference_db=max_level_difference,
            source_sha256=digest(self.args.source_audio),source_unchanged=digest(self.args.source_audio)==digest(self.out/'inputs/source-16bar/source.wav')))
        print('Style review ready:',self.out/'listening.md',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['prepare','render','pack','all'])
    for name in ['out','bin','model','vae','tokenizer','transcription-model','source-audio','reference','candidate']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--fixed-lora',type=Path,action='append',default=[])
    parser.add_argument('--device',default='cuda');parser.add_argument('--steps',type=int,default=32)
    args=parser.parse_args();gate=Gate(args)
    if args.phase in ('prepare','all'):gate.prepare()
    if args.phase in ('render','all'):gate.render()
    if args.phase in ('pack','all'):gate.pack()


if __name__=='__main__':main()

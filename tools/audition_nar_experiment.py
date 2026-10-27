#!/usr/bin/env python3
"""Matched local checkpoint renders, source-latent controls, and ABC remix checks."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from yue2_training_audit import compare, digest, native_runtime_identity


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--bin',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True,help='inference model; may be quantized')
    parser.add_argument('--vae',type=Path,required=True)
    parser.add_argument('--tokenizer',type=Path,required=True)
    parser.add_argument('--fixed-lora',type=Path,action='append',default=[])
    parser.add_argument('--adapter',type=Path,action='append',default=[])
    parser.add_argument('--source-controls',action='store_true')
    parser.add_argument('--remix',action='store_true')
    parser.add_argument('--transcription-model',type=Path)
    parser.add_argument('--instrumental-lora',type=Path)
    parser.add_argument('--include-train',action='store_true')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--steps',type=int,default=32)
    parser.add_argument('--seed',type=int,default=831001)
    parser.add_argument('--label',default='auditions',help='output directory label; choose a new label after rebuilding')
    args=parser.parse_args()
    if Path(args.label).name!=args.label or args.label in ('.','..'):parser.error('label must be one directory name')
    if args.remix and not args.transcription_model:parser.error('remix requires transcription-model')
    config=json.loads((args.root/'features.json').read_text())
    executable=lambda name:args.bin/(name+('.exe' if os.name=='nt' else ''))
    run={'model':digest(args.model),'vae':digest(args.vae),'tokenizer':digest(args.tokenizer),
         'fixed_loras':[digest(p) for p in args.fixed_lora],'seed':args.seed,'steps':args.steps,
         'renderer':digest(executable('yue2-render-tokens')),
         'runtime':native_runtime_identity(executable('yue2-render-tokens')),'device':args.device}
    if args.remix:
        run['generator_runtime']=native_runtime_identity(executable('yue2-generate'))
        run['transcriber_runtime']=native_runtime_identity(executable('yue2-transcribe'))
    def execute(command,target,identity):
        record=target.with_suffix('.provenance.json')
        if record.exists() and json.loads(record.read_text())!=identity:raise ValueError('render identity changed: '+str(target))
        if target.exists() and record.exists():return
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.with_suffix('.log').open('w',encoding='utf-8') as stream:
            subprocess.run([str(x) for x in command],check=True,stdout=stream,stderr=subprocess.STDOUT)
        record.write_text(json.dumps(identity,indent=2),encoding='utf-8')
    results=[]
    for item in config.get('windows',config['tracks']):
        if item['split']!='validation' and not args.include_train:continue
        directory=args.root/item.get('directory',item['sha256'][:12])
        caption=(directory/'caption.txt').read_text(encoding='utf-8')
        variants=[('baseline',None,None)]+[(p.stem,p,None) for p in args.adapter]
        if args.source_controls:variants += [(f'strength-{s}',None,s) for s in (0,.35,.7,1)]
        if len({v[0] for v in variants})!=len(variants):raise ValueError('duplicate variant name')
        for label,adapter,strength in variants:
            target=directory/args.label/(label+'.wav')
            command=[executable('yue2-render-tokens'),'--model',args.model,'--vae',args.vae,'--tokenizer',args.tokenizer,
                     '--tokens',directory/'codes.i32','--prompt',caption,'--out',target,'--device',args.device,
                     '--seed',args.seed,'--ode-steps',args.steps]
            for fixed in args.fixed_lora:command+=['--lora',fixed]
            if adapter:command+=['--lora',adapter]
            if strength is not None:command+=['--source-latents',directory/'means.f32','--strength',strength]
            identity=dict(run,source=item['sha256'],start_frame=item.get('start_frame',round(config['offset']*25)),
                          adapter=digest(adapter) if adapter else None,strength=strength)
            execute(command,target,identity)
            result=compare(directory/'source.wav',target)
            result.update(source=Path(item['source']).name,split=item['split'],variant=label,
                          window_offset_seconds=item.get('offset',config['offset']),path=str(target.resolve()))
            results.append(result);print(json.dumps(result),flush=True)
        if args.source_controls:
            if digest(directory/args.label/'baseline.wav')!=digest(directory/args.label/'strength-1.wav'):
                raise ValueError('strength=1 differs from ordinary generation')
        if args.remix:
            score=directory/('remix-'+args.label)/'score.abc'
            execute([executable('yue2-transcribe'),'--model',args.transcription_model,'--audio',directory/'source.wav',
                     '--output',score,'--full','--window-seconds',30,'--overlap-seconds',20,'--lookahead-seconds',10,
                     '--device',args.device],score,{'source':item['sha256'],'model':digest(args.transcription_model),
                                                  'runtime':run['transcriber_runtime'],'device':args.device})
            base_ids=None
            for label,adapter,_ in variants:
                if label.startswith('strength-'):continue
                target=directory/('remix-'+args.label)/(label+'.wav');ids=target.with_suffix('.i32')
                command=[executable('yue2-generate'),'--model',args.model,'--vae',args.vae,'--tokenizer',args.tokenizer,
                         '--prompt',caption,'--abc',score,'--instrumental','--out',target,'--semantic-output',ids,
                         '--semantic-min-tokens',config['frames'],'--semantic-max-tokens',config['frames'],
                         '--ode-steps',args.steps,'--seed',args.seed,'--device',args.device]
                for fixed in args.fixed_lora:command+=['--lora',fixed]
                if args.instrumental_lora:command+=['--lora',args.instrumental_lora]
                if adapter:command+=['--lora',adapter]
                identity=dict(run,source=item['sha256'],start_frame=item.get('start_frame',round(config['offset']*25)),
                              adapter=digest(adapter) if adapter else None,abc=digest(score),
                              instrumental_lora=digest(args.instrumental_lora) if args.instrumental_lora else None)
                execute(command,target,identity)
                if base_ids is None:base_ids=digest(ids)
                elif digest(ids)!=base_ids:raise ValueError('NAR checkpoint changed AR codec tokens')
                results.append({'source':Path(item['source']).name,'variant':label,'workflow':'ABC-to-fresh-AR-tokens',
                                'window_offset_seconds':item.get('offset',config['offset']),
                                'tokens_sha256':digest(ids),'path':str(target.resolve())})
                print('remix',Path(item['source']).name,label,flush=True)
    output=args.root/(args.label+'.json')
    output.write_text(json.dumps(results,indent=2),encoding='utf-8')


if __name__=='__main__':main()

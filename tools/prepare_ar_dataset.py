#!/usr/bin/env python3
"""Prepare immutable semantic caches and scored AR examples from a job JSON.

Dataset files are read-only. Prepared files and structured progress live below
the job's output directory; cache identity includes audio/model/runtime bytes.
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import time
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

from yue2_training_audit import digest, native_runtime_identity
from ar_dataset import caption_metadata,score_from_events,window_starts


def save(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    # Windows readers can briefly deny replacement of the destination. Keep
    # the old complete JSON visible while retrying; never truncate it in place.
    for attempt in range(50):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt==49:raise
            time.sleep(.02)


def wait_for_inference(urls, timeout=1800):
    deadline=time.monotonic()+timeout
    while True:
        busy=False
        for url in urls:
            try:
                with urllib.request.urlopen(url.rstrip('/')+'/health',timeout=2) as response:health=json.load(response)
                busy |= health.get('busy',False) or health.get('queued',0)>0
            except OSError:pass
        if not busy:return
        if time.monotonic()>deadline:raise TimeoutError('Inference service remained busy')
        time.sleep(2)


def execute(command, log, cwd, urls):
    command=list(map(str,command));log=Path(log)
    if log.exists():raise FileExistsError(log)
    log.parent.mkdir(parents=True,exist_ok=True);wait_for_inference(urls);start=time.monotonic()
    with log.open('w',encoding='utf-8') as output:
        result=subprocess.run(command,cwd=cwd,stdout=output,stderr=subprocess.STDOUT)
    save(log.with_suffix('.command.json'),dict(command=command,returncode=result.returncode,seconds=time.monotonic()-start))
    if result.returncode:raise RuntimeError(log.read_text(encoding='utf-8')[-5000:])


def prepare_scores(config,out):
    index=json.loads((out/'semantic-index.json').read_text())
    if not index['completed']:raise ValueError('Semantic preparation is incomplete')
    seconds=config['preparation']['window_seconds'];frames=round(seconds*25)
    root=Path(config['workspace']);binary=Path(config['bin'])
    extension='.exe' if os.name=='nt' else ''
    score_tool=Path(config['score_binary']);transcriber=binary/('yue2-transcribe'+extension)
    prefix_tool=binary/('yue2-ar-train'+extension)
    normalizer_hash=digest(Path(__file__).with_name('ar_dataset.py'))
    tool_identity=dict(normalizer_sha256=normalizer_hash,score_runtime=native_runtime_identity(score_tool),
                       transcription_runtime=native_runtime_identity(transcriber),prefix_runtime=native_runtime_identity(prefix_tool),
                       transcription_model_sha256=digest(Path(config['models']['transcriber'])),
                       text_tokenizer_sha256=digest(Path(config['models']['tokenizer'])),precision_environment={'GGML_CUDA_TF32':'0'})
    example_directory=config['preparation'].get('example_directory','examples')
    dataset_name=config['preparation'].get('dataset_output','dataset.json')
    if Path(example_directory).name!=example_directory or Path(dataset_name).name!=dataset_name:raise ValueError('Prepared names must be local basenames')
    reusable={}
    if config['preparation'].get('reuse_examples_manifest'):
        previous=json.loads(Path(config['preparation']['reuse_examples_manifest']).read_text())
        if not previous['completed']:raise ValueError('Reusable transcription manifest is incomplete')
        reusable={(r['source_sha256'],r['start_frame'],r['frames']):r for r in previous['rows']}
    rows=[];planned=[]
    for source in index['rows']:
        metadata=caption_metadata(source['caption'])
        real_frames=min(source['frames'],math.floor(source['seconds']*25))
        for start in window_starts(real_frames,frames,metadata['bpm']):planned.append((source,metadata,start))
    for number,(source,metadata,start) in enumerate(planned):
        if config.get('stop_file') and Path(config['stop_file']).exists():return
        path=Path(source['path'])
        if digest(path.with_suffix('.txt'))!=source['caption_sha256']:raise ValueError('Caption changed: '+path.name)
        folder=out/example_directory/(source['sha256'][:12]+f'-{start:07d}-{frames}')
        folder.mkdir(parents=True,exist_ok=True);record_path=folder/'example.json'
        identity=dict(source_sha256=source['sha256'],caption_sha256=source['caption_sha256'],semantic_sha256=source['codes_sha256'],
                      start_frame=start,frames=frames,tools=tool_identity,trigger=config['trigger'])
        save(out/'preparation-progress.json',dict(stage='scores',current=number,total=len(planned),track=path.name,start_frame=start))
        if record_path.exists():
            record=json.loads(record_path.read_text())
            if record['identity']!=identity:raise ValueError('Prepared example identity differs: '+str(folder))
            if record['eligible']:
                for key in ['prefix','codes','score']:
                    if digest(Path(record[key]))!=record[key+'_sha256']:raise ValueError('Prepared example changed')
        else:
            record=dict(identity=identity,source=source['path'],track=path.name,source_sha256=source['sha256'],split=source['split'],
                        frames=frames,start_frame=start,directory=str(folder),bpm=metadata['bpm'],key=metadata['key'],
                        caption=config['trigger']+'. Instrumental music. '+metadata['style'].rstrip('.')+'.',
                        caption_policy='Sidecar style preserved; BPM/key encoded in score from absolute note timing and pitches',predict_end=False,eligible=False)
            codes=np.fromfile(source['codes'],dtype='<i4')[start:start+frames]
            if len(codes)!=frames:raise ValueError('Source code window out of bounds')
            code_path=folder/'codes.i32'
            if code_path.exists():
                if not np.array_equal(np.fromfile(code_path,dtype='<i4'),codes):raise ValueError('Prepared code window changed')
            else:codes.tofile(code_path)
            previous=reusable.get((source['sha256'],start,frames))
            if previous:
                if previous['identity']['tools']['transcription_runtime']!=tool_identity['transcription_runtime'] or previous['identity']['tools']['transcription_model_sha256']!=tool_identity['transcription_model_sha256']:
                    raise ValueError('Reusable transcription runtime/model differs')
                prior_folder=Path(previous['directory']);audio=prior_folder/'source.wav';events_path=prior_folder/'events.json'
                if not audio.exists() or not events_path.exists():raise ValueError('Reusable transcription files missing')
                record['transcription_reuse']=dict(directory=str(prior_folder),events_sha256=digest(events_path),audio_sha256=digest(audio))
            else:audio=folder/'source.wav';events_path=folder/'events.json'
            if not audio.exists():
                with sf.SoundFile(path) as stream:
                    stream.seek(round(start/25*stream.samplerate));samples=stream.read(round(seconds*stream.samplerate),dtype='float32',always_2d=True);rate=stream.samplerate
                if len(samples)!=round(seconds*rate):raise ValueError('Insufficient source audio')
                sf.write(audio,samples,rate,subtype='FLOAT')
            if not events_path.exists():
                execute([transcriber,'--model',config['models']['transcriber'],'--audio',audio,'--output',folder/'transcribed.abc','--events',events_path,'--full','--device',config.get('device','cuda')],folder/'transcribe.log',root,config.get('inference_urls',[]))
            try:
                events=json.loads(events_path.read_text())['events']
                abc,stats=score_from_events(events,seconds,metadata['bpm'],metadata['key'])
                if stats['max_rounding_seconds']>stats['max_expected_grid_error_seconds']+1e-5:raise ValueError('Timing quantization exceeds one grid unit')
                normalized=folder/'normalized.abc'
                if normalized.exists() and normalized.read_text()!=abc:raise ValueError('Normalized score changed')
                normalized.write_text(abc)
                used=folder/'used.abc'
                if not used.exists():execute([score_tool,'--input',normalized,'--output',used,'--instrumental','--midi-dir',folder/'midi'],folder/'score-validation.log',root,[])
                prefix=folder/'prefix.i32'
                if not prefix.exists():execute([prefix_tool,'--prefix-out',prefix,'--tokenizer',config['models']['tokenizer'],'--abc',used,'--instrumental','--prompt',record['caption']],folder/'prefix.log',root,[])
                record.update(eligible=True,score=str(used),score_sha256=digest(used),prefix=str(prefix),prefix_sha256=digest(prefix),
                              codes=str(code_path),codes_sha256=digest(code_path),prefix_tokens=prefix.stat().st_size//4,score_stats=stats,
                              sparse_score=stats['retained_monophonic_notes']<8)
            except (ValueError,RuntimeError) as error:record['error']=str(error)
            save(record_path,record)
        rows.append(record)
        save(out/dataset_name,dict(format='yue2-ar-prepared-dataset-v1',completed=number+1==len(planned),frames=frames,window_seconds=seconds,
                                   tools=tool_identity,rows=rows,manual_note_accuracy_verified=False))
        print('SCORE',number+1,len(planned),path.name,start,record['eligible'],record.get('prefix_tokens',record.get('error')),flush=True)
    for split in ['train','validation']:
        expected={source['sha256'] for source in config['sources'] if source['split']==split}
        actual={row['source_sha256'] for row in rows if row['split']==split and row['eligible']}
        if actual!=expected:raise ValueError('At least one whole recording has no eligible score in '+split)
    save(out/'preparation-progress.json',dict(stage='scores-complete',current=len(rows),total=len(rows)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--phase',choices=['semantic','scores'],required=True)
    args=parser.parse_args();config=json.loads(args.config.read_text(encoding='utf-8'))
    if config.get('format')!='yue2-ar-training-v1':raise ValueError('Unsupported job format')
    root=Path(config['workspace']);out=Path(config['output']);out.mkdir(parents=True,exist_ok=True)
    os.environ['GGML_CUDA_TF32']='0'
    if args.phase=='scores':
        prepare_scores(config,out);return
    binary=Path(config['bin'])/('yue2-semantic-tokenize.exe' if os.name=='nt' else 'yue2-semantic-tokenize')
    model=Path(config['models']['semantic'])
    runtime=native_runtime_identity(binary)
    fingerprint=dict(model_sha256=digest(model),runtime=runtime,device=config.get('device','cuda'),preprocessing='native full-track MERT/v9; 25Hz; audio reader/resampler pinned by binary')
    reusable={}
    for manifest_path in config.get('reuse_semantic_manifests',[]):
        manifest_path=Path(manifest_path);manifest=json.loads(manifest_path.read_text())
        if manifest.get('head')!=fingerprint['model_sha256'] or manifest.get('semantic_device')!=fingerprint['device']:
            raise ValueError('Reusable cache semantic model/device differs')
        if manifest.get('semantic_runtime',{}).get('precision_environment',{}).get('GGML_CUDA_TF32')!='0':
            raise ValueError('Reusable cache precision is not pinned to TF32=0')
        for row in manifest['tracks']:
            reusable[row['sha256']]=manifest_path.parent/row['sha256'][:12]/'full.i32'
    rows=[]
    for index,source in enumerate(config['sources']):
        if config.get('stop_file') and Path(config['stop_file']).exists():return
        path=Path(source['path']);sha=digest(path)
        if sha!=source['sha256']:raise ValueError('Source changed: '+path.name)
        folder=out/'semantic'/sha;folder.mkdir(parents=True,exist_ok=True)
        target=folder/'codes.i32';metadata=folder/'cache.json'
        identity=dict(source_sha256=sha,**fingerprint)
        save(out/'preparation-progress.json',dict(stage='semantic',current=index,total=len(config['sources']),track=path.name))
        if target.exists():
            if not metadata.exists():raise ValueError('Unfinished cache requires inspection: '+str(folder))
            previous=json.loads(metadata.read_text())
            if previous['identity']!=identity or digest(target)!=previous['codes_sha256']:raise ValueError('Cached semantic identity differs')
            provenance=previous['provenance']
        elif sha in reusable:
            previous=reusable[sha];array=np.fromfile(previous,dtype='<i4')
            if not len(array) or np.any((array<0)|(array>=32768)):raise ValueError('Invalid reusable token stream')
            # The original native caches are immutable full-track streams.
            import shutil
            temporary=target.with_suffix('.i32.partial');shutil.copy2(previous,temporary);temporary.rename(target)
            provenance=dict(reused_from=str(previous),reused_codes_sha256=digest(previous))
        else:
            temporary=target.with_suffix('.i32.partial')
            execute([binary,'--model',model,'--audio',path,'--out',temporary,'--device',fingerprint['device']],folder/'encode.log',root,config.get('inference_urls',[]))
            temporary.rename(target);provenance=dict(encoded=True)
        codes=np.fromfile(target,dtype='<i4');info=sf.info(path)
        if not len(codes) or np.any((codes<0)|(codes>=32768)) or abs(len(codes)-info.duration*25)>2:
            raise ValueError('Invalid frame count or codec IDs for '+path.name)
        record=dict(identity=identity,codes=str(target),codes_sha256=digest(target),frames=len(codes),provenance=provenance)
        if not metadata.exists():save(metadata,record)
        rows.append(dict(source,**{k:record[k] for k in ['codes','codes_sha256','frames','provenance']}))
        save(out/'semantic-index.json',dict(format='yue2-ar-semantic-index-v1',completed=index+1==len(config['sources']),fingerprint=fingerprint,rows=rows))
        print('SEMANTIC',index+1,len(config['sources']),path.name,flush=True)
    save(out/'preparation-progress.json',dict(stage='semantic-complete',current=len(rows),total=len(rows)))


if __name__=='__main__':main()

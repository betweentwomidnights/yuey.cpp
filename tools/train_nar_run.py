#!/usr/bin/env python3
"""Named native NAR calibration with frozen AR provenance and heldout selection."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
import gguf
from prepare_ar_dataset import save,wait_for_inference
from train_ar_run import grouped_loss,evaluation_history,native_metric_paths
from training_provenance import digest,digests,native_runtime_identity


class NativeNarProgress:
    def __init__(self,log,target,resumed=0,phase='fit'):
        self.log=Path(log);self.target=target;self.phase=phase
        self.offset=0;self.pending='';self.events=False;self.since=time.monotonic()
        self.values=dict(activity='Validating cached training data',step=resumed)

    def activity(self,value):
        if value!=self.values['activity']:
            self.since=time.monotonic()
            for key in ['phase_current','phase_total','phase_unit']:self.values.pop(key,None)
        self.values['activity']=value

    def read(self):
        with self.log.open(encoding='utf8',errors='replace') as stream:
            stream.seek(self.offset);text=stream.read();self.offset=stream.tell()
        lines=(self.pending+text).split('\n');self.pending=lines.pop()
        for line in lines:
            if line.startswith('TRAIN_EVENT '):
                event=json.loads(line[len('TRAIN_EVENT '):]);stage=event['stage'];self.events=True
                if stage=='cache-validation':label='Validating cached training data'
                elif stage=='model-loading':label='Loading native NAR model'
                elif stage=='evaluation':
                    step=event['step'];self.values['step']=step
                    label='Initial evaluation' if not step else 'Final evaluation' if step>=self.target else 'Evaluating checkpoint'
                elif stage=='checkpoint-save':
                    self.values['step']=event['step'];label='Saving checkpoint'
                else:continue
                self.activity(label)
                if stage in ['cache-validation','evaluation']:
                    self.values.update(phase_current=event['current'],phase_total=event['total'],
                                       phase_unit='files' if stage=='cache-validation' else 'windows')
            elif line.startswith('[yue2:train]') and not self.events:
                self.activity('Native preflight' if self.phase!='fit' else
                              'Loading native optimizer state' if self.values['step'] else 'Initial evaluation')
            elif line.startswith('{'):
                try:record=json.loads(line)
                except json.JSONDecodeError:continue
                if 'loss' in record and 'step' in record:
                    self.values['step']=record['step'];self.activity('Training')
                    if 'seconds' in record:self.values['last_update_seconds']=record['seconds']
                    if 'cache_load_seconds' in record:self.values['last_cache_load_seconds']=record['cache_load_seconds']
        return dict(self.values,activity_elapsed_seconds=time.monotonic()-self.since)


def preflight_sample(rows,cache_path,frames):
    """Read GGUF headers only, then probe one real sample at the corpus maximum."""
    largest=0;first=None
    for row in rows:
        reader=gguf.GGUFReader(str(cache_path(row)))
        tokens=int(reader.get_field('yue2.training.ar_tokens').contents())
        tensors={t.name:t for t in reader.tensors}
        if tokens<1 or tensors['clean'].data.size!=frames*64 or tensors['clean'].tensor_type!=gguf.GGMLQuantizationType.F32:
            raise ValueError('Invalid NAR preflight cache shape or target precision')
        for key in ['keys','values']:
            if tensors[key].data.size!=tokens*128*8*28 or tensors[key].tensor_type not in [gguf.GGMLQuantizationType.F16,gguf.GGMLQuantizationType.F32]:
                raise ValueError('Invalid NAR preflight conditioning shape or precision')
        largest=max(largest,tokens)
        if first is None and row['split']=='train':first=cache_path(row)
        del tensors,reader
    if first is None:raise ValueError('NAR preflight requires a training sample')
    return first,largest


def select_checkpoints(out,rows,config):
    validation=[r for r in rows if r['split']=='validation'];retention=[r for r in rows if r['split']=='retention'];metrics=[]
    if not validation:raise ValueError('Heldout selection requires validation')
    for path in native_metric_paths(out,'history.jsonl'):
        for line in path.read_text().splitlines():
            metric=json.loads(line)
            if 'heldout_eval' not in metric:continue
            metric['heldout_group_mean']=grouped_loss(metric['heldout_per_sample'],validation)
            metric['retention_group_mean']=grouped_loss(metric.get('retention_per_sample',[]),retention)
            metric['segment']=str(path.parent);metrics.append(metric)
    by_step={m['step']:m for m in metrics};ordered=[by_step[n] for n in sorted(by_step)]
    selected=min(ordered,key=lambda m:m['heldout_group_mean']);folder=Path(selected['segment']);n=selected['step']
    adapter=folder/f'step-{n}.gguf';state=folder/f'step-{n}.state.gguf' if n else None
    save(out/'selection.json',dict(method='Minimum recording-balanced heldout flow MSE including zero; neutral retention and listening separate',
        initial=ordered[0],selected=selected,final=ordered[-1],adapter=str(adapter),adapter_sha256=digest(adapter),
        state=str(state) if state else None,evaluations=ordered,subjective_verdict_pending=True))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--phase',choices=['probe','fit','post-probe','select'],required=True)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--segment',default='native')
    args=parser.parse_args();config=json.loads(args.config.read_text())
    if config['format']!='yue2-nar-training-v1':raise ValueError('Unsupported NAR job')
    if Path(args.segment).name!=args.segment:raise ValueError('Segment must be a directory name')
    os.environ['GGML_CUDA_TF32']='0'
    out=Path(config['output']);out.mkdir(parents=True,exist_ok=True)
    manifest=Path(config['training_dataset']);data=json.loads(manifest.read_text())
    if data['format']!='yue2-nar-prepared-dataset-v1' or not data['completed']:raise ValueError('NAR data is incomplete')
    rows=[r for r in data['rows'] if r.get('eligible',True)]
    def cache_path(row):
        path=Path(row['cache']);return path if path.is_absolute() else manifest.parent/path
    for row,value in zip(rows,digests(cache_path(r) for r in rows)):
        if value!=row['cache_sha256']:raise ValueError('NAR cache bytes changed')
    models=config['models'];training=config['training'];binary=Path(config['training_binary'])
    if training.get('weight_decay',.01)!=.01 or training.get('clip',1)!=1:
        raise ValueError('This native CLI pins NAR weight decay .01 and clipping 1')
    identity=dict(format='yue2-nar-run-v1',training=training,dataset_sha256=digest(manifest),
        models=dict(zip(['base','companion','ar'],digests(Path(models[key]) for key in ['base','companion','ar']))),
        runtime=native_runtime_identity(binary),
        device=config.get('device','cuda'),threads=config.get('threads',6))
    cache_identity=data.get('identity',{})
    for key,field in [('base','base_sha256'),('companion','companion_sha256'),('ar','ar_adapter_sha256')]:
        if field in cache_identity and cache_identity[field]!=identity['models'][key]:
            raise ValueError('Frozen conditioning belongs to different '+key+' weights')
    identity_path=out/'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text())!=identity:raise ValueError('Resolved NAR identity changed')
    save(identity_path,identity)
    if args.phase=='select':
        select_checkpoints(out,rows,config)
        return
    run_hash=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    command=list(map(str,[binary,'--model',models['base'],'--conditioning-lora',models['companion'],
        '--conditioning-lora',models['ar'],'--dataset',manifest,'--stream-samples','--shuffle-samples','--group-samples',
        '--f32-projections','--noise-schedule',training.get('noise_schedule','uniform'),'--rank',training['rank'],
        '--lr',training['learning_rate'],'--seed',training.get('seed',831001),'--identity',run_hash,
        '--device',config.get('device','cuda'),'--threads',config.get('threads',6)]))
    if args.phase=='probe' and training.get('probe_scope')=='first-at-maximum-context':
        first,capacity=preflight_sample(rows,cache_path,data['frames'])
        i=command.index('--dataset');del command[i:i+2]
        command.remove('--group-samples')
        command+=['--sample',str(first),'--probe-ar-slots',str(capacity)]
    if args.phase=='fit' and 'evaluation_scope' in training:
        command+=['--eval-scope',training['evaluation_scope']]
    if args.phase in ['fit','post-probe'] and 'cache_fingerprint' in training:
        command+=['--cache-fingerprint',training['cache_fingerprint']]
    target=training['steps']
    resumed=0
    if args.phase=='fit':
        native=out/args.segment
        if args.resume:
            reader=gguf.GGUFReader(str(args.resume));resumed=int(reader.get_field('yue2.training.step').contents())
            command+=['--resume',str(args.resume)]
        if resumed>=target:raise ValueError('Target budget already reached')
        command+=list(map(str,['--out',native,'--steps',target-resumed,'--save-every',training.get('checkpoint_every',100),
            '--eval-every',training.get('eval_every',100),'--stop-file',out/'stop.request']))
    else:
        native=out/args.phase
        if args.phase=='post-probe':
            selected=json.loads((out/'selection.json').read_text())
            if not selected['state']:raise ValueError('No learned checkpoint selected')
            command+=['--resume',selected['state']]
        command+=list(map(str,['--out',native,'--probe-only','--diagnostics',out/(args.phase+'-diagnostics.gguf')]))
    log=out/(args.phase+('-'+args.segment if args.phase=='fit' else '')+'.log')
    if log.exists() or native.exists():raise FileExistsError('NAR segment already exists')
    wait_for_inference(config.get('inference_urls',[]));start=time.monotonic();peak=0;step=0
    save(out/'run.json',dict(status='running',stage=args.phase,name=config.get('name','NAR'),identity_sha256=run_hash,output=str(native)))
    print('START NAR',args.phase,flush=True)
    with log.open('w') as stream:
        native_progress=NativeNarProgress(log,target,resumed,args.phase)
        save(out/'progress.json',dict(stage=args.phase,status='running',target_steps=target,**native_progress.values))
        process=subprocess.Popen(command,cwd=config['workspace'],stdout=stream,stderr=subprocess.STDOUT)
        try:
            try:save(out/'process.json',dict(pid=process.pid,phase=args.phase,native=str(native)))
            except OSError as error:print('Process status deferred:',error,flush=True)
            while process.poll() is None:
                try:
                    query=subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3)
                    if query.returncode==0:peak=max(peak,int(query.stdout.strip().splitlines()[0]))
                except (OSError,subprocess.TimeoutExpired,ValueError):pass
                try:
                    progress=native_progress.read()
                    history=native/'history.jsonl'
                    if history.exists():
                        lines=history.read_text().splitlines()
                        if lines:
                            latest=json.loads(lines[-1]);step=latest['step']
                            evaluations=evaluation_history(out,rows,'nar')
                            evaluated=any(m['step']==step for m in evaluations['evaluations'])
                            if not native_progress.events:
                                if step==0:activity='Initial evaluation'
                                elif step==target:activity='Saving final checkpoint' if evaluated else 'Final evaluation'
                                elif step % training.get('eval_every',100)==0:activity='Saving checkpoint' if evaluated else 'Evaluating checkpoint'
                                else:activity='Training'
                                native_progress.activity(activity);progress=native_progress.read()
                            progress.update(step=step,evaluation_history=evaluations)
                    save(out/'progress.json',dict(stage=args.phase,status='running',target_steps=target,
                         tokens_seen=progress['step']*data['frames'],elapsed_seconds=time.monotonic()-start,
                         observed_gpu_peak_mib=peak,**progress))
                except (OSError,ValueError):pass
                time.sleep(1)
        finally:
            # Monitoring failures must not abandon ownership of a live fitter.
            process.wait()
    record=dict(command=command,returncode=process.returncode,seconds=time.monotonic()-start,observed_gpu_peak_mib=peak)
    save(log.with_suffix('.command.json'),record)
    if process.returncode:
        save(out/'run.json',dict(status='failed',stage=args.phase,command=record));raise RuntimeError(log.read_text()[-5000:])
    if digest(manifest)!=identity['dataset_sha256']:raise ValueError('NAR manifest changed during fitting')
    for row,value in zip(rows,digests(cache_path(r) for r in rows)):
        if value!=row['cache_sha256']:raise ValueError('NAR cache changed during fitting')
    if args.phase=='fit':
        select_checkpoints(out,rows,config)
        step=json.loads((native/'history.jsonl').read_text().splitlines()[-1])['step']
        save(out/'progress.json',dict(stage='fit',status='completed' if step>=target else 'stopped',step=step,target_steps=target,
            tokens_seen=step*data['frames'],observed_gpu_peak_mib=peak,evaluation_history=evaluation_history(out,rows,'nar')))
    save(out/'run.json',dict(status='stopped' if (out/'stop.request').exists() else 'completed',stage=args.phase,
        elapsed_seconds=time.monotonic()-start,observed_gpu_peak_mib=peak,identity_sha256=run_hash,output=str(native)))
    print('END NAR',args.phase,record,flush=True)


if __name__=='__main__':main()

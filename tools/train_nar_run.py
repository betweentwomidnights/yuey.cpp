#!/usr/bin/env python3
"""Named native NAR calibration with frozen AR provenance and heldout selection."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
import gguf
from prepare_ar_dataset import save,wait_for_inference
from train_ar_run import grouped_loss
from yue2_training_audit import digest,native_runtime_identity


def select_checkpoints(out,rows,config):
    validation=[r for r in rows if r['split']=='validation'];retention=[r for r in rows if r['split']=='retention'];metrics=[]
    if not validation:raise ValueError('Heldout selection requires validation')
    for path in out.glob('native*/history.jsonl'):
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
    for row in rows:
        if digest(cache_path(row))!=row['cache_sha256']:raise ValueError('NAR cache bytes changed')
    models=config['models'];training=config['training'];binary=Path(config['training_binary'])
    if training.get('weight_decay',.01)!=.01 or training.get('clip',1)!=1:
        raise ValueError('This native CLI pins NAR weight decay .01 and clipping 1')
    identity=dict(format='yue2-nar-run-v1',training=training,dataset_sha256=digest(manifest),
        models={key:digest(Path(models[key])) for key in ['base','companion','ar']},runtime=native_runtime_identity(binary),
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
    target=training['steps']
    if args.phase=='fit':
        native=out/args.segment;resumed=0
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
                    history=native/'history.jsonl'
                    if history.exists():
                        lines=history.read_text().splitlines()
                        if lines:
                            latest=json.loads(lines[-1]);step=latest['step']
                            save(out/'progress.json',dict(stage=args.phase,status='running',step=step,target_steps=target,
                                tokens_seen=step*data['frames'],elapsed_seconds=time.monotonic()-start,observed_gpu_peak_mib=peak))
                except (OSError,json.JSONDecodeError):pass
                time.sleep(1)
        finally:
            # Monitoring failures must not abandon ownership of a live fitter.
            process.wait()
    record=dict(command=command,returncode=process.returncode,seconds=time.monotonic()-start,observed_gpu_peak_mib=peak)
    save(log.with_suffix('.command.json'),record)
    if process.returncode:
        save(out/'run.json',dict(status='failed',stage=args.phase,command=record));raise RuntimeError(log.read_text()[-5000:])
    if digest(manifest)!=identity['dataset_sha256']:raise ValueError('NAR manifest changed during fitting')
    for row in rows:
        if digest(cache_path(row))!=row['cache_sha256']:raise ValueError('NAR cache changed during fitting')
    if args.phase=='fit':
        select_checkpoints(out,rows,config)
        step=json.loads((native/'history.jsonl').read_text().splitlines()[-1])['step']
        save(out/'progress.json',dict(stage='fit',status='completed' if step>=target else 'stopped',step=step,target_steps=target,
            tokens_seen=step*data['frames'],observed_gpu_peak_mib=peak))
    save(out/'run.json',dict(status='stopped' if (out/'stop.request').exists() else 'completed',stage=args.phase,
        elapsed_seconds=time.monotonic()-start,observed_gpu_peak_mib=peak,identity_sha256=run_hash,output=str(native)))
    print('END NAR',args.phase,record,flush=True)


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Launch a named, reproducible native AR run and record UI-ready run artifacts.

This is a local orchestration layer, not an HTTP training service. Native GGUF
states retain optimizer and sampler ownership; source datasets stay read-only.
"""
import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path

import gguf
from prepare_ar_dataset import save,wait_for_inference
from yue2_training_audit import digest,digests,native_runtime_identity


class NativeProgress:
    """Track startup separately from optimizer updates, including resumed runs."""
    def __init__(self,log,resumed_step=0,eval_every=100,target_steps=None):
        self.log=Path(log);self.offset=0;self.pending=''
        self.eval_every=eval_every;self.target_steps=target_steps
        self.values=dict(activity='Loading native trainer',step=resumed_step)
        self.activity_since=time.monotonic()

    def read(self):
        with self.log.open(encoding='utf-8',errors='replace') as stream:
            stream.seek(self.offset);text=stream.read();self.offset=stream.tell()
        lines=(self.pending+text).split('\n');self.pending=lines.pop()
        for line in lines:
            before=self.values['activity']
            teacher=re.fullmatch(r'prepare_teacher=(\d+)/(\d+)',line.strip())
            if teacher:
                self.values.update(activity='Preparing frozen teacher distributions',
                                   teacher_current=int(teacher[1]),teacher_total=int(teacher[2]))
            elif line.startswith('initial_loss='):
                self.values['activity']='Evaluating training and validation windows'
            elif line.startswith('step='):
                step=int(line.split()[0].split('=')[1])
                activity='Final evaluation and checkpoint save' if step==self.target_steps else (
                    'Evaluating checkpoint' if step % self.eval_every == 0 else 'Training')
                self.values.update(step=step,activity=activity)
            if self.values['activity']!=before:self.activity_since=time.monotonic()
        result=dict(self.values)
        if 'evaluat' in result['activity'].lower():
            result['activity_elapsed_seconds']=time.monotonic()-self.activity_since
        return result


def grouped_loss(values,rows):
    sums={}
    if len(values)!=len(rows):raise ValueError('Metric/example count differs')
    for value,row in zip(values,rows):
        group=row.get('group',row.get('source_sha256',row.get('track')))
        sums.setdefault(group,[]).append(value)
    return sum(sum(items)/len(items) for items in sums.values())/len(sums) if sums else 0.


def native_metric_paths(out,filename):
    # Numeric ordering makes the latest resumed segment win at a repeated step.
    def segment_order(path):
        suffix=re.search(r'-(\d+)$',path.parent.name)
        return (int(suffix[1]) if suffix else 0,str(path))
    return sorted(Path(out).glob('native*/'+filename),key=segment_order)


def evaluation_history(out,rows,expert='ar'):
    """Read completed evaluation records without publishing a selected adapter."""
    validation=[r for r in rows if r['split']=='validation']
    retention=[r for r in rows if r['split']=='retention'];by_step={}
    filename='metrics.jsonl' if expert=='ar' else 'history.jsonl'
    for path in native_metric_paths(out,filename):
        for line in path.read_text().splitlines(keepends=True):
            if not line.endswith('\n'):continue # Child may be writing this record.
            metric=json.loads(line)
            if 'heldout_per_sample' not in metric:continue # NAR optimizer record.
            entry=dict(step=metric['step'],heldout_group_mean=grouped_loss(metric['heldout_per_sample'],validation),
                       retention_group_mean=grouped_loss(metric.get('retention_per_sample',[]),retention))
            if not all(math.isfinite(v) for v in (entry['heldout_group_mean'],entry['retention_group_mean'])):
                raise ValueError('Nonfinite evaluation metric')
            by_step[entry['step']]=entry
    evaluations=[by_step[step] for step in sorted(by_step)]
    return dict(evaluations=evaluations,best_so_far=min(evaluations,key=lambda m:m['heldout_group_mean']) if evaluations else None)


def select_checkpoints(out,rows,config):
    metrics=[]
    validation=[r for r in rows if r['split']=='validation']
    retention=[r for r in rows if r['split']=='retention']
    for path in native_metric_paths(out,'metrics.jsonl'):
        for line in path.read_text().splitlines():
            metric=json.loads(line)
            metric['heldout_group_mean']=grouped_loss(metric['heldout_per_sample'],validation)
            metric['retention_group_mean']=grouped_loss(metric['retention_per_sample'],retention)
            metric['segment']=str(path.parent);metrics.append(metric)
    by_step={m['step']:m for m in metrics};ordered=[by_step[s] for s in sorted(by_step)]
    selected=min(ordered,key=lambda metric:metric['heldout_group_mean'])
    folder=Path(selected['segment']);adapter=folder/f"step-{selected['step']}.gguf"
    # The preflight probe supplies the explicit step-zero candidate.
    if selected['step']==0:adapter=out/'probe/step-0.gguf'
    else:
        # Resume evaluates the loaded checkpoint before updating, but that
        # checkpoint belongs to its original segment, not the new metrics file.
        candidates=sorted(out.glob(f"native*/step-{selected['step']}.gguf"))
        if not candidates:raise FileNotFoundError(f"No adapter for evaluated step {selected['step']}")
        if len({digest(path) for path in candidates})!=1:
            raise ValueError('Conflicting adapters for evaluated optimizer step')
        if not adapter.exists():adapter=candidates[0];folder=adapter.parent
    state=folder/f"step-{selected['step']}.state.gguf" if selected['step'] else None
    save(out/'selection.json',dict(method='Minimum recording-balanced heldout semantic CE including zero; retention separate; listening required',
         initial=ordered[0],selected=selected,final=ordered[-1],adapter=str(adapter),adapter_sha256=digest(adapter),
         state=str(state) if state else None,evaluations=ordered,subjective_verdict_pending=True))
    captions=sorted({r['caption'] for r in rows if r['split']=='train' and r.get('kind','real')=='real'})
    save(out/'prompts.json',dict(version=1,source=dict(name=config.get('name','Style'),unique_prompts=len(captions)),dice=dict(instrumental=captions)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--family',choices=['lora','dora'],required=True)
    parser.add_argument('--phase',choices=['probe','fit','post-probe','select'],required=True)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--segment',default='native')
    args=parser.parse_args();config=json.loads(args.config.read_text())
    if config['format']!='yue2-ar-training-v1':raise ValueError('Unsupported training job')
    if Path(args.segment).name!=args.segment:raise ValueError('Segment must be a directory name')
    os.environ['GGML_CUDA_TF32']='0'
    root=Path(config['workspace']);out=Path(config['output'])/args.family;out.mkdir(exist_ok=True)
    dataset=Path(config['training_dataset']);data=json.loads(dataset.read_text())
    if not data['completed']:raise ValueError('Dataset is not ready')
    rows=[r for r in data['rows'] if r.get('eligible',True)]
    examples=[(row,key) for row in rows for key in ['prefix','codes']]
    for (row,key),value in zip(examples,digests(Path(row[key]) for row,key in examples)):
        if value!=row[key+'_sha256']:raise ValueError('Example content changed')
    training=config['training'];binary=Path(config['training_binary'])
    model=Path(config['models']['base']);companion=Path(config['models']['companion'])
    dataset_sha256,base_sha256,companion_sha256=digests([dataset,model,companion])
    identity=dict(format='yue2-ar-run-v1',family=args.family,training=training,dataset_sha256=dataset_sha256,
                  base_sha256=base_sha256,companion_sha256=companion_sha256,runtime=native_runtime_identity(binary))
    identity_path=out/'identity.json'
    if identity_path.exists():
        if json.loads(identity_path.read_text())!=identity:raise ValueError('Resolved run identity changed')
    else:save(identity_path,identity)
    if args.phase=='select':
        select_checkpoints(out,rows,config)
        return
    run_hash=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    prefix_slots=max(len(Path(row['prefix']).read_bytes())//4 for row in rows)
    target_steps=math.ceil(training['token_budget']/data['frames'])
    cmd=[str(binary),'--model',str(model),'--fixed-lora',str(companion),'--dataset',str(dataset),
         '--group-samples','--shuffle-samples','--max-prefix',str(prefix_slots),'--device',config.get('device','cuda'),
         '--rank',str(training['rank']),'--lr',str(training['learning_rate']),'--kl-weight',str(training['kl_weight']),
         '--weight-decay',str(training['weight_decay']),'--clip',str(training['clip']),'--seed',str(training.get('seed',831001)),
         '--identity',run_hash]
    if args.family=='dora':cmd+=['--dora']
    if args.phase=='fit' and 'evaluation_scope' in training:cmd+=['--eval-scope',training['evaluation_scope']]
    resumed_step=0
    if args.phase=='fit':
        native=out/args.segment;cmd+=['--out',str(native),'--save-every',str(training.get('checkpoint_every',100)),
                                    '--eval-every',str(training.get('eval_every',100)),'--stop-file',str(out/'stop.request')]
        if args.resume:
            state=gguf.GGUFReader(str(args.resume));resumed_step=int(state.get_field('yue2.training.step').contents())
            cmd+=['--resume',str(args.resume)]
        if resumed_step>=target_steps:raise ValueError('Target budget already reached')
        cmd+=['--steps',str(target_steps-resumed_step)]
    else:
        native=out/args.phase
        if args.phase=='post-probe':
            selected=json.loads((out/'selection.json').read_text())
            if selected['selected']['step']==0:raise ValueError('No learned checkpoint selected')
            cmd+=['--resume',selected['state']]
        cmd+=['--out',str(native),'--probe-only','--diagnostics',str(out/(args.phase+'-diagnostics.gguf'))]
    log=out/(args.phase+('-'+args.segment if args.phase=='fit' else '')+'.log')
    if log.exists() or native.exists():raise FileExistsError('Run segment already exists')
    wait_for_inference(config.get('inference_urls',[]));start=time.monotonic();peak=0;step=0
    save(out/'run.json',dict(format='yue2-ar-run-status-v1',name=config.get('name','Style')+' '+args.family,
                            family=args.family,stage=args.phase,status='running',target_steps=target_steps,
                            identity_sha256=run_hash,output=str(native)))
    print('START',args.family,args.phase,flush=True)
    with log.open('w') as stream:
        native_progress=NativeProgress(log,resumed_step,training.get('eval_every',100),target_steps)
        # Replace terminal progress from the previous segment before the native
        # teacher/evaluation warmup, which can take several minutes on resume.
        save(out/'progress.json',dict(stage=args.phase,status='running',family=args.family,
             target_steps=target_steps,**native_progress.values))
        process=subprocess.Popen(cmd,cwd=root,stdout=stream,stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                query=subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True)
                if query.returncode==0:peak=max(peak,int(query.stdout.strip().splitlines()[0]))
                history=native/'history.jsonl'
                try:
                    progress=native_progress.read()
                    if args.phase=='fit':progress['evaluation_history']=evaluation_history(out,rows)
                    if history.exists():
                        lines=history.read_text().splitlines()
                        if lines:
                            try:latest=json.loads(lines[-1])
                            except json.JSONDecodeError:latest=None  # Child may be appending this line.
                            if latest is not None:
                                step=latest.get('step',step)
                                progress.update(step=step,last_update_seconds=latest.get('seconds'))
                    save(out/'progress.json',dict(stage=args.phase,status='running',family=args.family,target_steps=target_steps,
                         tokens_seen=progress['step']*data['frames'],elapsed_seconds=time.monotonic()-start,observed_gpu_peak_mib=peak,**progress))
                except (OSError,ValueError) as error:
                    # Progress is advisory. A reader lock/disk error must not drop
                    # ownership of the still-running native process.
                    print('Progress update deferred:',error,flush=True)
                time.sleep(1)
        finally:process.wait()
    command_record=dict(command=cmd,returncode=process.returncode,seconds=time.monotonic()-start,observed_gpu_peak_mib=peak)
    save(log.with_suffix('.command.json'),command_record)
    if process.returncode:
        save(out/'run.json',dict(status='failed',stage=args.phase,family=args.family,command=command_record))
        raise RuntimeError(log.read_text()[-5000:])
    if digest(dataset)!=identity['dataset_sha256']:raise RuntimeError('Dataset manifest changed during the run')
    for (row,key),value in zip(examples,digests(Path(row[key]) for row,key in examples)):
        if value!=row[key+'_sha256']:raise RuntimeError('Example changed during the run')
    if args.phase=='fit':
        select_checkpoints(out,rows,config)
    save(out/'run.json',dict(status='stopped' if (out/'stop.request').exists() else 'completed',stage=args.phase,family=args.family,
                             elapsed_seconds=time.monotonic()-start,observed_gpu_peak_mib=peak,identity_sha256=run_hash,output=str(native)))
    if args.phase=='fit':
        final_step=json.loads((native/'history.jsonl').read_text().splitlines()[-1])['step']
        save(out/'progress.json',dict(stage='fit',status='completed' if final_step>=target_steps else 'stopped',family=args.family,
             step=final_step,target_steps=target_steps,tokens_seen=final_step*data['frames'],
             elapsed_seconds=time.monotonic()-start,observed_gpu_peak_mib=peak,evaluation_history=evaluation_history(out,rows)))
    print('END',args.family,args.phase,command_record,flush=True)


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Durable local two-expert training worker, launched by yue2-server.

Native tools own all optimization. Python prepares data, tracks processes and
publishes a matched pair. The immutable request/runtime and all checkpoints
remain in the named run; publishing never edits optimizer state.
"""
import argparse
import ctypes
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

from prepare_ar_dataset import save
from yue2_training_audit import digest


class Stopped(Exception): pass


def process_identity(pid=None):
    pid=os.getpid() if pid is None else pid
    if os.name=='nt':
        from ctypes.wintypes import FILETIME
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.GetCurrentProcess.restype=ctypes.c_void_p
        kernel.OpenProcess.argtypes=[ctypes.c_uint32,ctypes.c_int,ctypes.c_uint32]
        kernel.OpenProcess.restype=ctypes.c_void_p
        kernel.CloseHandle.argtypes=[ctypes.c_void_p]
        kernel.GetProcessTimes.argtypes=[ctypes.c_void_p,*([ctypes.POINTER(FILETIME)]*4)]
        times=[FILETIME() for _ in range(4)]
        process=kernel.GetCurrentProcess() if pid==os.getpid() else kernel.OpenProcess(0x1000,False,pid)
        try:
            if not process or not kernel.GetProcessTimes(process,*[ctypes.byref(t) for t in times]):
                raise RuntimeError('Cannot identify training worker creation time')
        finally:
            if process and pid!=os.getpid():kernel.CloseHandle(process)
        return str((times[0].dwHighDateTime<<32)|times[0].dwLowDateTime)
    text=Path(f'/proc/{pid}/stat').read_text()
    return text[text.rfind(')')+2:].split()[19]


def wait_for_owner(run,timeout=5):
    """Never touch the device before the server commits its ownership record."""
    owners={(os.getpid(),process_identity())}
    if os.name=='nt':
        # Windows venv python.exe is a redirector that stays alive while the
        # real interpreter runs. The server owns that launcher's handle/PID.
        parent=os.getppid();owners.add((parent,process_identity(parent)))
    deadline=time.monotonic()+timeout
    while True:
        try:
            owner=json.loads((Path(run)/'owner.json').read_text())
            if (owner.get('pid'),owner.get('identity')) in owners:return
        except (OSError,json.JSONDecodeError):pass
        if time.monotonic()>=deadline:raise RuntimeError('Training worker ownership was not committed')
        time.sleep(.02)


def protect_children():
    """On Windows, a crashed coordinator must not leave a fitter using VRAM."""
    if os.name != 'nt': return None
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateJobObjectW.argtypes=[ctypes.c_void_p,ctypes.c_wchar_p]
    kernel.CreateJobObjectW.restype=ctypes.c_void_p
    kernel.GetCurrentProcess.restype=ctypes.c_void_p
    kernel.SetInformationJobObject.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_uint32]
    kernel.AssignProcessToJobObject.argtypes=[ctypes.c_void_p,ctypes.c_void_p]
    job=kernel.CreateJobObjectW(None,None)
    # JOBOBJECT_EXTENDED_LIMIT_INFORMATION on the supported 64-bit runtime.
    if ctypes.sizeof(ctypes.c_void_p)!=8: raise RuntimeError('Training requires a 64-bit Python runtime')
    info=ctypes.create_string_buffer(144)
    ctypes.c_uint32.from_buffer(info,16).value=0x2000 # KILL_ON_JOB_CLOSE
    if not job or not kernel.SetInformationJobObject(job,9,info,144) or not kernel.AssignProcessToJobObject(job,kernel.GetCurrentProcess()):
        raise RuntimeError('Cannot establish training child-process ownership')
    # Keep the handle open until process exit; children inherit membership.
    return job


def settings(request):
    def number(key,default,minimum,maximum,integer=False):
        value=request.get(key,default)
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not minimum<=value<=maximum:
            raise ValueError(f'{key} must be between {minimum} and {maximum}')
        if integer and int(value)!=value:raise ValueError(key+' must be an integer')
        return int(value) if integer else value
    family=request.get('family','dora')
    if family not in ['lora','dora']:raise ValueError('AR family must be lora or dora')
    trigger=request.get('trigger',request['id']+'_style')
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',trigger):raise ValueError('Trigger must be one word using letters, numbers, hyphens or underscores')
    options=dict(window_seconds=number('window_seconds',30,2,60,True),rank=number('rank',32,1,128,True),
                ar_steps=number('ar_steps',800,1,10000,True),nar_steps=number('nar_steps',400,1,10000,True),
                ar_lr=number('ar_lr',2e-5,1e-7,.001),nar_lr=number('nar_lr',3e-5,1e-7,.001),
                kl_weight=number('kl_weight',.2,0,5),seed=number('seed',831001,0,2**32-1,True),
                eval_every=number('eval_every',100,1,1000,True),family=family,trigger=trigger)
    version=request.get('recipe_version',1)
    if type(version)!=int or version not in [1,2,3]:raise ValueError('Unsupported training recipe version')
    if version>=2:
        scope=request.get('evaluation_scope','heldout')
        if scope not in ['heldout','full']:raise ValueError('Evaluation scope must be heldout or full')
        options['evaluation_scope']=scope
    if version>=3:options['cache_fingerprint']='bytes-v2'
    return options


def inventory(folder,seconds,seed):
    import soundfile as sf
    from ar_dataset import caption_metadata
    rows=[];seen=set()
    for audio in sorted(Path(folder).iterdir()):
        if audio.suffix.lower() not in ['.wav','.flac']:continue
        sidecar=audio.with_suffix('.txt')
        if not sidecar.is_file():raise ValueError('Missing caption: '+sidecar.name)
        text=sidecar.read_text(encoding='utf-8-sig').strip();meta=caption_metadata(text);info=sf.info(audio)
        if not meta['style']:raise ValueError('Caption has no style description: '+audio.name)
        if info.duration<seconds:raise ValueError('Recording shorter than training window: '+audio.name)
        sha=digest(audio)
        if sha in seen:raise ValueError('Duplicate recording: '+audio.name)
        seen.add(sha)
        rows.append(dict(path=str(audio.resolve()),sha256=sha,caption=text,caption_sha256=digest(sidecar),
                         seconds=info.duration,bpm=meta['bpm'],key=meta['key'],split='train'))
    if len(rows)<2:raise ValueError('Provide at least two distinct WAV/FLAC recordings with TXT captions')
    # Recording-level deterministic holdouts; never split neighboring windows.
    import hashlib
    ordered=sorted(rows,key=lambda r:hashlib.sha256((str(seed)+r['sha256']).encode()).hexdigest())
    for row in ordered[:max(1,round(len(rows)*.15))]:row['split']='validation'
    return rows


def latest_state(folder):
    import gguf
    candidates=[]
    for path in Path(folder).glob('native*/step-*.state.gguf'):
        reader=gguf.GGUFReader(str(path));step=int(reader.get_field('yue2.training.step').contents());del reader
        candidates.append((step,path))
    return max(candidates,key=lambda x:x[0]) if candidates else (0,None)


class Workflow:
    def __init__(self,run):
        self.run=Path(run).resolve();self.runtime=json.loads((self.run/'runtime.json').read_text())
        self.request=json.loads((self.run/'request.json').read_text());self.options=settings(self.request)
        self.workspace=Path(self.runtime['workspace']).resolve();self.tools=self.workspace/'tools'
        self.data=self.run/'data';self.family=self.options['family'];self.data.mkdir(exist_ok=True)
        self.state=json.loads((self.run/'state.json').read_text()) if (self.run/'state.json').exists() else {}
        self.state.update(name=self.request.get('name',self.request['id']),settings=self.options)
        self.state.pop('error',None)

    def update(self,**values):
        self.state.update(values,updated_at=time.time());save(self.run/'state.json',self.state)

    def check_stop(self):
        if (self.run/'stop.request').exists():raise Stopped()

    def command(self,script,args,stage,progress=None):
        self.check_stop();self.update(status='running',stage=stage,progress={})
        def signature():
            try:
                stat=Path(progress).stat() if progress else None
                return (stat.st_mtime_ns,stat.st_size) if stat else None
            except OSError:return None
        previous_progress=signature()
        def read_progress():
            # A resumed helper may hash models before replacing its old status.
            # Do not display the previous segment's stopped/completed snapshot.
            if signature()!=previous_progress:
                try:self.update(progress=json.loads(Path(progress).read_text()))
                except (OSError,json.JSONDecodeError):pass
        number=1
        while (self.run/f'{stage}-{number}.log').exists():number+=1
        log=self.run/f'{stage}-{number}.log';cmd=[sys.executable,str(self.tools/script),*map(str,args)]
        save(log.with_suffix('.command.json'),dict(command=cmd))
        with log.open('w',encoding='utf-8') as output:
            process=subprocess.Popen(cmd,cwd=self.workspace,stdout=output,stderr=subprocess.STDOUT)
            self.update(child_pid=process.pid,log=str(log))
            try:
                while process.poll() is None:
                    read_progress()
                    time.sleep(.5)
            finally:process.wait()
        # Fast native stages can finish between monitor ticks. Read their
        # committed terminal progress before leaving the stage.
        read_progress()
        self.update(child_pid=None)
        self.check_stop()
        if process.returncode:raise RuntimeError(log.read_text(encoding='utf-8',errors='replace')[-5000:])

    def config(self):
        return json.loads((self.run/'ar-job.json').read_text())

    def prepare(self):
        o=self.options;runtime=self.runtime;models=runtime['models'];binary=Path(runtime['bin']).resolve()
        suffix='.exe' if os.name=='nt' else ''
        for name in ['yue2-ar-train','yue2-train','yue2-score','yue2-transcribe','yue2-semantic-tokenize']:
            if not (binary/(name+suffix)).is_file():raise ValueError('Training tool missing: '+name)
        vae_binary=Path(runtime.get('vae_binary',binary/('yue2-vae-encode'+suffix)))
        if not vae_binary.is_file():raise ValueError('Training tool missing: yue2-vae-encode')
        for name in ['base','semantic','transcriber','tokenizer','companion','vae']:
            if not Path(models[name]).is_file():raise ValueError('Training model missing: '+name)
        rows=inventory(self.request['dataset'],o['window_seconds'],o['seed'])
        # Runtime is immutable. Every helper verifies source/cache/model hashes
        # on resume; resolved preparation is retained for inspection.
        cfg=dict(format='yue2-ar-training-v1',workspace=str(self.workspace),output=str(self.data),
            dataset=self.request['dataset'],bin=str(binary),device=runtime.get('device','cuda'),
            name=self.state['name'],models=models,trigger=o['trigger'],sources=rows,inference_urls=[],
            score_binary=str(binary/('yue2-score'+suffix)),training_binary=str(binary/('yue2-ar-train'+suffix)),
            training_dataset=str(self.data/'dataset.json'),stop_file=str(self.run/'stop.request'),
            preparation=dict(window_seconds=o['window_seconds']),
            training=dict(rank=o['rank'],learning_rate=o['ar_lr'],kl_weight=o['kl_weight'],weight_decay=.01,clip=1,
                token_budget=o['ar_steps']*o['window_seconds']*25,seed=o['seed'],checkpoint_every=o['eval_every'],eval_every=o['eval_every']))
        if 'evaluation_scope' in o:cfg['training']['evaluation_scope']=o['evaluation_scope']
        path=self.run/'ar-job.json'
        if path.exists() and json.loads(path.read_text())!=cfg:raise ValueError('Resolved dataset/runtime changed; create a new run')
        save(path,cfg);self.update(dataset=rows,status='running')
        self.command('prepare_ar_dataset.py',['--config',path,'--phase','semantic'],'semantic',self.data/'preparation-progress.json')
        self.command('prepare_ar_dataset.py',['--config',path,'--phase','scores'],'scores',self.data/'preparation-progress.json')
        self.command('prepare_nar_targets.py',['--dataset',cfg['training_dataset'],'--vae',models['vae'],
                     '--binary',vae_binary,'--device',runtime.get('vae_device',runtime.get('device','cuda')),
                     '--core-frames',runtime.get('vae_core_frames',128),
                     '--threads',runtime.get('threads',6),'--out',self.run/'vae-targets','--stop-file',self.run/'stop.request'],
                     'vae-targets-native',self.run/'vae-targets/progress.json')
        data=json.loads(Path(cfg['training_dataset']).read_text())
        eligible=[r for r in data['rows'] if r.get('eligible',True)]
        self.update(status='ready',stage='dataset-review',examples=len(eligible),excluded=len(data['rows'])-len(eligible),
                    train_examples=sum(r['split']=='train' for r in eligible),validation_examples=sum(r['split']=='validation' for r in eligible),
                    estimated_conditioning_gib=sum((r['prefix_tokens']+r['frames']+1)*114688+r['frames']*256 for r in eligible)/2**30)

    def fit(self,expert,config,folder,budget):
        self.check_stop()
        script='train_ar_run.py' if expert=='ar' else 'train_nar_run.py'
        extra=['--family',self.family] if expert=='ar' else []
        if not (folder/'probe').exists():
            self.command(script,['--config',config,*extra,'--phase','probe'],expert+'-probe',folder/'progress.json')
        step,resume=latest_state(folder)
        if step>=budget:
            # Recover a completed native fit whose coordinator stopped before
            # writing checkpoint selection. This performs no extra updates.
            if not (folder/'selection.json').exists():
                self.command(script,['--config',config,*extra,'--phase','select'],expert+'-select')
            return
        number=1
        while (folder/f'native-{number}').exists():number+=1
        args=['--config',config,*extra,'--phase','fit','--segment',f'native-{number}']
        if resume:args+=['--resume',resume]
        self.command(script,args,expert+'-fit',folder/'progress.json')

    def train(self):
        if not (self.run/'vae-targets/index.json').exists():raise ValueError('Prepare and review the dataset first')
        o=self.options;cfg=self.config();ar_folder=self.data/self.family
        self.fit('ar',self.run/'ar-job.json',ar_folder,o['ar_steps'])
        selection=json.loads((ar_folder/'selection.json').read_text());ar=Path(selection['adapter'])
        if not selection['selected']['step']:raise ValueError('Heldout selection chose the base AR; review the run before acoustic calibration')
        self.update(ar_selection=selection)
        nar_data=self.run/'nar-conditioning';binary=Path(self.runtime['bin'])/('yue2-train.exe' if os.name=='nt' else 'yue2-train')
        if not (nar_data/'dataset.json').exists() or not json.loads((nar_data/'dataset.json').read_text()).get('completed'):
            self.command('prepare_nar_dataset.py',['--config',self.run/'ar-job.json','--targets',self.run/'vae-targets/index.json',
                '--ar-adapter',ar,'--binary',binary,'--out',nar_data,'--conditioning-device',self.runtime.get('device','cuda'),
                '--conditioning-batch','0','--stop-file',self.run/'stop.request'],'nar-conditioning',nar_data/'progress.json')
        nar_cfg=dict(format='yue2-nar-training-v1',workspace=str(self.workspace),output=str(self.run/'nar-run'),
            name=self.state['name'],training_dataset=str(nar_data/'dataset.json'),training_binary=str(binary),
            device=self.runtime.get('device','cuda'),threads=self.runtime.get('threads',6),inference_urls=[],
            models=dict(base=cfg['models']['base'],companion=cfg['models']['companion'],ar=str(ar)),
            training=dict(rank=o['rank'],learning_rate=o['nar_lr'],weight_decay=.01,clip=1,seed=o['seed'],noise_schedule='uniform',
                          steps=o['nar_steps'],checkpoint_every=o['eval_every'],eval_every=o['eval_every']))
        if 'evaluation_scope' in o:
            nar_cfg['training'].update(evaluation_scope=o['evaluation_scope'],probe_scope='first-at-maximum-context')
        if 'cache_fingerprint' in o:nar_cfg['training']['cache_fingerprint']=o['cache_fingerprint']
        nar_path=self.run/'nar-job.json'
        if nar_path.exists() and json.loads(nar_path.read_text())!=nar_cfg:raise ValueError('Frozen AR/NAR run changed')
        save(nar_path,nar_cfg)
        self.fit('nar',nar_path,self.run/'nar-run',o['nar_steps'])
        selected=json.loads((self.run/'nar-run/selection.json').read_text())
        self.update(status='review',stage='listening-and-publication',nar_selection=selected)

    def publish(self):
        if self.state.get('status') not in ['review','published']:raise ValueError('Complete both expert stages before publishing')
        from publish_training_adapter import publish
        result=publish(self.run,self.runtime,self.request,self.state)
        self.update(status='published',stage='published',export=result)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--action',choices=['prepare','train','resume','publish'],required=True);args=parser.parse_args()
    os.environ['GGML_CUDA_TF32']='0';workflow=None
    try:
        wait_for_owner(args.run)
        job=protect_children()
        workflow=Workflow(args.run)
        action=args.action
        if action=='resume':
            targets=args.run/'vae-targets/index.json'
            complete=targets.exists() and json.loads(targets.read_text()).get('completed',False)
            action='train' if complete else 'prepare'
        getattr(workflow,action)()
    except Stopped:
        if workflow:workflow.update(status='stopped',child_pid=None)
    except Exception as error:
        if workflow:workflow.update(status='failed',error=str(error),child_pid=None)
        else:save(args.run/'state.json',dict(status='failed',error=str(error)))
        traceback.print_exc();return 1
    return 0


if __name__=='__main__':sys.exit(main())

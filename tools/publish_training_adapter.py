"""Publish one matched AR/NAR style and its actual caption pool atomically."""
import json
from pathlib import Path
import gguf
import numpy as np
from prepare_ar_dataset import save
from yue2_training_audit import digest


def publish(run,runtime,request,state):
    run=Path(run);ar_record=state['ar_selection'];nar_record=state['nar_selection']
    sources=[Path(r['adapter']) for r in [ar_record,nar_record]]
    for path,record in zip(sources,[ar_record,nar_record]):
        if digest(path)!=record['adapter_sha256']:raise ValueError('Selected checkpoint changed')
    readers=[gguf.GGUFReader(str(p)) for p in sources]
    bases=[r.get_field('yue2.adapter.base_sha256').contents() for r in readers]
    ranks=[r.get_field('yue2.adapter.rank').contents() for r in readers]
    alphas=[r.get_field('yue2.adapter.alpha').contents() for r in readers]
    if bases[0]!=bases[1] or ranks[0]!=ranks[1] or alphas[0]!=alphas[1]:raise ValueError('Expert pair base/rank/scaling differs')
    if any('.nar_' in t.name for t in readers[0].tensors) or any('.nar_' not in t.name for t in readers[1].tensors):
        raise ValueError('Adapter contains targets outside its intended expert')
    folder=Path(runtime['models_dir']).resolve();folder.mkdir(parents=True,exist_ok=True)
    target=folder/(request['id']+'-F32-LoRA.gguf');sidecar=target.with_suffix('.training.json')
    if sidecar.exists():
        previous=json.loads(sidecar.read_text())
        if previous['run']!=str(run) or digest(target)!=previous['sha256']:raise ValueError('Published name already belongs to another artifact')
        return previous
    if target.exists():raise ValueError('Adapter name already exists; inspect the unregistered export')
    temporary=target.with_suffix('.gguf.partial')
    if temporary.exists():raise ValueError('Incomplete publication exists; inspect before retrying')
    family=readers[0].get_field('yue2.adapter.type').contents()
    writer=gguf.GGUFWriter(str(temporary),'yue2_lora',use_temp_file=True)
    writer.add_name(state['name'])
    for key,value in {'yue2.component':'generation-adapter','yue2.adapter.type':'mixed_lora_dora' if family=='dora' else 'lora',
        'yue2.adapter.base_sha256':bases[0],'yue2.training.trigger':state['settings']['trigger'],
        'yue2.training.ar_source_sha256':ar_record['adapter_sha256'],'yue2.training.nar_source_sha256':nar_record['adapter_sha256']}.items():writer.add_string(key,value)
    for key,field in readers[0].fields.items():
        if key.startswith('yue2.adapter.dora.'):
            writer.add_key_value(key,field.contents(),field.types[0])
    writer.add_uint32('yue2.adapter.rank',ranks[0]);writer.add_float32('yue2.adapter.alpha',alphas[0])
    seen=set();targets=set()
    for reader in readers:
        for tensor in reader.tensors:
            if tensor.name in seen or tensor.tensor_type!=gguf.GGMLQuantizationType.F32:raise ValueError('Duplicate or unsupported adapter tensor')
            seen.add(tensor.name);targets.add(tensor.name.rsplit('.',1)[0]);writer.add_tensor(tensor.name,np.ascontiguousarray(tensor.data))
    writer.add_uint32('yue2.adapter.target_count',len(targets));writer.add_uint32('yue2.adapter.replacement_count',0)
    writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file();writer.close()
    # Verify the pair byte-for-byte before making it visible to /loras.
    merged=gguf.GGUFReader(str(temporary));actual={t.name:t.data.tobytes() for t in merged.tensors}
    if any(actual[t.name]!=t.data.tobytes() for r in readers for t in r.tensors):raise ValueError('Export tensor verification failed')
    del merged
    prompts=json.loads((run/'data'/state['settings']['family']/'prompts.json').read_text())
    result=dict(format='yue2-trained-style-v1',name=state['name'],run=str(run),adapter=str(target),sha256=digest(temporary),
        trigger=state['settings']['trigger'],instrumental=True,prompts=prompts['dice']['instrumental'],base_sha256=bases[0],
        ar_sha256=ar_record['adapter_sha256'],nar_sha256=nar_record['adapter_sha256'],
        companion=dict(name=Path(runtime['models']['companion']).name,path=runtime['models']['companion'],sha256=digest(Path(runtime['models']['companion'])),strength=1,order='before-style'),
        settings=state['settings'],selection='Recording-balanced heldout semantic CE and flow MSE; user publishes after review')
    temporary.rename(target)
    save(sidecar,result)
    return result

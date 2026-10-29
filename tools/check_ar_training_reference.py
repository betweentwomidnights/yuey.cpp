#!/usr/bin/env python3
"""Independent CPU/PyTorch semantic AR forward, gradient and AdamW oracle.

Reads GGUF data only; no downloaded Python model code. Frozen BF16 weights
round through F16, matching the native training loader. Includes causal GQA,
full prefix attention and target-only semantic/EOS cross-entropy.
"""
import argparse
import json
from pathlib import Path

import gguf
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def arrays(path):
    reader = gguf.GGUFReader(str(path))
    return reader, {t.name: t for t in reader.tensors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--diagnostics', type=Path, required=True)
    parser.add_argument('--fixed-lora', type=Path, action='append', default=[])
    parser.add_argument('--updated-adapter', type=Path)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    model_reader, source = arrays(args.model)
    diag_reader, diagnostic = arrays(args.diagnostics)
    fixed = [arrays(p) for p in args.fixed_lora]

    def convert(t, through_half=False):
        value = gguf.dequantize(t.data, t.tensor_type)
        if through_half and t.tensor_type == gguf.GGMLQuantizationType.BF16:
            value = value.astype(np.float16).astype(np.float32)
        return torch.from_numpy(np.array(value, dtype=np.float32, copy=True))

    weights = {}
    include_train = True
    def weight(name):
        if name not in weights:
            weights[name] = convert(source[name], True)
        return weights[name]

    train = {name: convert(t).requires_grad_() for name, t in diagnostic.items()
             if name.endswith(('.lora_A', '.lora_B', '.dora_magnitude'))}
    fixed_weights = {}
    frozen_directions = {}
    def linear(x, name):
        out = F.linear(x, weight(name + '.weight'))
        for index, (reader, tensors) in enumerate(fixed):
            if name + '.lora_A' in tensors:
                key = (index, name)
                if key not in fixed_weights:
                    fixed_weights[key] = (convert(tensors[name + '.lora_A']),
                                          convert(tensors[name + '.lora_B']))
                a, b = fixed_weights[key]
                scale = float(reader.get_field('yue2.adapter.alpha').contents()) / int(
                    reader.get_field('yue2.adapter.rank').contents())
                out = out + scale * F.linear(F.linear(x, a), b)
            if name + '.weight.replacement' in tensors:
                raise ValueError('oracle fixed replacements are unsupported')
        if include_train and name + '.lora_A' in train:
            out = out + F.linear(F.linear(x, train[name + '.lora_A']), train[name + '.lora_B'])
            if name + '.dora_magnitude' in train:
                if name not in frozen_directions:
                    effective=weight(name+'.weight').clone()
                    for index,(reader,tensors) in enumerate(fixed):
                        if name+'.lora_A' in tensors:
                            a,b=fixed_weights[(index,name)]
                            scale=float(reader.get_field('yue2.adapter.alpha').contents())/int(reader.get_field('yue2.adapter.rank').contents())
                            effective=effective+scale*(b@a)
                    frozen_directions[name]=effective
                direction=frozen_directions[name]+train[name+'.lora_B']@train[name+'.lora_A']
                denominator=(direction.square().sum(-1).clamp_min(0)+1e-12).sqrt()
                out=out*(train[name+'.dora_magnitude'].reshape(-1)/denominator)
        return out

    def norm(x, name):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * weight(name)

    ids = convert(diagnostic['token_ids']).reshape(-1).long()
    n = len(ids)
    prefix = int(diag_reader.get_field('yue2.training.ar_slots').contents())
    actual_field = diag_reader.get_field('yue2.training.actual_ar_tokens')
    actual_prefix = int(actual_field.contents()) if actual_field is not None else prefix
    pad = prefix - actual_prefix
    assert pad >= 0 and (not pad or torch.all(ids[:pad] == 151643))
    labels = convert(diagnostic['target']).reshape(-1, 32769)
    assert torch.all(labels.sum(-1) == 1) and torch.all((labels == 0) | (labels == 1))
    labels = labels.argmax(-1)
    end = bool(diag_reader.get_field('yue2.training.predict_end').contents())
    assert torch.equal(labels[:n-prefix], ids[prefix:] - 151852)
    if end:
        assert labels[-1] == 0
    x = F.embedding(ids, weight('model.embed_tokens.weight'))
    angle = (torch.arange(n)-pad).clamp_min(0)[:, None] * (1e6 ** (-torch.arange(0, 128, 2).float() / 128))[None]
    cos, sin = angle.cos()[:, None], angle.sin()[:, None]
    def rope(t):
        a, b = t.chunk(2, dim=-1)
        return torch.cat([a*cos-b*sin, a*sin+b*cos], dim=-1)
    mask = torch.ones(n, n, dtype=torch.bool).triu(1)
    mask[pad:,:pad] = True
    def block(x, layer):
        p = f'model.layers.{layer}.'
        a = p + 'self_attn.'
        h = norm(x, p + 'input_layernorm.weight')
        q = rope(norm(linear(h, a+'q_proj').reshape(n,16,128), a+'q_norm.weight')).transpose(0,1)
        k = rope(norm(linear(h, a+'k_proj').reshape(n,8,128), a+'k_norm.weight')).transpose(0,1)
        v = linear(h, a+'v_proj').reshape(n,8,128).transpose(0,1)
        k = k.repeat_interleave(2, dim=0)
        v = v.repeat_interleave(2, dim=0)
        scores = ((q @ k.transpose(1,2)) / np.sqrt(128)).masked_fill(mask, -torch.inf)
        attended = (scores.softmax(-1) @ v).transpose(0,1).reshape(n,2048)
        x = x + linear(attended, a+'o_proj')
        h = norm(x, p+'post_attention_layernorm.weight')
        return x + linear(F.silu(linear(h,p+'mlp.gate_proj')) * linear(h,p+'mlp.up_proj'),
                          p+'mlp.down_proj')
    for layer in range(28):
        x = checkpoint(lambda h, layer=layer: block(h, layer), x, use_reentrant=False)
    h = norm(x, 'model.norm.weight')[prefix-1:]
    logits = F.linear(h, weight('lm_head.weight')[151852:184621])
    task_loss = F.cross_entropy(logits, labels)
    loss = task_loss
    kl_value = 0.
    teacher_error = 0.
    kl_field = diag_reader.get_field('yue2.training.active_kl_weight')
    kl_weight = float(kl_field.contents()) if kl_field is not None else 0.
    if kl_weight > 0:
        teacher = convert(diagnostic['teacher_probabilities']).reshape(-1,32769)
        assert torch.all(teacher>=0) and torch.allclose(teacher.sum(-1),torch.ones(len(labels)),atol=1e-5)
        kl = F.kl_div(F.log_softmax(logits,-1),teacher,reduction='batchmean')
        loss = task_loss + kl_weight*kl
        kl_value=float(kl.detach())
        # Independently verify the cached teacher against the common model.
        with torch.no_grad():
            include_train=False
            tx=F.embedding(ids,weight('model.embed_tokens.weight'))
            for layer in range(28):tx=block(tx,layer)
            th=norm(tx,'model.norm.weight')[prefix-1:]
            reference_teacher=F.linear(th,weight('lm_head.weight')[151852:184621]).softmax(-1)
            teacher_error=float((reference_teacher-teacher).double().norm()/reference_teacher.double().norm())
            include_train=True
    loss.backward()
    native = convert(diagnostic['logits']).reshape(-1,32769)
    logit_error = float((logits.detach()-native).double().norm() / native.double().norm())
    native_loss=F.cross_entropy(native,labels)
    if kl_weight>0:native_loss=native_loss+kl_weight*F.kl_div(F.log_softmax(native,-1),teacher,reduction='batchmean')
    result = dict(reference_loss=float(loss.detach()), native_loss=float(native_loss),
                  reference_cross_entropy=float(task_loss.detach()),reference_teacher_kl=kl_value,
                  teacher_distribution_relative_l2=teacher_error,active_kl_weight=kl_weight,
                  logits_relative_l2=logit_error, target_tokens=len(labels), predict_end=end,
                  actual_prefix_tokens=actual_prefix, prefix_padding_tokens=pad,
                  gradients={})
    errors = references = 0.
    for name, parameter in train.items():
        expected = parameter.grad.detach().double()
        actual = convert(diagnostic[name+'.grad']).reshape_as(expected).double()
        error = float((actual-expected).norm().square())
        reference = float(expected.norm().square())
        errors += error; references += reference
        result['gradients'][name] = dict(relative_l2=np.sqrt(error/max(reference,1e-24)),
                                       reference_norm=np.sqrt(reference))
    result['gradient_relative_l2'] = np.sqrt(errors/max(references,1e-24))
    result['max_significant_gradient_relative_l2'] = max(
        (v['relative_l2'] for v in result['gradients'].values() if v['reference_norm'] > 1e-7), default=0.)
    result['passed'] = bool(logit_error < .02 and teacher_error < .002 and result['gradient_relative_l2'] < .05 and
                            result['max_significant_gradient_relative_l2'] < .05)
    if args.updated_adapter:
        assert int(diag_reader.get_field('yue2.training.step').contents()) == 0, 'first AdamW step only'
        optimizer = torch.optim.AdamW(list(train.values()),
            lr=float(diag_reader.get_field('yue2.training.learning_rate').contents()),
            weight_decay=float(diag_reader.get_field('yue2.training.weight_decay').contents()),
            betas=(.9,.999),eps=1e-8,foreach=False)
        for name, parameter in train.items():
            parameter.grad = convert(diagnostic[name+'.grad'])
        clip = float(diag_reader.get_field('yue2.training.gradient_clip').contents())
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(list(train.values()),clip,foreach=False)
        optimizer.step()
        updated_reader, updated = arrays(args.updated_adapter)
        error=factor_error=magnitude_units=0.
        for name,p in train.items():
            difference=(p.detach()-convert(updated[name])).abs();error=max(error,float(difference.max()))
            if name.endswith('.dora_magnitude'):
                unit=torch.finfo(torch.float32).eps*p.detach().abs().clamp_min(1.)
                magnitude_units=max(magnitude_units,float((difference/unit).max()))
            else:factor_error=max(factor_error,float(difference.max()))
        result['adamw_max_absolute_error'] = error
        result['adamw_factor_max_absolute_error']=factor_error
        result['adamw_magnitude_max_f32_rounding_units']=magnitude_units
        result['passed'] = result['passed'] and factor_error<2e-7 and magnitude_units<=2.
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='gradients'},indent=2),flush=True)
    if not result['passed']:
        raise SystemExit('AR forward/gradient oracle failed')


if __name__ == '__main__':
    main()

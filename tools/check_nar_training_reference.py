#!/usr/bin/env python3
"""Independent PyTorch forward/backward oracle for yue2-train --probe-only.

Loads only the NAR expert from the unquantized generation GGUF. Converts BF16
storage through F16, matching the native frozen-weight training policy. No
downloaded model code is imported. Memory is bounded with block checkpointing.
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
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--fixed-lora", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--updated-adapter", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    model_reader, source = arrays(args.model)
    diag_reader, diagnostic = arrays(args.diagnostics)
    fixed_readers, fixed = [], []
    for path in args.fixed_lora:
        reader, tensors = arrays(path)
        fixed_readers.append(reader)
        fixed.append(tensors)
    weights = {}
    def convert(tensor, through_half=False):
        result = gguf.dequantize(tensor.data, tensor.tensor_type)
        if through_half and tensor.tensor_type == gguf.GGMLQuantizationType.BF16:
            result = result.astype(np.float16).astype(np.float32)
        return torch.from_numpy(np.array(result, dtype=np.float32, copy=True))
    def weight(name):
        if name not in weights:
            weights[name] = convert(source[name], True)
        return weights[name]
    train = {name: convert(tensor).requires_grad_() for name, tensor in diagnostic.items()
             if name.endswith((".lora_A", ".lora_B", ".dora_magnitude"))}
    def linear(x, name):
        base = F.linear(x, weight(name + ".weight"))
        result = base
        for tensors in fixed:
            replacement = name + ".weight.replacement"
            if replacement in tensors:
                result = result + F.linear(x, convert(tensors[replacement])) - base
            if name + ".lora_A" in tensors:
                result = result + F.linear(F.linear(x, convert(tensors[name + ".lora_A"])),
                                           convert(tensors[name + ".lora_B"]))
        if name + ".lora_A" in train:
            result = result + F.linear(F.linear(x, train[name + ".lora_A"]), train[name + ".lora_B"])
            if name + ".dora_magnitude" in train:
                # Independent dense direction/norm, rather than the native
                # low-rank norm identity. Full denominator gradients match SA3.
                direction = weight(name + ".weight") + train[name + ".lora_B"] @ train[name + ".lora_A"]
                denominator = (direction.square().sum(dim=1).clamp_min(0) + 1e-12).sqrt()
                result = result * (train[name + ".dora_magnitude"].reshape(-1) / denominator)
        return result
    def bias(x, name):
        b = weight(name + ".bias")
        result = b
        for tensors in fixed:
            key = name + ".bias.replacement"
            if key in tensors:
                result = result + convert(tensors[key]) - b
        return linear(x, name) + result
    def norm(x, name):
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-6) * weight(name)
    mixed = convert(diagnostic["mixed"]).reshape(-1, 64)
    target = convert(diagnostic["target"]).reshape(-1, 64)
    native_velocity = convert(diagnostic["velocity"]).reshape(-1, 64)
    sigma = float(diag_reader.get_field("yue2.training.sigma").contents())
    count = int(diag_reader.get_field("yue2.training.actual_ar_tokens").contents())
    n = mixed.shape[0]
    frequencies = torch.exp(-np.log(10000.0) * torch.arange(128) / 128)
    time = torch.cat([(sigma * frequencies).cos(), (sigma * frequencies).sin()])
    x = bias(mixed, "vae2llm") + bias(F.silu(bias(time, "time_embedder.mlp.0")), "time_embedder.mlp.2")
    x = x + weight("latent_pos_embed.pe")[:n]
    positions = torch.arange(count, count + n)
    angle = positions[:, None] * (1e6 ** (-torch.arange(0, 128, 2).float() / 128))[None]
    cos, sin = angle.cos()[:, None], angle.sin()[:, None]
    def rope(x):
        first, second = x.chunk(2, dim=-1)
        return torch.cat([first*cos-second*sin, first*sin+second*cos], dim=-1)
    keys = [convert(diagnostic[f"key.{i}"]).reshape(8,-1,128) for i in range(28)]
    values = [convert(diagnostic[f"value.{i}"]).reshape(8,-1,128) for i in range(28)]
    def block(x, layer):
        prefix = f"model.layers.{layer}."
        a = prefix + "nar_self_attn."
        h = norm(x, prefix + "nar_input_layernorm.weight")
        q = rope(norm(linear(h, a+"q_proj").reshape(n,16,128), a+"q_norm.weight")).transpose(0,1)
        k = rope(norm(linear(h, a+"k_proj").reshape(n,8,128), a+"k_norm.weight")).transpose(0,1)
        v = linear(h, a+"v_proj").reshape(n,8,128).transpose(0,1)
        k = torch.cat([keys[layer],k],dim=1).repeat_interleave(2,dim=0)
        v = torch.cat([values[layer],v],dim=1).repeat_interleave(2,dim=0)
        scores = (q @ k.transpose(1,2)) / np.sqrt(128)
        if count < keys[layer].shape[1]:
            mask = torch.zeros(scores.shape[-1]); mask[count:keys[layer].shape[1]] = -torch.inf
            scores = scores + mask
        attended = (scores.softmax(-1) @ v).transpose(0,1).reshape(n,2048)
        x = x + linear(attended,a+"o_proj")
        h = norm(x,prefix+"nar_pre_mlp_layernorm.weight")
        return x + linear(F.silu(linear(h,prefix+"nar_mlp.gate_proj"))*linear(h,prefix+"nar_mlp.up_proj"),
                          prefix+"nar_mlp.down_proj")
    for layer in range(28):
        x = checkpoint(lambda hidden, layer=layer: block(hidden, layer), x, use_reentrant=False)
    velocity = bias(norm(x,"model.norm.weight"),"llm2vae")[1:-1]
    loss = F.mse_loss(velocity,target)
    loss.backward()
    difference = (velocity.detach()-native_velocity).double()
    relative = float(difference.norm() / native_velocity.double().norm().clamp_min(1e-12))
    result = {"reference_loss": float(loss.detach()), "native_loss": float(F.mse_loss(native_velocity,target)),
              "velocity_relative_l2": relative, "gradients": {}}
    squared_error = squared_reference = 0.
    for name, p in train.items():
        actual = convert(diagnostic[name+".grad"]).reshape(-1).double()
        expected = p.grad.detach().reshape(-1).double()
        error = (actual-expected).norm().square().item()
        ref = expected.norm().square().item()
        squared_error += error; squared_reference += ref
        result["gradients"][name] = {"relative_l2": float(np.sqrt(error/max(ref,1e-24))),
                                    "reference_norm": float(np.sqrt(ref))}
    result["gradient_relative_l2"] = float(np.sqrt(squared_error/max(squared_reference,1e-24)))
    significant=[v['relative_l2'] for v in result['gradients'].values() if v['reference_norm']>1e-7]
    result['max_significant_gradient_relative_l2']=max(significant,default=0.)
    result["passed"] = (relative < .02 and result["gradient_relative_l2"] < .05
                        and result['max_significant_gradient_relative_l2'] < .05)
    if args.updated_adapter:
        # Use native gradients to isolate optimizer arithmetic from the small
        # forward/backward precision differences reported above.
        optimizer=torch.optim.AdamW(list(train.values()),
            lr=float(diag_reader.get_field('yue2.training.learning_rate').contents()),
            weight_decay=float(diag_reader.get_field('yue2.training.weight_decay').contents()),
            betas=(.9,.999),eps=1e-8,foreach=False)
        for name,p in train.items():p.grad=convert(diagnostic[name+'.grad'])
        clip=float(diag_reader.get_field('yue2.training.gradient_clip').contents())
        if clip>0:torch.nn.utils.clip_grad_norm_(list(train.values()),clip,foreach=False)
        optimizer.step()
        updated_reader,updated=arrays(args.updated_adapter)
        largest=0.
        largest_factor=0.
        magnitude_rounding_units=0.
        for name,p in train.items():
            difference=(p.detach()-convert(updated[name])).abs()
            largest=max(largest,float(difference.max()))
            if name.endswith('.dora_magnitude'):
                # Magnitudes can be much larger than A/B. The native fused
                # subtraction and torch's separate decay/subtraction can
                # differ by an F32 rounding unit. Keep factor tolerance strict.
                unit=torch.finfo(torch.float32).eps*p.detach().abs().clamp_min(1.)
                magnitude_rounding_units=max(magnitude_rounding_units,float((difference/unit).max()))
            else:largest_factor=max(largest_factor,float(difference.max()))
        result['adamw_max_absolute_error']=largest
        result['adamw_factor_max_absolute_error']=largest_factor
        result['adamw_magnitude_max_f32_rounding_units']=magnitude_rounding_units
        result['passed']=result['passed'] and largest_factor<2e-7 and magnitude_rounding_units<=2.
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="gradients"},indent=2),flush=True)
    if not result["passed"]:
        raise SystemExit("NAR forward/gradient reference tolerance exceeded")


if __name__ == "__main__": main()

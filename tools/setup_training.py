#!/usr/bin/env python3
"""Resolve installed training tools/models into an explicit local server config."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bin',type=Path,required=True)
    parser.add_argument('--models-dir',type=Path,default=Path('models'))
    parser.add_argument('--run-root',type=Path,default=Path('outputs/training'))
    parser.add_argument('--out',type=Path,default=Path('training-runtime.json'))
    args=parser.parse_args()
    if sys.platform not in ['win32','linux']:parser.error('Local training ownership currently supports Windows and Linux')
    if args.out.exists():parser.error('Output exists; choose a new config path')
    missing=[name for name in ['numpy','soundfile','scipy','gguf','torch'] if importlib.util.find_spec(name) is None]
    if missing:parser.error('Missing preprocessing dependencies: '+', '.join(missing)+'; install tools/requirements-training.txt')
    def model(pattern):
        paths=list(args.models_dir.rglob(pattern))
        if len(paths)!=1:parser.error('Expected one installed '+pattern+'; found '+str(len(paths)))
        return str(paths[0].resolve())
    suffix='.exe' if os.name=='nt' else ''
    for tool in ['yue2-ar-train','yue2-train','yue2-score','yue2-transcribe','yue2-semantic-tokenize']:
        if not (args.bin/(tool+suffix)).is_file():parser.error('Missing native tool: '+tool)
    config=dict(workspace=str(Path(__file__).resolve().parents[1]),python=sys.executable,
        bin=str(args.bin.resolve()),run_root=str(args.run_root.resolve()),models_dir=str(args.models_dir.resolve()),
        device='cuda',threads=6,models=dict(base=model('yue2-3.6B-v1.0-BF16.gguf'),
        vae=model('yue2-vae-v1.0-F16.gguf'),semantic=model('yue2-semantic-tokenizer-0.7B-v1.0-F16.gguf'),
        transcriber=model('sheetsage2-mert2-f16.gguf'),tokenizer=model('yue2-qwen.tiktoken'),
        companion=model('yue2-instrumental-cot-full-v1.0-F16-LoRA.gguf')))
    args.out.write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
    print('Start yue2-server with --training-config '+str(args.out.resolve()))


if __name__=='__main__':main()

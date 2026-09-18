#!/usr/bin/env python3
"""Download a complete yuey.cpp GGUF set from Hugging Face into ./models."""

import argparse
import importlib.util
import os
import sys

from model_artifacts import PROFILES, PUBLISHED_ENCODINGS, build_download_plan

DEFAULT_NAMESPACE = "thepatch"


def print_transfer_info():
    try:
        import huggingface_hub
        from huggingface_hub import constants
    except Exception:
        return
    xet_installed = importlib.util.find_spec("hf_xet") is not None
    xet_disabled = bool(getattr(constants, "HF_HUB_DISABLE_XET", False))
    xet_ready = xet_installed and not xet_disabled
    high_perf = bool(getattr(constants, "HF_XET_HIGH_PERFORMANCE", False))
    print(
        f"[hf] huggingface_hub {huggingface_hub.__version__}; "
        f"transfer={'hf_xet' if xet_ready else 'http'}; "
        f"HF_XET_HIGH_PERFORMANCE={'1' if high_perf else '0'}"
    )
    if xet_disabled:
        print("[hf] HF_HUB_DISABLE_XET=1, so Xet downloads are disabled.")
    elif not xet_installed:
        print('[hf] install "huggingface_hub[hf_xet]" for Xet-backed downloads.')
    elif not high_perf:
        print("[hf] set HF_XET_HIGH_PERFORMANCE=1 for high-throughput Xet mode.")


def main():
    parser = argparse.ArgumentParser(
        description="Download yuey.cpp GGUF models from Hugging Face."
    )
    parser.add_argument(
        "--encoding", default="q4_k_m",
        choices=[value.lower() for value in PUBLISHED_ENCODINGS],
        help="generation-model encoding (currently q4_k_m)",
    )
    parser.add_argument(
        "--profile", default="full", choices=list(PROFILES),
        help="core=generation; transcribe=core+SheetSage2; full=all Yuey features",
    )
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--out", default="models", help="output directory")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the resolved repo/file plan without downloading",
    )
    args = parser.parse_args()
    plan = build_download_plan(args.namespace, args.encoding, args.profile)

    if args.dry_run:
        for repo, files in plan:
            for filename in files:
                print(
                    f"[plan] https://huggingface.co/{repo}/resolve/main/{filename} "
                    f"-> {os.path.join(args.out, filename)}"
                )
        return

    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import get_token
    except ImportError:
        sys.exit('missing dependency: python -m pip install -U "huggingface_hub"')

    os.makedirs(args.out, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or get_token()
    print_transfer_info()
    for repo, files in plan:
        print(f"[download] {repo}")
        for filename in files:
            print(f"           {filename}")
        snapshot_download(
            repo_id=repo,
            allow_patterns=files,
            local_dir=args.out,
            max_workers=min(8, len(files)),
            token=token,
        )
    print(f"[done] Yuey {args.profile} ({args.encoding.upper()}) -> {args.out}/")
    print("run: yue2-server --models-dir models --encoding " + args.encoding.upper())


if __name__ == "__main__":
    main()

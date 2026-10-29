#!/usr/bin/env python3
"""Prepare immutable audio probes and score fixed-token reconstructions.

Dependencies: numpy, scipy, soundfile. This is a feasibility audit, not a trainer.
Sidecars are data; captions are never evaluated as commands or instructions.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, stft

# Re-exported for research scripts; production coordinators import it directly.
from training_provenance import digest, digests, native_runtime_identity


def load_audio(path, rate=24000):
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    if not np.isfinite(audio).all():
        raise ValueError(f"nonfinite audio: {path}")
    if sr != rate:
        divisor = math.gcd(sr, rate)
        audio = resample_poly(audio, rate // divisor, sr // divisor, axis=0)
    return audio


def correlation(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    if np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def features(audio, sr=24000):
    mono = audio.mean(axis=1)
    freq, _, spectrum = stft(mono, fs=sr, nperseg=2048, noverlap=1808,
                             boundary="zeros", padded=True)
    magnitude = np.abs(spectrum)
    power = magnitude ** 2
    valid = (freq >= 55) & (freq <= 5000)
    pitch = np.rint(69 + 12 * np.log2(freq[valid] / 440)).astype(int) % 12
    chroma = np.zeros((12, power.shape[1]))
    np.add.at(chroma, pitch, power[valid])
    chroma /= np.linalg.norm(chroma, axis=0, keepdims=True).clip(1e-10)
    onset = np.maximum(np.diff(np.log1p(1000 * magnitude), axis=1), 0).mean(axis=0)
    return magnitude, chroma, onset


def compare(source, render):
    a, b = load_audio(source), load_audio(render)
    # No lag search or dynamic alignment: metrics must not hide timing drift.
    duration_a, duration_b = len(a) / 24000, len(b) / 24000
    length = min(len(a), len(b))
    if length < 2048:
        raise ValueError("at least 2048 common samples are required")
    a, b = a[:length], b[:length]
    ma, ca, oa = features(a)
    mb, cb, ob = features(b)
    def level(x):
        return float(20 * np.log10(np.sqrt(np.mean(x.astype(float)**2)) + 1e-12))
    def width(x):
        if x.shape[1] != 2:
            return None
        mid, side = (x[:, 0] + x[:, 1]) / 2, (x[:, 0] - x[:, 1]) / 2
        return float(np.sqrt(np.mean(side**2)) / max(np.sqrt(np.mean(mid**2)), 1e-10))
    return {"source_seconds": duration_a, "render_seconds": duration_b,
            "compared_seconds": length / 24000,
            "chroma_correlation": correlation(ca, cb), "onset_correlation": correlation(oa, ob),
            "log_spectral_l1": float(np.mean(np.abs(np.log(ma + 1e-5) - np.log(mb + 1e-5)))),
            "source_rms_db": level(a), "render_rms_db": level(b),
            "source_side_mid_ratio": width(a), "render_side_mid_ratio": width(b),
            "render_peak": float(np.abs(b).max()),
            "render_samples_ge_1_fraction": float(np.mean(np.abs(b) >= 1))}


def token_stats(path):
    if path.stat().st_size % 4:
        raise ValueError("semantic stream has an incomplete int32")
    codes = np.fromfile(path, dtype="<i4")
    if not len(codes) or np.any(codes < 0) or np.any(codes >= 32768):
        raise ValueError("invalid semantic stream")
    _, counts = np.unique(codes, return_counts=True)
    p = counts / len(codes)
    return {"frames": len(codes), "seconds": len(codes) / 25,
            "distinct_codes": len(counts), "entropy_bits": float(-np.sum(p*np.log2(p))),
            "adjacent_repeat_fraction": float(np.mean(codes[1:] == codes[:-1])) if len(codes)>1 else 0.0}


def prepare(args):
    if args.seconds <= 0 or args.offset < 0 or args.count < 1:
        raise ValueError("seconds/count must be positive and offset nonnegative")
    if args.out.exists():
        raise ValueError("output exists; use a new directory to preserve probe provenance")
    inventory, probes = [], []
    for dataset in args.dataset:
        label, root = dataset.split("=", 1)
        if not label or Path(label).name != label or label in (".", ".."):
            raise ValueError("dataset label must be one path component")
        tracks = []
        for path in sorted(Path(root).resolve().rglob("*.wav")):
            info = sf.info(path)
            fingerprint = digest(path)
            # Split by content identity, before taking excerpts. Duplicates retain
            # their split even when copied/renamed into another dataset.
            split = "validation" if int(fingerprint[:8], 16) % 5 == 0 else "train"
            tracks.append({"dataset": label, "source": str(path), "sha256": fingerprint,
                           "sample_rate": info.samplerate, "channels": info.channels,
                           "seconds": info.duration, "split": split})
        if not tracks:
            raise ValueError(f"no WAV files in {root}")
        inventory.extend(tracks)
        # Deterministic spread across the sorted catalog, avoiding just the opening files.
        eligible = [t for t in tracks if t["seconds"] >= args.offset + args.seconds]
        if not eligible:
            raise ValueError(f"no tracks long enough in {root}")
        indices = np.linspace(0, len(eligible)-1, min(args.count, len(eligible)), dtype=int)
        for index in indices:
            track = eligible[index]
            path = Path(track["source"])
            sr = track["sample_rate"]
            with sf.SoundFile(path) as stream:
                stream.seek(round(args.offset * sr))
                audio = stream.read(round(args.seconds * sr), dtype="float32", always_2d=True)
            if not np.isfinite(audio).all():
                raise ValueError(f"nonfinite audio in {path}")
            directory = args.out / label / track["sha256"][:12]
            directory.mkdir(parents=True)
            sf.write(directory / "source.wav", audio, sr, subtype="FLOAT")
            caption = path.with_suffix(".txt")
            record = dict(track, start_seconds=args.offset, excerpt_seconds=len(audio)/sr,
                          probe=str(directory.resolve()),
                          excerpt_sha256=digest(directory / "source.wav"),
                          caption=caption.read_text(encoding="utf-8-sig").strip() if caption.exists() else "")
            (directory / "provenance.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            probes.append(record)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    (args.out / "probes.json").write_text(json.dumps(probes, indent=2), encoding="utf-8")
    print(json.dumps({"tracks": len(inventory), "hours": sum(t["seconds"] for t in inventory)/3600,
                      "probes": probes}, indent=2))


def run(args):
    probes = json.loads((args.root / "probes.json").read_text(encoding="utf-8"))
    files = {"model": args.model, "vae": args.vae, "tokenizer": args.tokenizer,
             "semantic_model": args.semantic_model, "nar_lora": args.nar_lora,
             "tokenize_executable": args.bin / "yue2-semantic-tokenize.exe" if os.name == 'nt'
                 else args.bin / "yue2-semantic-tokenize",
             "render_executable": args.bin / "yue2-render-tokens.exe" if os.name == 'nt'
                 else args.bin / "yue2-render-tokens"}
    provenance = {k: {"path": str(p.resolve()), "sha256": digest(p)} for k, p in files.items()}
    provenance.update(device=args.device, seed=args.seed, ode_steps=args.steps,
                      guidance="unguided native midpoint", version=1)
    # Never reuse tokens/renders under changed weights or options.
    run_path = args.root / "run.json"
    if run_path.exists() and json.loads(run_path.read_text()) != provenance:
        raise ValueError("run provenance changed; prepare a new audit directory")
    run_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    def execute(command, output, log):
        if output.exists():
            return
        start = time.monotonic()
        with log.open("w", encoding="utf-8") as stream:
            subprocess.run([str(x) for x in command], stdout=stream, stderr=subprocess.STDOUT, check=True)
        print(f"{output}: {time.monotonic()-start:.1f}s", flush=True)
    results = []
    for probe in probes:
        directory = Path(probe["probe"])
        if "excerpt_sha256" in probe and digest(directory / "source.wav") != probe["excerpt_sha256"]:
            raise ValueError(f"probe audio changed: {directory}")
        codes = directory / "tokens.i32"
        execute([files["tokenize_executable"], "--model", args.semantic_model,
                 "--audio", directory / "source.wav", "--out", codes, "--device", args.device],
                codes, directory / "tokenize.log")
        for variant in ("stock", "v9"):
            output = directory / f"{variant}.wav"
            command = [files["render_executable"], "--model", args.model, "--vae", args.vae,
                       "--tokenizer", args.tokenizer, "--tokens", codes, "--out", output,
                       "--prompt", probe["caption"], "--seed", args.seed,
                       "--ode-steps", args.steps, "--device", args.device]
            if variant == "v9":
                command += ["--lora", args.nar_lora]
            execute(command, output, directory / f"{variant}.log")
            metrics = compare(directory / "source.wav", output)
            metrics.update(dataset=probe["dataset"], source=probe["source"], variant=variant,
                           tokens=token_stats(codes))
            (directory / f"{variant}.metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            results.append(metrics)
            print(json.dumps(metrics), flush=True)
    (args.root / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


def vae_control(args):
    import torch
    from encode_yue2_vae_targets import Encoder
    torch.set_num_threads(args.threads)
    renderer = args.bin / ("yue2-render-tokens.exe" if os.name == "nt" else "yue2-render-tokens")
    manifest = {"vae_sha256": digest(args.vae), "renderer_sha256": digest(renderer),
                "encoder_sha256": digest(Path(__file__).with_name("encode_yue2_vae_targets.py")),
                "posterior": "mean", "device": args.device, "threads": args.threads}
    encoder = Encoder(args.vae)
    results = []
    for probe in json.loads((args.root / "probes.json").read_text(encoding="utf-8")):
        directory = Path(probe["probe"])
        target = directory / "vae-control"
        target.mkdir(exist_ok=True)
        record = dict(manifest, source_sha256=digest(directory / "source.wav"))
        provenance = target / "provenance.json"
        if provenance.exists() and json.loads(provenance.read_text()) != record:
            raise ValueError(f"VAE control provenance changed: {target}")
        provenance.write_text(json.dumps(record, indent=2), encoding="utf-8")
        means, render = target / "means.f32", target / "render.wav"
        if not means.exists():
            encoder.encode(load_audio(directory / "source.wav", 48000)).tofile(means)
        if not render.exists():
            with (target / "decode.log").open("w", encoding="utf-8") as log:
                subprocess.run([str(renderer), "--vae", str(args.vae), "--latents-in", str(means),
                                "--out", str(render), "--device", args.device],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        result = compare(directory / "source.wav", render)
        result.update(dataset=probe["dataset"], source=probe["source"], variant="vae-only")
        (target / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        results.append(result)
        print(json.dumps(result), flush=True)
    (args.root / "vae-controls.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


def context_control(args):
    """Compare crop-first tokenization with full-track context/normalization."""
    tokenizer = args.bin / ("yue2-semantic-tokenize.exe" if os.name == "nt" else "yue2-semantic-tokenize")
    renderer = args.bin / ("yue2-render-tokens.exe" if os.name == "nt" else "yue2-render-tokens")
    manifest = {k: digest(getattr(args, k)) for k in ("model", "vae", "tokenizer", "semantic_model", "nar_lora")}
    manifest.update(device=args.device, seed=args.seed, steps=args.steps,
                    tokenizer_exe=digest(tokenizer), renderer_exe=digest(renderer))
    results = []
    for probe in json.loads((args.root / "probes.json").read_text(encoding="utf-8")):
        source = Path(probe["source"])
        if digest(source) != probe["sha256"]:
            raise ValueError(f"source track changed: {source}")
        directory = Path(probe["probe"])
        target = directory / "full-context"
        target.mkdir(exist_ok=True)
        record = dict(manifest, source_sha256=probe["sha256"],
                      start_seconds=probe["start_seconds"], seconds=probe["excerpt_seconds"])
        provenance = target / "provenance.json"
        if provenance.exists() and json.loads(provenance.read_text()) != record:
            raise ValueError(f"context-control provenance changed: {target}")
        provenance.write_text(json.dumps(record, indent=2), encoding="utf-8")
        full, codes, render = target / "full.i32", target / "crop.i32", target / "v9.wav"
        if not full.exists():
            with (target / "tokenize.log").open("w", encoding="utf-8") as log:
                subprocess.run([str(tokenizer), "--model", str(args.semantic_model), "--audio", str(source),
                                "--out", str(full), "--device", args.device],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        tokens = np.fromfile(full, dtype="<i4")
        start, length = round(probe["start_seconds"]*25), round(probe["excerpt_seconds"]*25)
        if len(tokens) < start+length:
            raise ValueError("full-track tokenizer returned too few frames")
        tokens[start:start+length].tofile(codes)
        if not render.exists():
            with (target / "render.log").open("w", encoding="utf-8") as log:
                subprocess.run([str(renderer), "--model", str(args.model), "--vae", str(args.vae),
                                "--tokenizer", str(args.tokenizer), "--tokens", str(codes), "--out", str(render),
                                "--prompt", probe["caption"], "--seed", str(args.seed), "--ode-steps", str(args.steps),
                                "--lora", str(args.nar_lora), "--device", args.device],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        original = np.fromfile(directory / "tokens.i32", dtype="<i4")
        if len(original) != length:
            raise ValueError("crop-first token stream has a different length")
        result = compare(directory / "source.wav", render)
        result.update(dataset=probe["dataset"], source=probe["source"], variant="v9-full-context",
                      token_agreement_with_crop=float(np.mean(original==tokens[start:start+length])))
        (target / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        results.append(result)
        print(json.dumps(result), flush=True)
    (args.root / "context-controls.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--dataset", action="append", required=True, help="label=directory")
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--seconds", type=float, default=12)
    prep.add_argument("--offset", type=float, default=20)
    prep.add_argument("--count", type=int, default=2, help="probes per dataset")
    score = commands.add_parser("compare")
    score.add_argument("--source", type=Path, required=True)
    score.add_argument("--render", type=Path, required=True)
    score.add_argument("--tokens", type=Path)
    score.add_argument("--out", type=Path, required=True)
    experiment = commands.add_parser("run")
    experiment.add_argument("--root", type=Path, required=True)
    experiment.add_argument("--bin", type=Path, required=True)
    for name in ("model", "vae", "tokenizer", "semantic-model", "nar-lora"):
        experiment.add_argument("--" + name, type=Path, required=True)
    experiment.add_argument("--device", default="cuda")
    experiment.add_argument("--seed", type=int, default=831001)
    experiment.add_argument("--steps", type=int, default=32)
    control = commands.add_parser("vae-control")
    control.add_argument("--root", type=Path, required=True)
    control.add_argument("--vae", type=Path, required=True)
    control.add_argument("--bin", type=Path, required=True)
    control.add_argument("--device", default="cuda")
    control.add_argument("--threads", type=int, default=4)
    context = commands.add_parser("context-control")
    context.add_argument("--root", type=Path, required=True)
    context.add_argument("--bin", type=Path, required=True)
    for name in ("model", "vae", "tokenizer", "semantic-model", "nar-lora"):
        context.add_argument("--" + name, type=Path, required=True)
    context.add_argument("--device", default="cuda")
    context.add_argument("--seed", type=int, default=831001)
    context.add_argument("--steps", type=int, default=32)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "run":
        run(args)
    elif args.command == "vae-control":
        vae_control(args)
    elif args.command == "context-control":
        context_control(args)
    else:
        result = compare(args.source, args.render)
        if args.tokens:
            result["tokens"] = token_stats(args.tokens)
        args.out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

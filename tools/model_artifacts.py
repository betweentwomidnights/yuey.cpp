"""Pure naming and download-manifest helpers for published yuey.cpp artifacts."""

VERSION = "v1.0"
REPO = "YuE2-3B-GGUF"
PUBLISHED_ENCODINGS = ("Q4_K_M",)
PROFILES = ("core", "transcribe", "full")


def generation_filename(encoding):
    enc = encoding.upper()
    if enc not in PUBLISHED_ENCODINGS:
        raise ValueError(
            f"unpublished encoding: {encoding} "
            f"(currently available: {', '.join(e.lower() for e in PUBLISHED_ENCODINGS)})"
        )
    return f"yue2-3.6B-{VERSION}-{enc}.gguf"


def profile_files(encoding="Q4_K_M", profile="full"):
    """Return the complete repo-relative file set for one install profile."""
    if profile not in PROFILES:
        raise ValueError(
            f"unknown profile: {profile} (expected one of {', '.join(PROFILES)})"
        )
    files = [
        generation_filename(encoding),
        f"yue2-vae-{VERSION}-F16.gguf",
        "yue2-qwen.tiktoken",
    ]
    if profile in ("transcribe", "full"):
        files.append(f"sheetsage2-mert2-0.7B-{VERSION}-F16.gguf")
    if profile == "full":
        files.extend([
            f"yue2-instrumental-cot-full-{VERSION}-F16-LoRA.gguf",
            f"yue2-realaudio-nar-v9-{VERSION}-F16-LoRA.gguf",
            f"yue2-semantic-tokenizer-0.7B-{VERSION}-F16.gguf",
        ])
    return files


def build_download_plan(namespace, encoding="Q4_K_M", profile="full"):
    """Return ``[(repo_id, [filenames...])]`` for the downloader."""
    return [(f"{namespace}/{REPO}", profile_files(encoding, profile))]

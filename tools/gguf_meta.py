#!/usr/bin/env python3
"""Shared GGUF `general.*` metadata and file naming for the yue2.cpp converters.

Follows the convention sa3.cpp and audiocraft.cpp take from the GGUF
specification (ggml/docs/gguf.md):

    <BaseName>-<SizeLabel>-<Version>-<Encoding>[-<Type>].gguf

SizeLabel appears on model-like components only; the VAE is exempt, as the
autoencoders are in the sibling repositories. The `general.architecture` value
is still set by each converter's GGUFWriter and is what the loaders key off;
this only adds the descriptive/catalog metadata on top.
"""

VERSION = "v1.0"
# The released YuE2, SheetSage2, and MERT2 weights are CC BY-NC 4.0.
LICENSE = "cc-by-nc-4.0"

ENCODINGS = {"f32": "F32", "f16": "F16", "bf16": "BF16"}
# general.file_type values from the GGUF specification (llama.cpp's LLAMA_FTYPE).
FILE_TYPES = {"f32": 0, "f16": 1, "bf16": 32}


def size_label(n_params):
    """Param-count class per the gguf convention. Sub-billion models render as '0.xB'
    down to 100M; smaller use M/K. e.g. 3.63e9 -> '3.6B', 6.8e8 -> '0.7B'."""
    if n_params >= 1e8:
        s = f"{n_params / 1e9:.1f}B"
    elif n_params >= 1e5:
        s = f"{n_params / 1e6:.0f}M"
    else:
        s = f"{n_params / 1e3:.0f}K"
    return s.replace(".0B", "B")


def encoding(storage_type):
    """The Encoding field for a converter storage type such as 'bf16'."""
    try:
        return ENCODINGS[storage_type]
    except KeyError as exc:
        raise ValueError(f"unsupported storage type: {storage_type}") from exc


def gguf_filename(basename, storage_type, n_params=None, kind=None):
    """e.g. gguf_filename('yue2', 'bf16', 3.63e9) -> 'yue2-3.6B-v1.0-BF16.gguf'."""
    fields = [basename]
    if n_params is not None:
        fields.append(size_label(n_params))
    fields += [VERSION, encoding(storage_type)]
    if kind:
        fields.append(kind)
    return "-".join(fields) + ".gguf"


def add_general(w, basename, name, n_params=None, license_id=LICENSE):
    """Stamp the convention's catalog metadata. Pass n_params on model-like
    components to emit size_label; omit it for the VAE."""
    w.add_name(name)
    w.add_string("general.basename", basename)
    w.add_string("general.version", VERSION)
    w.add_string("general.license", license_id)
    if n_params is not None:
        w.add_string("general.size_label", size_label(n_params))


def add_file_type(w, storage_type):
    """Record the majority tensor storage as general.file_type."""
    encoding(storage_type)
    w.add_file_type(FILE_TYPES[storage_type])


def add_sources(w, sources):
    """Record upstream checkpoint identities with the standard base-model fields.
    `sources` holds (name, organization, repo_url, revision-or-None) tuples."""
    w.add_base_model_count(len(sources))
    for index, (name, organization, repo_url, revision) in enumerate(sources):
        w.add_base_model_name(index, name)
        w.add_base_model_organization(index, organization)
        w.add_base_model_repo_url(index, repo_url)
        if revision:
            w.add_base_model_version(index, revision)

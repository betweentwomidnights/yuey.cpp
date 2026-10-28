"""SHA-256 provenance shared by the training coordinators.

Standard library only: the production training add-on imports this module,
while the research audit's numpy/scipy/soundfile metrics stay optional.
"""
import hashlib
import os
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def native_runtime_identity(executable):
    """Include shared backends: an unchanged executable can load a new GGML."""
    executable = Path(executable)
    libraries = sorted(p for p in executable.parent.iterdir() if p.is_file() and
                       (p.suffix.lower() in ('.dll', '.dylib') or '.so' in p.name))
    return {
        'executable': {executable.name: digest(executable)},
        'shared_libraries': {p.name: digest(p) for p in libraries},
        'precision_environment': {key: os.environ.get(key) for key in
                                  ('GGML_CUDA_TF32', 'NVIDIA_TF32_OVERRIDE',
                                   'CUBLAS_WORKSPACE_CONFIG')},
    }

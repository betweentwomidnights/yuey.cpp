"""SHA-256 provenance shared by the training coordinators.

Standard library only: the production training add-on imports this module,
while the research audit's numpy/scipy/soundfile metrics stay optional.
"""
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digests(paths, workers=None):
    """SHA-256 of several files concurrently, in input order.

    File reads and hashlib updates of large blocks release the GIL, so threads
    overlap I/O and hashing. Every byte of every file is still hashed.
    """
    paths = list(paths)
    workers = workers or min(8, os.cpu_count() or 1, max(len(paths), 1))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(digest, paths))


def native_runtime_identity(executable):
    """Include shared backends: an unchanged executable can load a new GGML."""
    executable = Path(executable)
    libraries = sorted(p for p in executable.parent.iterdir() if p.is_file() and
                       (p.suffix.lower() in ('.dll', '.dylib') or '.so' in p.name))
    values = digests([executable, *libraries])
    return {
        'executable': {executable.name: values[0]},
        'shared_libraries': dict(zip((p.name for p in libraries), values[1:])),
        'precision_environment': {key: os.environ.get(key) for key in
                                  ('GGML_CUDA_TF32', 'NVIDIA_TF32_OVERRIDE',
                                   'CUBLAS_WORKSPACE_CONFIG')},
    }

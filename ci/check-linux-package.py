"""Verify extracted Linux native packages without models or desktop dependencies."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

SERVICE = 'yuey'
SERVERS = [('yue2-server', 'yue2')]
HELP_TOOLS = []

def main():
    if not __debug__:
        raise RuntimeError("Package verification requires assertions; remove PYTHONOPTIMIZE")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('version')
    parser.add_argument('--expect-platform', required=True)
    parser.add_argument('--require-backend', choices=['cuda','vulkan'])
    args = parser.parse_args()
    directory = args.directory.resolve(strict=True)
    version = args.version.removeprefix('v')
    env = {k:v for k,v in os.environ.items()
           if not k.startswith(('YUE2_', 'AC_', 'ACE_', 'GGML_')) and k != 'LD_LIBRARY_PATH'}
    with tempfile.TemporaryDirectory(prefix=SERVICE+'-package-probe-') as scratch:
        for key in ('YUE2_MODELS_DIR', 'AC_MODELS_DIR', 'ACE_MODELS_DIR'):
            env[key] = str(Path(scratch)/'missing-models')
        def output(tool, *flags):
            result = subprocess.run([str(directory/tool), *flags], cwd=scratch, env=env,
                                    capture_output=True, text=True, timeout=60)
            assert result.returncode == 0, (tool, flags, result.stderr)
            return result.stdout.strip()
        for tool, service in SERVERS:
            assert output(tool, '--version') == version, tool
            props = json.loads(output(tool, '--props'))
            assert props['service'] == service and props['version'] == version, props
            assert any(d['backend'].upper().startswith('CPU') for d in props['devices']), props
            if args.require_backend:
                assert any(d['backend'].lower() == args.require_backend
                           and d['type'] in {'gpu','igpu','integrated_gpu'}
                           for d in props['devices']), props
        for tool in HELP_TOOLS:
            output(tool, '--help')
        if SERVICE == 'acestep':
            assert props['capabilities']['native_training_cli'] is True, props
            assert props['capabilities']['functional_adapters'] is True, props
            assert (directory/'genres_vocab.txt').is_file()
        if SERVICE == 'audiocraft':
            assert props['runtime'] == 'audiocraft', props
        assert not list(Path(scratch).iterdir()), 'Offline probes wrote runtime files'
    info = json.loads((directory/'BUILD-INFO.json').read_text())
    assert info['service'] == SERVICE and info['version'] == args.version, info
    assert info['platform'] == args.expect_platform and info['commit'] and info['ggml_commit'], info
    print('PASS extracted '+SERVICE+' versions/devices, metadata and native controls')

if __name__ == '__main__':
    main()

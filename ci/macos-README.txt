yuey for macOS
==============

Requires macOS 13.3 or later. Apple Silicon uses Metal or CPU; Intel uses CPU.
Keep the executables and all dylibs together in this folder.

Download the models from a Terminal in this folder:

    bash models.sh --encoding q4_k_m --profile full

Start the server:

    ./yue2-server --models-dir models

The default port is 8007. The server selects an available GPU automatically.
Use --device cpu to run on CPU, or --device MTL to require Apple Silicon Metal.
Run ./yue2-server --help for the available options.

The package is signed with Developer ID and notarized for release builds.
Bare executables in a zip cannot carry a stapled ticket; Gatekeeper retrieves
the notarization ticket online when checking a downloaded binary.

BUILD-INFO.json identifies the yuey and GGML commits and signing status.
LICENSE, LICENSE-ggml.txt and THIRD_PARTY_NOTICES.md cover the runtime code.
Model weights have their own license, described by the model repository.

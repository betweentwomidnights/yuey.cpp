yuey (yuey.cpp) - standalone Windows package
=============================================

Everything yuey needs in one folder: the server and tools, the CPU backend in
every instruction-set variant, both GPU backends (CUDA and Vulkan), and the
CUDA runtime. ggml loads whichever backend this machine can run: CUDA on an
NVIDIA GPU, Vulkan on AMD and Intel. BUILD-INFO.json names the exact commits
and toolchain this package was built from.

1. Download the models (about 6 GB at Q4_K_M, into .\models):

     models.cmd

   Other tiers: models.cmd --encoding q8_0   (or bf16). models.cmd --help
   lists the options.

2. Start the server:

     yue2-server.exe --models-dir models

   Then open http://127.0.0.1:8007/ for the built-in UI, or point a client
   such as gary4juce at localhost:8007.

3. To pick a GPU backend yourself: yue2-server.exe --device cuda (or vulkan).
   yue2-server.exe --props lists the devices this machine offers.

Server reference, request formats and CLI options:
https://github.com/betweentwomidnights/yuey.cpp/blob/main/docs/server.md

Licenses: LICENSE (yuey.cpp), LICENSE-ggml.txt, THIRD_PARTY_NOTICES.md, and
NVIDIA-CUDA-EULA.txt for the CUDA runtime DLLs. Model licenses stay with the
model repositories.

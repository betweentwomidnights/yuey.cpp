yuey (yuey.cpp), standalone windows package
============================================

everything yuey needs is in this folder: the server and tools, the CPU backend
in every instruction-set variant, both GPU backends (CUDA and Vulkan), and the
CUDA runtime. ggml loads whichever backend this machine can run: CUDA on an
NVIDIA GPU, Vulkan on AMD and Intel. BUILD-INFO.json names the exact commits
and toolchain this package was built from.

1. download the models, about 6 GB at Q4_K_M, into .\models:

     models.cmd

   for another tier, run models.cmd --encoding q8_0 (or bf16).
   models.cmd --help lists the rest of the options.

2. start the server:

     yue2-server.exe --models-dir models

   then open http://127.0.0.1:8007/ for the built-in UI, or point a client
   like gary4juce at localhost:8007.

3. to choose the GPU backend yourself, run yue2-server.exe --device cuda (or
   vulkan). yue2-server.exe --props lists the devices this machine has.

server reference, request formats and CLI options:
https://github.com/betweentwomidnights/yuey.cpp/blob/main/docs/server.md

licenses: LICENSE (yuey.cpp), LICENSE-ggml.txt, THIRD_PARTY_NOTICES.md, and
NVIDIA-CUDA-EULA.txt for the CUDA runtime DLLs. model licenses stay with the
model repositories.

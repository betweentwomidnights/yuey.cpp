cmake_minimum_required(VERSION 3.20)
project(yue2_cpp VERSION 0.2.0 LANGUAGES C CXX)

include(CTest)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

option(YUE2_CUDA "Build the GGML CUDA backend" OFF)
option(YUE2_VULKAN "Build the GGML Vulkan backend" OFF)
option(YUE2_METAL "Build the GGML Metal backend" OFF)
option(YUE2_HIP "Build the GGML HIP/ROCm backend" OFF)
option(YUE2_BUILD_TOOLS "Build command-line tools" ON)
option(YUE2_BUILD_TESTS "Build unit tests" ${BUILD_TESTING})
option(YUE2_BUILD_SHARED "Build the stable C ABI shared library" ON)

if(YUE2_BUILD_TOOLS OR YUE2_BUILD_TESTS)
    add_library(yue2-server-support STATIC
        tools/server/base64.cpp
        tools/server/http.cpp
        tools/server/json.cpp
        tools/server/multipart.cpp
    )
    target_include_directories(yue2-server-support PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/tools
    )
    if(WIN32)
        target_link_libraries(yue2-server-support PUBLIC ws2_32)
    endif()
endif()

if(YUE2_CUDA)
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
endif()
if(YUE2_VULKAN)
    set(GGML_VULKAN ON CACHE BOOL "" FORCE)
endif()
if(YUE2_METAL)
    set(GGML_METAL ON CACHE BOOL "" FORCE)
endif()
if(YUE2_HIP)
    set(GGML_HIP ON CACHE BOOL "" FORCE)
endif()

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ggml/CMakeLists.txt")
    find_package(Git QUIET)
    if(GIT_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
        message(STATUS "Initializing ggml submodule")
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" submodule update --init --recursive ggml
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
    endif()
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ggml/CMakeLists.txt")
        message(FATAL_ERROR "ggml is missing; run git submodule update --init --recursive")
    endif()
endif()

add_subdirectory(ggml)

add_library(yue2 STATIC
    src/autoregressive.cpp
    src/audio.cpp
    src/generation.cpp
    src/generation_pipeline.cpp
    src/lora.cpp
    src/mert2_encoder.cpp
    src/mert2_frontend.cpp
    src/quantize.cpp
    src/runtime_info.cpp
    src/sheetsage2_tokens.cpp
    src/tokenizer.cpp
    src/transcription.cpp
    src/vae.cpp
)
set_target_properties(yue2 PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(yue2 PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(yue2 PUBLIC ggml)

if(YUE2_BUILD_SHARED)
    add_library(yue2_shared SHARED src/c_api.cpp)
    target_compile_definitions(yue2_shared PRIVATE YUE2_BUILD_DLL)
    target_include_directories(yue2_shared PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    )
    target_link_libraries(yue2_shared PRIVATE yue2)
    set_target_properties(yue2_shared PROPERTIES OUTPUT_NAME yue2)
    if(WIN32)
        # Keep the import library distinct from the static yue2.lib while the
        # runtime artifact remains yue2.dll.
        set_target_properties(yue2_shared PROPERTIES ARCHIVE_OUTPUT_NAME yue2-dll)
    endif()
endif()

if(YUE2_BUILD_TOOLS)
    add_executable(yue2-transcribe tools/yue2-transcribe.cpp)
    target_link_libraries(yue2-transcribe PRIVATE yue2)

    add_executable(yue2-inspect-generation tools/yue2-inspect-generation.cpp)
    target_link_libraries(yue2-inspect-generation PRIVATE yue2)

    # yue2-quantize: K-quant/Q8_0 generation GGUFs; yue2-quant-check: per-tensor cosine.
    add_executable(yue2-quantize tools/yue2-quantize.cpp)
    target_link_libraries(yue2-quantize PRIVATE yue2)

    add_executable(yue2-quant-check tools/yue2-quant-check.cpp)
    target_link_libraries(yue2-quant-check PRIVATE yue2)

    # yue2-quant-eval: teacher-forced AR, fixed-noise flow and rendered comparisons.
    add_executable(yue2-quant-eval tools/yue2-quant-eval.cpp)
    target_link_libraries(yue2-quant-eval PRIVATE yue2)

    add_executable(yue2-generate tools/yue2-generate.cpp)
    target_link_libraries(yue2-generate PRIVATE yue2)

    set(YUE2_UI_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/yue2_ui_html.h")
    add_custom_command(
        OUTPUT "${YUE2_UI_HEADER}"
        COMMAND "${CMAKE_COMMAND}"
            -DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/tools/ui/index.html
            -DOUTPUT=${YUE2_UI_HEADER}
            -DSYMBOL=index_html
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_resource.cmake
        DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/ui/index.html
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_resource.cmake
        VERBATIM
    )
    add_executable(yue2-server tools/yue2-server.cpp "${YUE2_UI_HEADER}")
    target_include_directories(yue2-server PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
    target_link_libraries(yue2-server PRIVATE yue2 yue2-server-support)
endif()

if(YUE2_BUILD_TESTS)
    function(yue2_enable_test_assertions target)
        if(MSVC)
            target_compile_options(${target} PRIVATE /UNDEBUG)
        else()
            target_compile_options(${target} PRIVATE -UNDEBUG)
        endif()
    endfunction()

    add_executable(yue2-audio-test tests/audio_test.cpp)
    target_link_libraries(yue2-audio-test PRIVATE yue2)
    yue2_enable_test_assertions(yue2-audio-test)
    add_test(NAME yue2-audio-test COMMAND yue2-audio-test)

    add_executable(yue2-mert2-frontend-test tests/mert2_frontend_test.cpp)
    target_link_libraries(yue2-mert2-frontend-test PRIVATE yue2)
    yue2_enable_test_assertions(yue2-mert2-frontend-test)
    add_test(NAME yue2-mert2-frontend-test COMMAND yue2-mert2-frontend-test)

    # This is a deliberately manual integration test because it loads the
    # 1.35 GB converted model and an official PyTorch fixture:
    # yue2-mert2-subsampler-test MODEL.gguf EXPECTED.f32
    add_executable(yue2-mert2-subsampler-test tests/mert2_subsampler_test.cpp)
    target_link_libraries(yue2-mert2-subsampler-test PRIVATE yue2)
    yue2_enable_test_assertions(yue2-mert2-subsampler-test)

    add_executable(yue2-sheetsage2-decoder-test tests/sheetsage2_decoder_test.cpp)
    target_link_libraries(yue2-sheetsage2-decoder-test PRIVATE yue2)
    yue2_enable_test_assertions(yue2-sheetsage2-decoder-test)

    # Manual real-weight smoke test of WAV -> MERT2 -> SheetSage2 -> ABC/MIDI:
    # yue2-transcription-integration-test MODEL.gguf
    add_executable(yue2-transcription-integration-test tests/transcription_integration_test.cpp)
    target_link_libraries(yue2-transcription-integration-test PRIVATE yue2)
    yue2_enable_test_assertions(yue2-transcription-integration-test)

    # Manual real-weight parity test against an official FP32 VAE fixture:
    # yue2-vae-decoder-test VAE.gguf LATENTS.f32 EXPECTED.f32 FRAMES [DEVICE]
    add_executable(yue2-vae-decoder-test tests/vae_decoder_test.cpp)
    target_link_libraries(yue2-vae-decoder-test PRIVATE yue2)
    yue2_enable_test_assertions(yue2-vae-decoder-test)

    add_executable(yue2-tokenizer-test tests/tokenizer_test.cpp)
    target_link_libraries(yue2-tokenizer-test PRIVATE yue2)
    yue2_enable_test_assertions(yue2-tokenizer-test)
    add_test(NAME yue2-tokenizer-test COMMAND yue2-tokenizer-test)

    add_executable(yue2-transcription-json-test tests/transcription_json_test.cpp)
    target_link_libraries(yue2-transcription-json-test PRIVATE yue2)
    yue2_enable_test_assertions(yue2-transcription-json-test)
    add_test(NAME yue2-transcription-json-test COMMAND yue2-transcription-json-test)

    add_executable(yue2-midi-export-test tests/midi_export_test.cpp)
    target_link_libraries(yue2-midi-export-test PRIVATE yue2)
    yue2_enable_test_assertions(yue2-midi-export-test)
    add_test(NAME yue2-midi-export-test COMMAND yue2-midi-export-test)

    add_executable(yue2-lora-test tests/lora_test.cpp)
    target_include_directories(yue2-lora-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(yue2-lora-test PRIVATE yue2)
    yue2_enable_test_assertions(yue2-lora-test)
    add_test(NAME yue2-lora-test COMMAND yue2-lora-test)

    # Synthetic quantizer plan, layout, and quantized kernel test:
    # yue2-quantize-test [DEVICE]
    add_executable(yue2-quantize-test tests/quantize_test.cpp)
    target_include_directories(yue2-quantize-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(yue2-quantize-test PRIVATE yue2)
    yue2_enable_test_assertions(yue2-quantize-test)
    add_test(NAME yue2-quantize-test COMMAND yue2-quantize-test)

    add_executable(yue2-server-support-test tests/server_support_test.cpp)
    target_link_libraries(yue2-server-support-test PRIVATE yue2-server-support)
    yue2_enable_test_assertions(yue2-server-support-test)
    add_test(NAME yue2-server-support-test COMMAND yue2-server-support-test)

    add_executable(yue2-runtime-info-test tests/runtime_info_test.cpp)
    target_link_libraries(yue2-runtime-info-test PRIVATE yue2)
    yue2_enable_test_assertions(yue2-runtime-info-test)
    add_test(NAME yue2-runtime-info-test COMMAND yue2-runtime-info-test)

    add_executable(yue2-planning-header-test tests/planning_header_test.cpp)
    target_link_libraries(yue2-planning-header-test PRIVATE yue2)
    yue2_enable_test_assertions(yue2-planning-header-test)
    add_test(NAME yue2-planning-header-test COMMAND yue2-planning-header-test)

    # Manual exact parity test against vectors emitted by the official Python
    # tokenizer: yue2-text-tokenizer-test QWEN.TIKTOKEN FIXTURE
    add_executable(yue2-text-tokenizer-test tests/text_tokenizer_test.cpp)
    target_link_libraries(yue2-text-tokenizer-test PRIVATE yue2)
    yue2_enable_test_assertions(yue2-text-tokenizer-test)

    # Manual upstream protocol parity test:
    # yue2-generation-protocol-test QWEN.TIKTOKEN FIXTURE
    add_executable(yue2-generation-protocol-test tests/generation_protocol_test.cpp)
    target_link_libraries(yue2-generation-protocol-test PRIVATE yue2)
    yue2_enable_test_assertions(yue2-generation-protocol-test)

    # Manual real-weight parity test:
    # yue2-autoregressive-test MODEL.gguf OFFICIAL_LOGITS.bin [DEVICE]
    add_executable(yue2-autoregressive-test tests/autoregressive_test.cpp)
    target_link_libraries(yue2-autoregressive-test PRIVATE yue2)
    yue2_enable_test_assertions(yue2-autoregressive-test)

    # Manual real-weight NAR/flow parity test:
    # yue2-flow-test MODEL.gguf VAE.gguf OFFICIAL_FLOW.bin [DEVICE]
    add_executable(yue2-flow-test tests/flow_test.cpp)
    target_link_libraries(yue2-flow-test PRIVATE yue2)
    yue2_enable_test_assertions(yue2-flow-test)

    # Manual real-weight text/score -> semantic -> flow -> VAE smoke test:
    # yue2-generation-integration-test MODEL.gguf VAE.gguf QWEN.TIKTOKEN [DEVICE] [LORA.gguf]
    add_executable(yue2-generation-integration-test tests/generation_integration_test.cpp)
    target_link_libraries(yue2-generation-integration-test PRIVATE yue2)
    yue2_enable_test_assertions(yue2-generation-integration-test)

    if(YUE2_BUILD_SHARED)
        add_executable(yue2-c-api-test tests/c_api_test.c)
        target_link_libraries(yue2-c-api-test PRIVATE yue2_shared)
        add_test(NAME yue2-c-api-test COMMAND yue2-c-api-test)

        # Manual real-weight pure-C generation smoke test:
        # yue2-c-generation-test MODEL.gguf VAE.gguf QWEN.TIKTOKEN [DEVICE] [LORA.gguf]
        add_executable(yue2-c-generation-test tests/c_generation_test.c)
        target_link_libraries(yue2-c-generation-test PRIVATE yue2_shared)

        # Manual real-weight pure-C audio-to-score smoke test:
        # yue2-c-transcription-test TRANSCRIPTION.gguf [DEVICE]
        add_executable(yue2-c-transcription-test tests/c_transcription_test.c)
        target_link_libraries(yue2-c-transcription-test PRIVATE yue2_shared)
        if(NOT WIN32)
            target_link_libraries(yue2-c-transcription-test PRIVATE m)
        endif()
    endif()
endif()

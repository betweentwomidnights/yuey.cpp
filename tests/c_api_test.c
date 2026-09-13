#include "yue2/c_api.h"

#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    char error[128] = {0};
    yue2_transcriber_config transcriber;
    memset(&transcriber, 0, sizeof transcriber);
    transcriber.size = (uint32_t)sizeof transcriber;
    assert(yue2_transcriber_create(&transcriber, error, (int32_t)sizeof error) == NULL);
    assert(error[0] != '\0');

    memset(error, 0, sizeof error);
    yue2_generator_config generator;
    memset(&generator, 0, sizeof generator);
    generator.size = (uint32_t)sizeof generator;
    assert(yue2_generator_create(&generator, error, (int32_t)sizeof error) == NULL);
    assert(error[0] != '\0');

    /* Tail extension of generator config remains compatible with the initial
     * ABI: an older caller's size reaches model loading rather than being
     * rejected as a too-small configuration. */
    memset(error, 0, sizeof error);
    generator.size = (uint32_t)offsetof(yue2_generator_config, lora_adapters);
    generator.model_path = "missing-model.gguf";
    generator.vae_path = "missing-vae.gguf";
    generator.tokenizer_path = "missing-tokenizer";
    assert(yue2_generator_create(&generator, error, (int32_t)sizeof error) == NULL);
    assert(strstr(error, "invalid generator configuration") == NULL);

    yue2_generation_result generation;
    memset(&generation, 0, sizeof generation);
    generation.size = (uint32_t)sizeof generation;
    yue2_free_generation_result(&generation);
    assert(generation.size == sizeof generation);

    yue2_transcription_result transcription;
    memset(&transcription, 0, sizeof transcription);
    transcription.size = (uint32_t)sizeof transcription;
    yue2_free_transcription_result(&transcription);
    assert(transcription.size == sizeof transcription);

    assert(yue2_c_version() != NULL && strlen(yue2_c_version()) != 0);
    printf("%s C ABI ownership and errors: ok\n", yue2_c_version());
    return 0;
}

# yue2.cpp C ABI V1

The stable embedding contract is declared in
[`include/yue2/c_api_v1.h`](../include/yue2/c_api_v1.h). A host dynamically
resolves one symbol:

```c
const yue2_api_v1 *api = yue2_get_api(YUE2_ABI_VERSION_1);
```

`NULL` means the requested ABI major is unsupported. The returned table is
static library memory and remains valid until the module is unloaded.

The older individual functions in `c_api.h` remain available for existing
callers, but new gary4local, gary4juce, and third-party integrations should use
the V1 table.

## Compatibility rules

- Every public data structure begins with `uint32_t size`.
- Set `size`, then call the matching initializer from the table. Initializers
  establish the native defaults and preserve any unknown caller tail.
- The `*_MIN_SIZE` macros freeze the V1 prefix. A later library may append
  fields but will continue accepting that prefix.
- Structures embedded by value, such as `yue2_audio_view_v1` inside a
  transcription request, are frozen for V1.
- Array entries use a caller-provided stride so entries may grow in a later ABI.
- Do not pass C++ standard-library objects across this boundary.
- No C++ exception crosses the ABI.

Windows functions use `__cdecl`. All library-owned arrays and strings must be
released through the matching result-free function. This keeps allocation and
deallocation in the same DLL/CRT. Result-free calls are idempotent and preserve
unknown caller tails.

## Contexts and residency

Transcription and generation deliberately use separate contexts. A host can
keep the small SheetSage2/MERT2 transcriber resident without loading YuE2's
generation model and VAE.

Create functions copy all configuration strings and adapter definitions. The
caller may release its configuration storage after creation. `*_unload()`
releases model and backend allocations while retaining the copied
configuration; the next operation reloads lazily. `*_destroy()` releases the
context itself.

A context is not reentrant. Serialize calls through one worker queue, or create
additional contexts when duplicate model residency is acceptable.

## Operations

### Transcription

`transcribe()` accepts planar or interleaved float PCM and returns:

- YuE2 ABC;
- combined format-1 MIDI;
- separate melody, vocal, instrumental, and chord MIDI files;
- the lossless typed events/windows JSON document; and
- event count and source duration.

Every MIDI pointer is an independent complete Standard MIDI File.

### Planning

`plan()` runs symbolic planning only. It returns ABC, raw ABC token IDs, seed,
bar count, musical duration, and truncation state. It skips semantic generation,
flow, and VAE decoding. The generator remains resident for a later render.

This is the native foundation for a plan/edit/render workflow. Supplying a
complete `abc` in a generation request bypasses planning; `abc_prefix` instead
locks a validated header or continuation prefix before the planner composes.

### Generation

`generate()` returns interleaved float PCM together with the ABC, ABC token IDs,
semantic codec IDs, flow latents, seed, score length, derived semantic budget,
and truncation flags.

`target_bars == 0` retains the complete plan. A nonzero target is enforced
during planning so YuE2 cannot end a newly planned score before the requested
bar count. `YUE2_ENDING_OUTRO_V1` fits a completed longer score to the target by
retaining its true ending.

`instrumental` is best effort: YuE2 may still synthesize voice-like material.
Lyrics are not transcribed from reference audio, so a host that wants to retain
specific sung words must provide them.

## Progress and cancellation

Callbacks run synchronously on the inference thread. Progress payloads are
borrowed for the duration of the callback and report typed stages plus a stable
human-readable stage name. Callback code should update atomics or enqueue a
small message rather than block.

Cancellation is cooperative. It is checked between autoregressive tokens, flow
steps, VAE tiles, and transcription windows; an in-flight backend graph may
finish first. A cancelled call returns `YUE2_STATUS_CANCELLED_V1` and leaves its
result initialized and freeable.

## Errors

Every fallible entry returns `yue2_status_v1` and updates the optional fixed-size
`yue2_error_v1`:

| Status | Meaning |
|---|---|
| `OK` | Operation completed |
| `INVALID_ARGUMENT` | Invalid size, enum, value, or ownership state |
| `UNSUPPORTED_ABI` | Reserved for versioned extensions |
| `CANCELLED` | Cooperative cancellation |
| `MODEL_ERROR` | Model resolution, load, backend, or inference failure |
| `IO_ERROR` | File I/O failure |
| `OUT_OF_MEMORY` | Allocation failed |
| `INTERNAL_ERROR` | Unclassified internal failure |

The message buffer requires no allocation and is safe to populate on an
out-of-memory path. Callers should branch on the status code; the message is for
logs and people.

## Loading from a host

On Windows, resolve `yue2_get_api` from `yue2.dll`; on Unix, resolve it from
`libyue2.so`. A host should verify all three of these before using the table:

```c
const yue2_api_v1 *api = get_api(YUE2_ABI_VERSION_1);
if (!api || api->abi_version != YUE2_ABI_VERSION_1 ||
    api->size < YUE2_API_V1_MIN_SIZE) {
    /* incompatible library */
}
```

The CMake shared-library target is enabled with `YUE2_BUILD_SHARED=ON`. Windows
produces `yue2.dll` and `yue2-dll.lib`; Unix produces `libyue2.so`.

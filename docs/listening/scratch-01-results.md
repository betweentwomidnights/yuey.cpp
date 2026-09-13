# Scratch 01: Neon Static

Date: 2026-09-13  
Native commit: `a8efbb5`  
Request: [scratch-01.json](scratch-01.json)

The first from-scratch matrix rendered 768 semantic frames (30.72 seconds) per
sample with the F16 VAE and 32 midpoint flow steps. Each native tier first ran
end to end with an independently sampled ABC plan. Q8_0, Q5_K_M and Q4_K_M
then rendered again with the BF16 plan to isolate post-planning behavior.

## Planning observations

The prompt requested 4/4, exactly 92 BPM, and D minor in free-form tags.

| Planner | Meter | Tempo | Key | ABC bytes | Completed naturally |
|---|---:|---:|---:|---:|---:|
| Native BF16 | 4/4 | 92 | F minor | 1,200 | yes |
| Native Q8_0 | 4/4 | 92 | F minor | 982 | yes |
| Native Q5_K_M | 4/4 | 90 | F minor | 1,337 | yes |
| Native Q4_K_M | 4/4 | 90 | F minor | 1,370 | yes |
| Released Python BF16 | 4/4 | 92 | C minor | 1,196 | yes |

The released Python BF16 planner was run separately with the same prompt,
seed, checkpoint, and official sampling defaults. It also missed D minor. This
shows that the observed key miss is a released-model instruction-adherence
limitation rather than evidence that the native prompt omitted the key or that
quantization alone caused it. Different sampled plans are expected because the
native and PyTorch backends use different random-number implementations; the
test is about requested metadata adherence, not token identity.

## Initial listening result

All native outputs were judged musically good. Vocalist and instrumentation
quality remained good across tiers, with audible but acceptable variation.
No tier was rejected in this first sample. Planning adherence remains a
separate concern even when the resulting song sounds convincing.

## Locked-header proof

Commit `a3c1346` added the low-level `abc_prefix` injection point. Native BF16
was rerun with [scratch-01-locked-header.abc](scratch-01-locked-header.abc)
prefilled immediately after `ABC_START`. The completed 1,588-byte plan preserved
the prefix byte-for-byte and continued with D-minor harmony and pitch material,
beginning `Dm` to `Gm` and later moving through `Bb`. It did not emit a second
header and did not require post-hoc transposition. The completed plan then
conditioned the normal semantic, 32-step flow, and VAE stages for a 30.72-second
listening render.

This validates the injection location. Typed key, BPM, and meter fields still
need a host-side builder and strict ABC validation before this becomes the
high-level gary4local or plugin interface.

## Follow-up gates

- Score requested versus planned key, tempo, and meter explicitly.
- Add prompts without strict metadata so general musical quality is not
  conflated with instruction following.
- Add instrumental, dense transient-heavy, acoustic, and exposed-vocal cases.
- Repeat important prompts across seeds before assigning a failure to a model
  tier.
- Wrap the validated planning prefix with structured key, tempo, meter, and
  unit-length fields for gary4local instead of exposing raw ABC construction to
  UI clients.

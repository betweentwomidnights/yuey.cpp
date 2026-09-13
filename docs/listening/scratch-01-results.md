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

## Follow-up gates

- Score requested versus planned key, tempo, and meter explicitly.
- Add prompts without strict metadata so general musical quality is not
  conflated with instruction following.
- Add instrumental, dense transient-heavy, acoustic, and exposed-vocal cases.
- Repeat important prompts across seeds before assigning a failure to a model
  tier.
- Consider structured key/tempo/meter request fields and constrained or retried
  planning for gary4local instead of relying only on free-form tags.

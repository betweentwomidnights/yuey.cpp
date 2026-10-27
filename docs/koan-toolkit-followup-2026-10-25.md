# Koan AI Toolkit follow-up

This follows the [pinned source review](ai-toolkit-yue2-review-2026-10-25.md).
The existing AR150/NAR400 web UI remains unchanged. Experiments are staged so
label choice and acoustic noise sampling can be judged separately from a larger
scored-data recipe. No GGML submodule changes were made.

## Head audit: retain v9

The v4 head and matched NAR companion were downloaded as safetensors, verified
against the archived Hugging Face LFS SHA-256 values, and converted with the
existing native converters. No remote Python was executed. The v4 head hash is
`ce52f7e4b3e437f204bdac71db685d86f3912e4085659a711387de87cfeb5127`.

Three complete Koan tracks were encoded with both heads before extracting
matched 12/12/10.24-second passages. Head agreement was 33.33%, 25.00% and 30.47%.
The comparison held the exact score/prompt prefix, current encoder binary,
TF32=0, renderer, flow seed 831001 and 32 midpoint steps fixed. Primary arms
used native NAR with the common instrumental AR companion; each matched
head/NAR pair and VAE-only reconstruction were separate controls. No learned
Koan adapter was used. Preview audio was RMS matched per passage with a shared
peak-safe gain; the source was resampled to 48 kHz for previews when necessary.

The user preferred **v9**. Retain it for the follow-up. This is a relative
preference, not an absolute semantic-label quality pass. The user did not rate
every passage/companion separately. Agreement with two historical v9 crops is
99.33% (2/300 IDs differ in each); the current head audit does not mix historical
v9 output with current v4 output. A later controlled TF32 toggle reproduced
both historical crops exactly with TF32 enabled; a repeated TF32=0 extraction
reproduced the current crop exactly. Thus the tiny historical difference can
be explained by this precision setting without changing the head or binary.
The expanded cache already records TF32=0 and remains unchanged. Include the
precision setting in tokenizer cache identities, not only the head filename.

Local listening page: `outputs/koan-toolkit-followup-2026-10-25/listening.md`.

## Acoustic times: retain uniform

Two fresh rank-32 NAR adapters received 100 updates from identical starting
factors on the existing 18 real windows and two neutral replay examples.
AR150, caches, optimizer, learning rate 3e-5, shuffled sample ordering and
Gaussian flow noise were held fixed. Only the noise-time distribution changed.
The new sigmoid option has its own checkpointed RNG and retains a discarded
legacy uniform draw to keep the other random streams paired.

| Schedule | Initial heldout mean | Step-100 heldout mean | sigma .2 | sigma .5 | sigma .8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uniform [.05,.95] | 1.36803 | 1.10094 | 1.21573 | 1.13048 | .956612 |
| Sigmoid(normal) | 1.36803 | 1.10125 | 1.22093 | 1.13011 | .952728 |

These aggregate losses are essentially tied. In the fresh-score matched-token
audition the user preferred **A, uniform**. Retain uniform; this does not
establish a universal schedule preference. The full outside-dataset source
audition is also available but was not separately rated.

Legacy uniform fixture optimizer state is byte-identical to the historical
fixture. All 392 step-100 uniform factors are byte-identical to the old
step-100 experiment. Sigmoid uninterrupted/resumed states are byte-identical,
and a schedule mismatch on resume is rejected. Neither new 100-step checkpoint
was substituted for the current selected NAR400 model in the UI.

Local listening page: `outputs/koan-toolkit-followup-2026-10-25/noise-listening.md`.

## Coverage and actual scored conditioning

The candidate pool contains 34 non-overlapping 512-frame real excerpts across
the same 12 training tracks: 696.32 seconds, compared with the previous 184.32
seconds. Shorter tracks use two rather than three excerpts to avoid overlap.
All six whole-track holdouts remain excluded from training. The pool increases
musical-event coverage; it does not increase the number of distinct tracks.

VAE targets retain two-second source halos and posterior means. Semantic labels
are cropped from the established full-track v9 cache. Every excerpt is
transcribed separately, normalized through the instrumental score path, and
assigned a full scored prefix. Captions retain `koan_palette` plus the existing
track's genre, tempo and key. These are inherited sidecars, not newly verified
descriptions of individual sound-design events.

Structural score triage requires finite/in-range notes, at least eight note
payloads, three pitches and 25% estimated note-time coverage. It admits 32
training windows (655.36 seconds, 10.92 minutes); two are excluded. All six
validation windows are retained, including two that do not pass the density
thresholds. No validation track is silently removed for being difficult.
These checks do not establish pitch/timing accuracy or exact polyphonic scores.

The resource probe used the longest prefix among triage candidates: 618 tokens,
512 semantic frames, rank 32, F32 projections. AR forward/backward used about
6,303 MiB observed device memory; NAR used 4,113 MiB. Frozen BF16 conditioning
for the resource example was prepared on CPU. These are successful finite
forward/backward resource probes, not new independent 512-frame full-model
gradient oracles or throughput guarantees for every prefix.

A fresh bounded semantic AR pilot uses the admitted scored pool with **zero
score dropout**, rank 32, learning rate 2e-5, KL .2, weight decay .01 and global
clip 1. It keeps the common instrumental AR companion and common-model planning.
One neutral generated 512-frame replay is trained; a different seed of the same
user-source score supplies retention-only evaluation. The source recording is
outside the training files; its underlying composition's provenance is unknown.
That is limited
neutral coverage, not two independent compositions. All examples have matching
512-frame lengths. There are at most 200 updates, evaluations/checkpoints every
50 and early stopping after two non-improvements at step >=100.

This fit changes coverage/context, captions and score-dropout curriculum
together. It tests the proposed practical recipe, not which component caused
an audible change. Its direct loss remains semantic-only, with no direct ABC
planning supervision. EOS remains disabled on arbitrary mid-song crops; no
false complete-song targets were introduced. Complete-song/end supervision and
optional full-vocabulary ABC loss remain later work.

## DoRA feasibility, not a validated trainer

`tests/dora_norm_test.cpp` exercises the SA3-style low-rank norm identity with
nonzero A/B factors and independently checks it against a dense adapted matrix.
Numerical gradients cover A, B, output magnitudes and the input, under both full
normalization gradients and Toolkit's detached-denominator convention. A zero-B
adapter initialized with base magnitudes reproduces the base projection.

Both CPU and CUDA pass. Largest numerical-gradient errors were about 3.4e-6/
8.1e-7 on CPU and 8.0e-6/9.0e-6 on CUDA for full/detached conventions. This tiny
shape oracle does not measure real-model DoRA memory/quality or implement YuE2
magnitude export, loading, stacking or the full training path.

An initial guard using CLAMP hit unsupported backward in this GGML version;
the tested graph uses an additive epsilon before sqrt instead. Near-cancellation
stability still needs explicit production treatment. No submodule patch was
necessary for this probe. A real DoRA implementation needs its own full-model
gradient, checkpoint, export and zero-strength inference checks before a fair
NAR LoRA/DoRA fit.

Raw commands, hashes, metrics, feedback and outputs are under
`outputs/koan-toolkit-followup-2026-10-25/`. The final scored-AR checkpoint and
auditions are recorded in the accompanying experiment manifest after selection.

## Scored-only AR result

The bounded pilot stopped at step 200 and selected step
200 by held-out semantic CE, including step zero as an explicit
candidate. Held-out mean changed from 5.897788 to
5.782496 (1.955% reduction). Neutral retention changed
from 3.145244 to 3.150078
(+0.154%). These losses are on the new 512-frame validation examples;
they must not be compared numerically with the old 256-frame experiment's mean.

For a direct checkpoint comparison, old AR150 was separately evaluated on the
identical new prefixes, 512-frame labels and neutral retention example, with
zero optimizer updates. Its held-out mean is 5.839523 and retention is 3.157658.
The new AR200 therefore has about 0.98% lower held-out CE on this same set,
with slightly lower retention CE as well. That remains an objective comparison,
not proof of better audible sound design.

Selected adapter: `outputs/koan-toolkit-followup-2026-10-25/scored-ar/step-200.gguf`.
SHA-256: `b7ca1f20cd9b664f9a7cbfaf47cd7b5af64f9ae95b99e6479287c19e97e7d2b6`.

The fresh-score, full outside-dataset source and standard-generation auditions
compare old AR150 with the new selected AR using the same native NAR. They reuse
the same prompt, score, forced duration, seed and solver. The old AR150/NAR400
render is a separate quality reference. The new adapter was **not** combined
with NAR400 as if that were a calibrated pair. No new adapter was installed in
the running web UI during this pilot. The user subsequently reports that AR200
is picking up the triplets and sounds great. This supports retaining AR200 for
matched acoustic calibration. No separate case-by-case ratings, exact
melody/chord pass or long-form stability claim were supplied.

Local listening page:
`outputs/koan-toolkit-followup-2026-10-25/scored-ar-listening.md`.

All 19 application CTests and 23 Python unittests pass. Full GGML CTest discovery
also lists unbuilt upstream test binaries; the application-only run is the
reported suite. No new full-model numerical gradient oracle was claimed for
the longer context. Source snapshots and command/runtime identities accompany
the raw experiment. Portable results:
[`experiments/yue2-toolkit-followup-2026-10-25.json`](experiments/yue2-toolkit-followup-2026-10-25.json).

The subsequent [matched AR200/NAR400 calibration](koan-ar200-nar-calibration-2026-10-26.md)
completed October 26. The user prefers its calibrated acoustic output; its
combined export and AR200-only comparison are now available in the web UI.

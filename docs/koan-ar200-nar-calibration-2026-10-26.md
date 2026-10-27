# AR200 matched acoustic calibration

Started October 25 and completed October 26, 2026 (America/Phoenix). The user endorsed the expanded scored-only AR200 as picking up the triplets and sounding great. This run freezes that exact AR200 and calibrates NAR against its conditioning states. It follows the [toolkit review experiments](koan-toolkit-followup-2026-10-25.md).

## Data and training

Fresh caches use common instrumental AR plus AR200 (`b7ca1f20cd9b664f9a7cbfaf47cd7b5af64f9ae95b99e6479287c19e97e7d2b6`). Old AR150 conditioning is rejected. All source prefix/code/mean hashes are checked. There are 32 real training windows across 12 tracks (512 frames each; 655.36 seconds), six whole-track holdouts, one neutral generated replay and one separate neutral retention example. All six holdouts are retained, including sparse transcriptions. Real targets are clean posterior-mean VAE latents; semantic labels remain full-context v9. Neutral replay/retention use two seeds of one user-source score, not independent compositions. The recording is outside the training files; underlying composition provenance is unknown.

NAR rank 32 adapts all seven projections in all 28 layers (34,865,152 parameters), with F32 projection arithmetic, learning rate 3e-5, weight decay .01, global clip 1 and uniform sigma [.05,.95]. Samples are shuffled in balanced epochs. AR, VAE and all base weights stay fixed. There is no fixed v9 NAR companion. This is ordinary LoRA, not DoRA.

## Objective selection and listening

| Step | Real heldout flow MSE | Neutral retention MSE |
| ---: | ---: | ---: |
| 0 | 1.282780 | 0.796469 |
| 100 | 1.055680 | 0.837936 |
| 200 | 1.046150 | 0.825502 |
| 300 | 1.041370 | 0.817560 |
| 400 | 1.038750 | 0.812911 |

The 400-update budget completed and selected step 400, including step zero as a candidate. Real heldout error fell by 19.02%; neutral error increased by 2.06%. Both use fixed Gaussian noise and the mean of sigma .2/.5/.8; neutral retention is reported separately and does not select the checkpoint. Loss reduction is not a perceptual-fidelity score.

The user rates calibrated NAR400 better than native NAR in the submitted AR200 comparison. No separate per-case, neutral-control or early-NAR100 ratings were supplied. This is a positive relative quality result, not a full long-form/broad-seed pass.

Matched auditions cover the authored fresh score, full user source, common-model standard-generation score and a second fresh-score seed. Each arm holds AR200, exact saved codec IDs, prefix, flow seed and 32 midpoint steps fixed. The native baseline reproduces the earlier AR200 sample exactly. Compare acoustic quality, palette and musical preservation separately; NAR can change the audible interpretation even though token IDs are unchanged. Preview audio is RMS matched with a common peak-safe gain per case. Forced duration means stopping is not tested.

Local listening set: `outputs/koan-ar200-nar-calibration-2026-10-25/auditions/listening.md`.

## Correctness and resources

Full 512-frame GPU diagnostics pass the independent CPU/PyTorch forward and gradient reference before and after training. Aggregate relative gradient errors are about 4.59e-5 and 5.38e-5, respectively; maximum significant-factor errors are below .03%. Tests also show that adding evaluation-only retention changes neither optimizer nor RNG state, uninterrupted/resumed states match exactly, and legacy uniform states remain byte-identical. `--retention` is evaluation only and excluded from heldout checkpoint selection.

Caches were prepared sequentially on CUDA, peaking at 7803 MiB. Training peaked at 4143 MiB. The six real holdouts and neutral retention have separate metrics. All 19 application CTests and 23 Python unittests pass. No new GGML edits were made; the submodule remains at `b77ae790e628270af46a8af1eca38441c23f1252`.

## Web UI and exports

Refresh [the local web UI](http://127.0.0.1:8007/) and choose `koan-yue2-r32-ar200-nar400`. `koan-yue2-r32-ar200` supplies the AR-only comparison. Older models and the saved default preference remain available; the server did not need restarting. Start with `koan_palette`, strength 1, the common instrumental planner, temperature 1 and 32 ODE steps. The existing optional duration ceiling still applies.

Combined file: `models/koan-yue2-r32-ar200-nar400-F32-LoRA.gguf`. SHA-256: `af757b4a5da8ca29dec29a6127bd9582b3b6fec37ea07cd11aea6bb0604ca80d`. AR-only file: `models/koan-yue2-r32-ar200-F32-LoRA.gguf`.

All 784 exported factors (69,730,304 parameters total) are byte-identical to the selected AR/NAR sources. The combined export produces identical fresh-score semantic IDs and a bit-identical WAV to separate stacking. Both require the fixed common instrumental AR companion and common-model planning before adapted semantic/audio stages.

Live HTTP generation and source-to-score remix pass with the new pair. Zero strength reproduces common-model audio exactly, active style changes the output, metadata names the correct adapter, and both new and old presets are discoverable. These are API/inference checks, not claims of browser listening quality or full long-form performance.

DoRA magnitude checkpoint/export/loading, full-model gradients and a matched LoRA/DoRA fit remain later implementation work. Existing DoRA results are primitive feasibility checks only.

[Portable experiment manifest](experiments/yue2-koan-ar200-nar-calibration-2026-10-26.json). Commands, numerical diagnostics, feedback, source snapshots and raw outputs remain under `outputs/koan-ar200-nar-calibration-2026-10-25/`.

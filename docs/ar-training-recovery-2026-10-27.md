# Windows AR training slowdown and recovery

The `kev-cuda-ooops` UI run used 30-second windows, 750 semantic frames,
1149 prefix slots, LoRA rank 32, LR 2e-5 and teacher KL 0.2 on the local
RTX 5070 Laptop GPU (8 GB), with 64 GB host RAM.

The first five updates took 85.88–85.97 seconds each. GPU usage reached
7797/8151 MiB globally; a separate `ace-server` process held approximately
752 MiB. The trainer had approximately 6822 MiB dedicated and 170 MiB shared
GPU allocation. There was over 21 GB free host RAM.

Unloading ACE-Step freed its allocation, but the existing trainer still took
86.45 seconds for update 6. The UI's cooperative stop checkpointed that update
and retained its optimizer/sampler state. No server termination or model change
was needed.

A fresh process, using the same native executable, GGML CUDA library and
1149-slot shape, completed three controlled updates in 3.626, 3.614 and 3.664
seconds. Its GPU allocation was approximately 6918 MiB dedicated and 74 MiB
shared. The roughly 96 MiB shift from dedicated to shared allocation in the slow
process, together with recovery after releasing the competing allocation and
recreating the trainer, strongly implicates Windows GPU memory residency.
Shared allocation alone does not prove paging or establish a universal threshold.

Shape probes at 1131, 1149 and 1155 prefix slots took 12.33, 12.13 and 11.53
seconds end to end, respectively, including loading, one teacher distribution,
one complete gradient computation and export. The current and previous GGML
CUDA DLLs have identical executable code sections, and AR trainer source matches
the earlier full Koan experiment. No kernel, precision, context or training
hyperparameter change was made to recover performance.

The full UI run was resumed from its step-6 state in `native-3`. Updates 7–9
took 3.858, 3.804 and 3.809 seconds, confirming approximately 22× recovery on
the actual dataset with the original optimizer/sampler state and settings.
Raw local artifacts are in `outputs/ar-slowdown-2026-10-27`, alongside the
original run's immutable requests, logs and saved states.

## Progress and resume fixes

The AR helper now clears previous segment status before native startup and
reports teacher preparation, initial/checkpoint evaluation and last-update
duration. The coordinator ignores unchanged status from an earlier segment
while a newly launched helper hashes its inputs. The web page displays these
activities separately from the saved optimizer step.

An additional resume bug was exposed by the safe stop: initial evaluation in a
new segment may select the checkpoint loaded from the previous segment. Selection
now resolves that adapter/state from its original segment, and rejects conflicting
adapter bytes for the same evaluated step. Tests cover split log writes, resumed
progress, evaluation phases, stale coordinator status and cross-segment selection.

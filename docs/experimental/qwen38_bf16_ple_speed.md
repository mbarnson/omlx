# Qwen3.8 PLE speed: small changes and a maintenance budget

Base: `mbarnson/omlx` main at `24060057ea19dba9bc270171413b7a5c2590eeb5`.
Hardware: 128 GiB M4 Max, with normal background work continuing.

## Retained change

Upload selected BF16 rows by reinterpreting their existing uint16 bits as BF16,
removing the FP32 expansion and cast back to BF16. This adds no cache, thread,
configuration option, file format or resource ownership. A regression test checks
all 65,536 BF16 bit patterns. Existing row and lifecycle tests cover the real
mmap path. This small change was already part of
[PR #3602](https://github.com/jundot/omlx/pull/3602); this branch isolates it,
rather than claiming a new optimization.

Validation of the narrowed branch: 88 tests passed across
`test_mlx_vlm_qwen4_exp_compat.py`, `test_vlm_qwen4_exp_loader.py`,
`test_qwen4_ple_load_resources_cpu.py` and
`test_qwen4_runtime_ple_fork_cpu.py`. Ruff passed for the changed test file,
and `git diff --check` passed.

## Experiments deliberately left out

A bounded 128 MiB exact-row cache passed its parity and lifecycle checks and
avoided 38.16% of row fetches in a cold replay of the earlier 16K-output trace.
However, full-model timing changes were small and mixed. That does not justify
adding persistent state, synchronization, eviction and another memory budget to
the runtime. Its code and raw results are preserved locally as a parked experiment.

The prior GDN overlay was committed and then reverted on this branch. It already
belongs to #3602, and duplicating that work does not address the maintainer's
concern. The branch's net production diff contains only the BF16 upload change.

Jundot's [review](https://github.com/jundot/omlx/pull/3602#issuecomment-5657909730)
reported M3 Ultra decode changes between -1.6% and +1.7%, with identical output
but no clear speed benefit. His concern was maintenance across HC, PLE, GDN and
QSA. Keep future experiments independently measurable and proportional to that
maintenance cost.

## Measurements and their limits

The fixed aggressive checkpoint has Q2 experts, Q4 dense/MTP matrices, four
selected experts, Q8 HC injection, BF16 routers and original BF16 PLE on SSD.
All measurements below included the same previously validated GDN overlay,
including the baseline. They isolate the PLE work, not the whole effect of #3602.
Greedy sampling, MTP depth ceiling 5, prefill chunks 2048, no prefix-cache hits.

| Workload, tokens/s | Baseline | Direct BF16 upload only | Upload + parked row cache |
|---|---:|---:|---:|
| Systems decode | 54.83 | 57.30 | 56.40 |
| Story decode | 49.24 | 47.67 | 49.93 |
| Code decode | 68.33 | 68.70 | 69.12 |
| 2,071-token prefill | 768.52 | 829.66 | 767.61 |
| 8,215-token prefill | 384.47 | 413.03 | 412.11 |

Decode values are medians of two 512-token requests; prefills are single,
repetitive-text measurements. These are provisional, mixed changes under
background load, not a demonstrated large speedup. Systems and code hashes
matched. The uncached story variant produced two different outputs across its
own repeats, consistent with adaptive MTP changing verification widths. A
separate speculation-disabled comparison of legacy upload/no cache against direct
upload/cache produced 1,584 identical output tokens across four requests.

## Eight-bit PLE

Replaying the same recorded row IDs against page-aligned layouts predicted the
following reads with an initially empty, unlimited page cache:

| Layout | Storage | 16K prompt reads | 16K decode-trace reads |
|---|---:|---:|---:|
| BF16 | 102.4 GB | 2.966 GB | 3.590 GB |
| FP8, shared scale | 51.2 GB | 2.902 GB | 3.497 GB |
| Affine Q8 g32, separate scales/biases | 57.6 GB | 6.755 GB | 7.772 GB |

This is a layout simulation, not a quantized-model benchmark. It preserves the
original token/row stream and excludes dequantization cost. FP8 halves row size
but cold reads still fetch 16 KiB pages: modeled traffic fell only 2.18% for
prefill and 2.60% for decode. Separate affine metadata arrays can increase page
faults. The earlier actual BF16 decode measurement was 3.141 GB because some
pages were already resident. Synthetic alignment also differs slightly from
actual safetensors headers.

MLX supports [selected-row FP8 decoding](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.from_fp8.html)
and documents [affine scales/biases](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html).
No quantized PLE checkpoint was created. FP8 remains primarily a storage
experiment here; its output quality and real speed would still need measurement.

## Bound the next search

Prefer existing runtime controls and checkpoint artifacts: MTP depth, prefill
chunk size and native supported quantization formats. Keep the BF16 router and
current model as the comparison baseline. Profile a specific operation before
adding a specialized path. Take independently demonstrated improvements one at
a time; avoid another collection of model-specific serving mechanisms.

For I/O changes, demand exact rows and stable greedy output with speculation off.
For precision changes, use a fixed small set of code, prose, structured-output
and long-context tasks. Check repetition, termination and cases the baseline
already passes. Keep prefill and decode measurements separate, and require a
repeatable substantial benefit before accepting increased maintenance burden.

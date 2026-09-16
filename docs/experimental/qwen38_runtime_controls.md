# Qwen3.8 runtime controls, revision 2

## Outcome

Keep the fixed Q2 target experts / Q4 dense and MTP / top-four checkpoint,
BF16 routers, and original BF16 PLE on SSD. Keep adaptive MTP's ceiling at 5
and the default kernel switches. A 4,096-token prefill chunk is an optional
prefill-oriented setting; its measured advantage is modest. This search did
not establish a substantial, repeatable decode improvement.

No serving-library code changes were added in this revision. The additions
are a benchmark, an isolated experimental launcher, and these measurements.
No new row cache, kernel implementation, or production setting was added.

## Build correction

The initial control screen and earlier experiments imported source checkouts
without optional compiled extensions. They do not represent the user's usual
custom-kernel installation. Discard that screen when choosing settings for
the native build. In particular, do not count rebuilding the already available
kernels as a new optimization over the user's normal installation.

All five existing extensions were built in the isolated worktree using:

```sh
OMLX_WITH_CUSTOM_KERNEL=1 CMAKE_BUILD_PARALLEL_LEVEL=2 \
  python setup.py build_ext --inplace
```

This builds locally without changing the editable-install target of the shared
environment. The build used MLX 0.32.2, nanobind 2.15.0 and Python 3.13.5.
The benchmark refuses to run without the native QSA symbols and records whether
score, top-k and main-attention kernels actually engaged. All three engaged in
the final runs. The user's usual `OMLX_WITH_CUSTOM_KERNEL=1 python -m pip
install -e ...` also builds the kernels; the variable is a build-time switch.

## Native measurements

128 GiB M4 Max, 16 CPU / 40 GPU cores. Background work continued. Greedy,
thinking disabled, one request, no prefix-cache hits, no external GDN overlay.
Long prompts use varied repository text, followed by the same systems question.
They contain 8,265 or 16,457 prompt tokens and produce 512 output tokens.
Prefill means prompt tokens divided by time to first token; decode uses
producer timestamps and excludes the first token.

| Measurement, tokens/s | Depth 5, chunk 2048 | Depth 5, chunk 4096 |
|---|---:|---:|
| 8K prefill, two-run median | 717.2 | 763.6 |
| 16K prefill, two-run median | 782.1 | 809.4 |
| Decode after 8K prompt, two-run median | 45.6 | 45.3 |
| Decode after 16K prompt, two-run median | 45.4 | 47.4 |

The prefill differences are +6.5% and +3.5%. These are tentative under background
load, not a large performance win. The 4K-chunk runs ranged from 733 to 794 tok/s
at 8K; the second also slowed on short decode prompts that fit in either chunk
size. Do not attribute those short-prompt changes to chunk size. A later traced
run verified actual 4,096-token chunks, without cache-boundary subdivision.
All native runs reported approximately 45.68 GiB overall peak MLX allocation;
this is not a measurement of total machine RAM or a 64 GiB capacity test.

Short-prompt baseline decode medians were 56.0 systems, 48.9 story and 70.0 code
tok/s (three observations per task). Alternative control combinations did not
establish an overall decode win:

| Additional screen | 8K prefill | 16K prefill | Decode after 8K / 16K |
|---|---:|---:|---:|
| QSA thresholds 32/8/24, depth 5, chunk 1024 | 736.3 | 675.5 | 46.8 / 48.0 |
| GDN block 16, depth 8, chunk 4096 | 835.8 | 830.4 | 44.2 / 42.0 |
| Aggressive burst, depth 5, chunk 4096 | 824.3 | 858.2 | 45.7 / 47.2 |

Each long-context cell in that screen is one observation. Combined settings
were screened as candidates, not as measurements of each individual flag's
effect. Aggressive burst's repeated short-prompt medians were 55.8 / 47.6 /
68.7 tok/s; no decode gain over balanced mode was established. Burst mode does
not explain the faster prefill observation. Keep balanced as the default.

Raw per-request timing, output hashes, environment values and native dispatch
confirmation are in [the compact results](qwen38_runtime_controls_results.json).
Full generated text, logs, fallback-screen results, build manifests and the API
smoke response remain in the local `work/qwen38-controls-r2` experiment directory.

## Environment variables worth distinguishing

| Control | Behavior on this stack | Decision |
|---|---|---|
| `OMLX_WITH_CUSTOM_KERNEL=1` | Builds optional extensions during installation/build | Required for the intended native baseline |
| `OMLX_QWEN4_EAGER_DISPATCH` | Defaults to 1; overlaps layer graph construction and GPU work | Keep default |
| `OMLX_QWEN4_HC_FUSED` / `OMLX_QWEN4_HC_HYBRID` | Already enabled by default | Keep defaults |
| `OMLX_QWEN4_QSA_NATIVE_SCORE_MIN_ROWS` / `TOPK_MIN_ROWS` / `MAIN_MIN_ROWS` | Native dispatch thresholds; M4 defaults are 0; NAX defaults are 32/8/24 | The combined override did not earn a recommendation |
| `OMLX_QWEN4_GATHERED_MIN_QUERY` | Defaults to 16 for the gathered prefill path; verify eligibility is separate | Not changed |
| `OMLX_GDN_BLOCK_T` | 16/32/48 time blocks; default depends on input dtype and threadgroup memory constraints | The block-16 combination was not a general winner |
| `OMLX_GDN_FUSED_G_BETA=1` | Helper absent in installed mlx-vlm; the switch has no effect here | Do not recommend |
| `OMLX_DECODE_BURST_BUDGET_SINGLE_S` | Direct engine default 0.1 seconds; aggressive mode uses 0.2 | Tested without a demonstrated decode win |

The server CLI overwrites burst environment values from
`server.burst_decode_mode`. Use that saved setting, or the experimental launcher's
`--burst aggressive`, rather than expecting an exported burst budget to survive
CLI initialization. Larger bursts trade streaming granularity for fewer
event-loop handoffs. Do not export a collection of flags merely because they
have performance-related names.

## Output checks and validation

Both depth-5, chunk-4096 repeats matched the corresponding baseline output hashes
for every request: 5,456 generated tokens including warmups. The aggressive
burst run matched another 4,384 tokens. Depth 8 changed some greedy outputs;
changing verification widths can change floating-point results, so exact text
parity across MTP ceilings is not guaranteed.

The small Python function passed six functional checks. The baseline and the
depth-5 variants all failed a JSON filter/sum task identically, omitting one
active record, and added Markdown despite the code-only instruction. These
checks preserve visibility into existing errors; they do not qualify quality.
No model weights were changed during this revision.

The selected native QSA, GDN dispatch and PLE compatibility tests passed:
125 tests, no skips. Ruff and `git diff --check` passed. The new launcher passed
a real localhost `/v1/models` and `/v1/chat/completions` smoke test, returning
106 readable tokens. The temporary server was stopped afterwards.

## Reproduce and serve

Use the built checkout and its MLX-compatible Python environment:

```sh
python benchmarks/bench_qwen4_runtime_controls.py \
  --model /path/to/Qwen3.8-Speed-K4-E2-D4-M4-HC8-BF16PLE \
  --output /tmp/qwen4-controls.json --depth 5 --chunk 4096 \
  --tokens 512 --repeats 2 --prefill 8192 16384 \
  --prefill-repeats 2 --prefill-output-tokens 512

python benchmarks/serve_qwen4_profile.py \
  --model /path/to/Qwen3.8-Speed-K4-E2-D4-M4-HC8-BF16PLE \
  --state-dir /path/to/experimental-server-runs --chunk 4096
```

The launcher creates fresh server state on each invocation and binds localhost
port 8001. It sets the existing `SchedulerConfig.prefill_step_size` through a
launcher-local wrapper because the stock CLI does not expose that field. Its
default remains 2048; use `--chunk 4096` for the modest prefill candidate. The
memory limits are selected for this 128 GiB machine. Neither script is an
upstream serving-feature proposal.

# K2 Horizon and Uno

The native K2 implementation covers dense 0.9B, 3.7B, 7B and 32B, MoVA-36B-A4B,
and the MoE-only 375B-A23B architecture. The cached 0.9B, 3.7B and MoVA BF16
checkpoints have passed load/generation probes. Larger uncached releases are
covered by configuration and reduced numerical fixtures, not real-weight tests.
The 3.7B reference comparison, 0.9B conditional-adapter comparison, and an
8,208-token 0.9B prefill/cache probe pass. These are bounded numerical checks;
they do not certify the full advertised context or uncached checkpoints.

## Selecting Uno

Uno is a conditional diffusion adapter plus its base. It is not a standalone
model or a LoRA that can be permanently fused into the autoregressive weights.
oMLX uses a dedicated serial Uno engine with the released linear two-pass
draft/verify sampler. It retains the original base for verification and applies
the adapter only to noisy draft rows.

When the Hugging Face Hub cache is a configured model directory, these official
adapter snapshots are recognized automatically:

| Adapter | Required local base revision | Uno context ceiling |
| --- | --- | --- |
| IFM/K2-Horizon-0.9B-Uno | `ee770e713760cf6350e4322cdbbff91a163b7d70` | 131072 |
| IFM/K2-Horizon-7B-Uno | `586b03f0fd1fbbf2f13eeafc33749e95ae34dd10` | 262144 |

Both base and adapter files must already be complete. Resolution never downloads
weights. A missing base or shard produces a discovery diagnostic. The 7B Uno
ceiling follows its released inference recipe; it is lower than the base's
declared context. These are admission ceilings, not claims of tested context.

The cache-backed API model ID is `IFM--K2-Horizon-0.9B-Uno`. The ordinary base
remains separately selectable as `IFM--K2-Horizon-0.9B`.

For local exports, create a directory under a configured model directory with
`uno_config.json`:

```json
{
  "format": "k2_uno",
  "version": 1,
  "base_model_id": "IFM/K2-Horizon-0.9B",
  "base_path": "/absolute/path/to/base",
  "adapter_path": "/absolute/path/to/adapter",
  "block_size": 8
}
```

Relative paths resolve against that registration directory. Block size must be
1–64; the default is 8. Unrelated PEFT adapters remain unsupported. There is no
released MoVA Uno adapter, and dense adapters cannot be used with MoVA.

## Current serving capabilities

- BF16 base and adapter execution; strict tensor, shape, dtype and PEFT-subset checks.
- Completions, chat, streaming, IFM reasoning markers, tool calls and tool-result follow-up.
- Temperature, top-p, top-k, request-local seeds, output limits and stop strings.
- Serialized requests, cancellable chunked prefill, request-owned KV, and pool unload/reload.
- Memory admission includes both artifacts, future KV, and proposal/logit workspace.

The initial Uno engine does not implement quantization, persistent prefix caching,
continuous batching, tree proposals, MTP/DFlash composition, or distributed
execution. Unsupported sampling penalties, min-p, XTC, thinking budgets and
grammar constraints are rejected rather than silently changing the sampler.
Responses API requests for output logprobs are also rejected before streaming.
Reasoning effort through the chat template is supported separately from a forced
thinking-token budget. Cached-token usage is always zero for Uno.

Base AR and Uno engines own separate model instances. Keeping both loaded costs
memory for both bases; pool admission accounts for that. A high proposal
acceptance rate alone does not establish a wall-clock speedup on Apple Silicon.

## Local validation

The probe tools accept existing snapshots and write evidence to an explicitly
chosen output path:

```sh
python tools/k2_uno_serving_probe.py --adapter /path/to/cached/uno/snapshot --out /tmp/uno-engine.json
python tools/k2_uno_http_probe.py --cache /path/to/huggingface/hub --out /tmp/uno-http.json
python tools/k2_uno_http_probe.py --cache /path/to/huggingface/hub --out /tmp/uno-tcp.json --tcp
python tools/k2_protocol_probe.py --cache /path/to/huggingface/hub --model IFM--K2-Horizon-3.7B --out /tmp/k2-protocols.json
```

The HTTP probe exercises production routes through an in-process ASGI transport,
including actual pool loading, tools, stop strings, seed reproducibility, base/Uno
alternation, errors before streaming, and reload. It does not start or modify a
separate running oMLX server. Real 7B Uno execution remains unverified.
With `--tcp`, it starts an isolated loopback listener and additionally closes an
active response stream, checks producer cleanup, and verifies the next request.
The cached 0.9B Uno passed both transport modes.

The protocol probe accepts both base and Uno model IDs. It exercises three
prompt types with repeated and streamed requests, low/medium/high reasoning
effort, assistant history, streamed tools, and Responses/Anthropic Messages in
both streaming and non-streaming modes. The cached 0.9B base, 3.7B base, and
0.9B Uno passed that matrix. It validates API behavior, not benchmark accuracy.

Numerical evidence includes all 196 real 0.9B LoRA projection pairs against
independent torch calculations, plus draft and verifier logits for three fixed
blocks. The largest projection relative RMS error was 0.0284%; every draft and
verifier top-token prediction matched on those blocks. This does not establish
cross-framework identity for arbitrary generated sequences. BF16 matrix shapes
can change near-tie greedy decisions even with the adapter disabled.

375B uses the released HF source's full-width BF16 router linear. MoVA retains
its separately verified two-part router calculation. A reduced 375B fixture
compares routing and expert output with the released torch class; real 375B
weights have not been loaded here.

A separate `--prefix-cache-only --ssd-cache-dir /path/to/isolated/cache` probe
validates persistent base KV restoration across engine unload/reload. Both cached
0.9B and 3.7B restored 768 prompt tokens and preserved the response. This base
cache capability does not enable persistent prefix caching for Uno.

The independent full torch/MPS sampler completed nine runs across three prompts
and three seeds. Independent replay passed all 370 native cycles, including
acceptance, correction distributions and committed cache frontiers. All 740
captured nucleus supports satisfy the threshold contract; tied BF16 logits can
produce different valid boundary supports across frameworks. Three categorical
distribution checks used 65,536 samples each and passed the predeclared
family-wise error bound. These checks establish sampler behavior, not general
answer accuracy or identical cross-framework random sequences.

On the tested M4 Max, five timed trials per prompt/lane with 64 fixed output
tokens gave native AR 95.9–98.4 tok/s and block-8 Uno 90.5–101.6 tok/s across
the three prompts, or 0.943–1.032x the matched AR throughput. EOS stopping was
disabled in both lanes, and loading/tokenization were excluded. There was no
consistent speedup. Instrumented draft/verify/rollback profiles are separate
from these timings; evaluation wait includes GPU execution. No ANE execution
or acceleration is implemented for Uno.

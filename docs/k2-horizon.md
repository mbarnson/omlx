# K2 Horizon and Uno

K2 Horizon supports dense and routed architectures. Real BF16 checkpoints for
0.9B, 3.7B and MoVA-36B-A4B have passed load and generation checks. Dense 7B/32B,
MoE 375B-A23B and 7B Uno have configuration or reduced-fixture coverage; their
real checkpoints remain unverified.

## Pair Uno with its base

Uno is a conditional diffusion adapter. Selecting it loads the compatible IFM
base and adapter into one dedicated engine. Draft passes apply the adapter to
noisy rows; verification uses the unchanged base weights. Do not fuse the
adapter permanently or load it through the ordinary mlx-lm LoRA loader.

Add your Hugging Face Hub cache (normally `~/.cache/huggingface/hub`) as an oMLX
model directory. Complete local snapshots of both components are required:

| Uno adapter | Required base | Automatic base revision |
| --- | --- | --- |
| IFM/K2-Horizon-0.9B-Uno | IFM/K2-Horizon-0.9B | `ee770e713760cf6350e4322cdbbff91a163b7d70` |
| IFM/K2-Horizon-7B-Uno | IFM/K2-Horizon-7B | `586b03f0fd1fbbf2f13eeafc33749e95ae34dd10` |

Discovery resolves the pair locally and never downloads weights. Missing bases
or shards produce a diagnostic. Select `IFM--K2-Horizon-0.9B-Uno` in an API
request to use the pair; select `IFM--K2-Horizon-0.9B` for ordinary autoregressive
inference. The base needs no separate load request before using Uno.

```json
{
  "model": "IFM--K2-Horizon-0.9B-Uno",
  "messages": [{"role": "user", "content": "Write a Python palindrome function."}],
  "max_tokens": 128,
  "temperature": 0
}
```

For local exports or snapshots elsewhere, create a directory under a configured
model directory containing `uno_config.json`:

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

Select that directory's model ID. Relative paths resolve from the registration
directory. Block size defaults to 8 and must be 1–64. Base identity, adapter
metadata, tensor shapes and BF16 dtypes are validated before use. Unrelated PEFT
adapters remain unsupported. There is no released MoVA Uno adapter; a dense Uno
adapter cannot be paired with MoVA or 3.7B.

## Serving behavior and limits

Uno supports completions, chat, streaming, IFM reasoning markers, tools and tool
follow-up, temperature/top-p/top-k, request-local seeds, stop strings and output
limits. Requests are serialized, KV is request-local, and only committed tokens
are streamed. Cancellation and pool unload/reload are supported.

The engine requires unquantized BF16 weights. It does not support continuous
batching, persistent prefix caching, distributed execution, MTP/DFlash
composition, tree proposals, sampling penalties, min-p, XTC, forced thinking
budgets, grammar constraints or output logprobs. Unsupported controls are
rejected. Template reasoning effort is supported; cached-token usage is zero.
Uno context admission is capped at 131072 tokens for 0.9B and 262144 for 7B;
these ceilings are not validated full-context claims.

Ordinary base and Uno selections own separate instances when both are loaded.
Memory admission includes both base copies, the adapter and request workspace.
On an M4 Max, block-8 GPU Uno measured 90.5–101.6 tok/s versus matched stock AR
at 95.9–98.4 tok/s across three prompts. Five trials used 64 fixed output tokens,
with EOS stopping disabled and loading/tokenization excluded. There was no
consistent throughput benefit; measure your own workload.

## Validation

Tests cover dense/routed math, conditional LoRA, draft/verify sampling, discovery,
request validation and engine lifecycle. Cached 0.9B Uno passed independent
projection/logit comparisons and sampler replay, plus real HTTP streaming,
tools, base/Uno alternation, disconnect cleanup and reload. These are bounded
numerical and protocol checks, not general answer-quality guarantees.

Run probes with existing snapshots and an explicit output path:

```sh
python tools/k2_uno_http_probe.py --cache /path/to/huggingface/hub --out /tmp/uno-http.json --tcp
python tools/k2_uno_serving_probe.py --adapter /path/to/uno/snapshot --out /tmp/uno-engine.json
python tools/k2_protocol_probe.py --cache /path/to/huggingface/hub --model IFM--K2-Horizon-0.9B-Uno --out /tmp/uno-protocols.json
```

The HTTP probe starts an isolated loopback listener. It does not modify a running
oMLX server. Base-model prefix-cache probes are separate; their success does not
enable persistent caching in the Uno engine.

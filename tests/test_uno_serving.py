# SPDX-License-Identifier: Apache-2.0
"""Uno registration, serving protocol, and executor lifetime contracts."""

import asyncio
import json
import struct
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omlx.engine.base import GenerationOutput
from omlx.engine.uno import UnoEngine, _StopBuffer
from omlx.exceptions import InvalidRequestError
from omlx.model_discovery import discover_models
from omlx.uno_bundle import RELEASED_BASES, resolve_uno_bundle


def write_weights(path):
    header = json.dumps(
        {"weight": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}}
    ).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\x00\x00")


@pytest.fixture
def cache(tmp_path):
    base_id = "IFM/K2-Horizon-0.9B"
    base = (
        tmp_path
        / "models--IFM--K2-Horizon-0.9B"
        / "snapshots"
        / RELEASED_BASES[base_id]
    )
    adapter = (
        tmp_path / "models--IFM--K2-Horizon-0.9B-Uno" / "snapshots" / "adapter-revision"
    )
    base.mkdir(parents=True)
    adapter.mkdir(parents=True)
    (base / "config.json").write_text(
        json.dumps(
            {
                "model_type": "k2_horizon",
                "max_position_embeddings": 4096,
                "vocab_size": 100,
            }
        )
    )
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": base_id})
    )
    write_weights(base / "model.safetensors")
    write_weights(adapter / "adapter_model.safetensors")
    return tmp_path, base, adapter


def test_cache_resolution_and_discovery_charge_both_artifacts(cache):
    root, base, adapter = cache
    bundle = resolve_uno_bundle(adapter)
    assert bundle.base_path == base
    assert (
        bundle.estimated_size
        == (base / "model.safetensors").stat().st_size
        + (adapter / "adapter_model.safetensors").stat().st_size
    )
    found = discover_models(root)
    assert found["IFM--K2-Horizon-0.9B"].engine_type == "batched"
    uno = found["IFM--K2-Horizon-0.9B-Uno"]
    assert uno.engine_type == "uno" and not uno.is_helper
    assert uno.estimated_size == bundle.estimated_size
    assert uno.model_context_length == 4096


@pytest.mark.parametrize("layout", ["direct", "flat", "organized"])
def test_local_descriptor_layouts(cache, tmp_path, layout):
    _, base, adapter = cache
    root = tmp_path / "registry"
    entry = (
        root
        if layout == "direct"
        else root / ("IFM/uno" if layout == "organized" else "uno")
    )
    entry.mkdir(parents=True)
    (entry / "uno_config.json").write_text(
        json.dumps(
            {
                "format": "k2_uno",
                "version": 1,
                "base_model_id": "IFM/K2-Horizon-0.9B",
                "base_path": str(base),
                "adapter_path": str(adapter),
                "block_size": 4,
            }
        )
    )
    found = discover_models(root)
    assert len(found) == 1
    assert next(iter(found.values())).engine_type == "uno"
    assert resolve_uno_bundle(entry).block_size == 4


def test_missing_base_is_actionable_and_never_downloads(cache, caplog):
    root, base, adapter = cache
    (base / "model.safetensors").unlink()
    assert "IFM--K2-Horizon-0.9B-Uno" not in discover_models(root)
    assert "No K2 base safetensors" in caplog.text
    with pytest.raises(FileNotFoundError):
        resolve_uno_bundle(adapter)


def test_unrelated_peft_stays_unsupported(tmp_path):
    adapter = tmp_path / "ordinary-lora"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {"base_model_name_or_path": "IFM/K2-Horizon-0.9B", "task_type": "CAUSAL_LM"}
        )
    )
    write_weights(adapter / "adapter_model.safetensors")
    assert discover_models(tmp_path) == {}
    with pytest.raises(ValueError, match="official HF cache"):
        resolve_uno_bundle(adapter)


@pytest.mark.parametrize(
    "config",
    [
        {"num_experts": 4},
        {"attention_gate_func": "softplus"},
        {"quantization": {"bits": 4}},
    ],
)
def test_incompatible_base_rejected(cache, config):
    _, base, adapter = cache
    path = base / "config.json"
    data = json.loads(path.read_text())
    data.update(config)
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        resolve_uno_bundle(adapter)


def test_7b_uno_metadata_and_context_limit(tmp_path):
    fixture = Path(__file__).parent / "fixtures" / "k2_horizon"
    adapter_config = json.loads((fixture / "7B-Uno-adapter_config.json").read_text())
    assert adapter_config["r"] == 128
    assert adapter_config["lora_alpha"] / adapter_config["r"] == 64
    base_id = adapter_config["base_model_name_or_path"]
    assert base_id == "IFM/K2-Horizon-7B"
    base = (
        tmp_path / "models--IFM--K2-Horizon-7B" / "snapshots" / RELEASED_BASES[base_id]
    )
    adapter = (
        tmp_path / "models--IFM--K2-Horizon-7B-Uno" / "snapshots" / "adapter-revision"
    )
    base.mkdir(parents=True)
    adapter.mkdir(parents=True)
    (base / "config.json").write_text((fixture / "7B.json").read_text())
    (adapter / "adapter_config.json").write_text(json.dumps(adapter_config))
    write_weights(base / "model.safetensors")
    write_weights(adapter / "adapter_model.safetensors")
    bundle = resolve_uno_bundle(adapter)
    assert bundle.config["max_position_embeddings"] == 524288
    assert bundle.context_length == 262144


def test_stop_buffer_split_overlap_and_final_flush():
    buffer = _StopBuffer(["END", "ENDING"])
    assert buffer.feed("hello E") == "hello "
    assert buffer.feed("N") == ""
    assert buffer.feed("D ignored") == "" and buffer.stopped
    assert buffer.feed("anything", final=True) == ""
    buffer = _StopBuffer(["END"])
    assert buffer.feed("E") == ""
    assert buffer.feed("", final=True) == "E"


@pytest.fixture
def engine():
    engine = UnoEngine("test")
    engine._tokenizer = SimpleNamespace(encode=lambda prompt: [3, 4])
    engine._bundle = SimpleNamespace(
        config={"vocab_size": 100}, context_length=4096, block_size=8
    )
    engine._prefill_guard = MagicMock(_last_mlx_active_memory_bytes=0)
    return engine


@pytest.mark.parametrize(
    "options",
    [
        {"min_p": 0.1},
        {"presence_penalty": 1},
        {"frequency_penalty": 1},
        {"compiled_grammar": object()},
        {"thinking_budget": 10},
        {"top_logprobs": 0},
        {"logprobs": True},
        {"seed": -1},
        {"max_tokens": -1},
        {"top_k": 101},
        {"max_tokens": 5000},
    ],
)
def test_unsupported_controls_and_limits_rejected(engine, options):
    with pytest.raises(InvalidRequestError):
        engine._preflight([3, 4], **options)


def test_only_consumed_tokens_count_toward_stop_and_acceptance(engine, monkeypatch):
    import omlx.engine.uno as module
    from omlx.adapter.output_parser import (
        OutputParserFinalizeResult,
        OutputParserTokenResult,
    )
    from omlx.patches.k2_horizon.uno_decode import UnoCycle

    class Decoder:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, *args, **kwargs):
            yield UnoCycle((10, 11, 12, 13), 3, 7, 2, 5, "length")

    monkeypatch.setattr(module, "UnoDecoder", Decoder)
    engine._executor_tokenizer = SimpleNamespace(
        decode=lambda ids: "prompt", eos_token_ids={99}
    )
    parser = SimpleNamespace(
        process_token=lambda token: OutputParserTokenResult(
            stream_text={10: "hello ", 11: "EN", 12: "D", 13: "unseen"}[token]
        ),
        finalize=lambda: OutputParserFinalizeResult(),
    )
    engine._output_parser_factory = SimpleNamespace(
        thinking_marker_pairs=(),
        create_session_with_tools=lambda tokenizer, tools: parser,
    )
    outputs = []
    engine._run(
        [3, 4],
        dict(max_tokens=8, temperature=0.0, top_p=1.0, top_k=0),
        ["END"],
        threading.Event(),
        outputs.append,
    )
    assert outputs[-1].text == "hello "
    assert outputs[-1].tokens == [10, 11, 12]
    assert outputs[-1].completion_tokens == 3
    assert outputs[-1].finish_reason == "stop"
    assert engine._last_speculation["accepted_proposals"] == 2


@pytest.mark.asyncio
async def test_stream_and_nonstream_share_output(engine):
    def run(ids, options, stops, event, publish):
        publish(GenerationOutput(text="a", new_text="a", tokens=[5], finished=False))
        publish(
            GenerationOutput(
                text="ab", new_text="b", tokens=[5, 6], completion_tokens=2
            )
        )

    engine._run = run
    chunks = [chunk async for chunk in engine.stream_generate("prompt")]
    result = await engine.generate("prompt")
    assert "".join(chunk.new_text for chunk in chunks) == result.text == "ab"
    assert result.completion_tokens == 2
    assert not engine.has_active_requests()


@pytest.mark.asyncio
async def test_close_stream_cancels_worker_before_unlock(engine):
    ended = threading.Event()

    def run(ids, options, stops, event, publish):
        try:
            publish(GenerationOutput(text="a", new_text="a", finished=False))
            event.wait(5)
        finally:
            ended.set()

    engine._run = run
    stream = engine.stream_generate("prompt")
    await anext(stream)
    assert engine.has_active_requests()
    await asyncio.wait_for(stream.aclose(), timeout=2)
    assert ended.is_set() and not engine._lock.locked()
    assert not engine.has_active_requests()


@pytest.mark.asyncio
async def test_external_stop_without_terminal_publication_does_not_hang(engine):
    entered = threading.Event()

    def run(ids, options, stops, event, publish):
        entered.set()
        event.wait(5)

    engine._run = run
    task = asyncio.create_task(engine.generate("prompt"))
    await asyncio.to_thread(entered.wait, 2)
    await asyncio.wait_for(engine.stop(), timeout=2)
    with pytest.raises(RuntimeError, match="without a completion"):
        await asyncio.wait_for(task, timeout=2)


@pytest.mark.asyncio
async def test_stop_can_unload_while_consumer_is_paused_at_yield(engine):
    ended = threading.Event()

    def run(ids, options, stops, event, publish):
        publish(GenerationOutput(text="a", new_text="a", finished=False))
        event.wait(5)
        ended.set()

    engine._run = run
    stream = engine.stream_generate("prompt")
    await anext(stream)
    await asyncio.wait_for(engine.stop(), timeout=2)
    assert ended.is_set() and not engine.has_active_requests()
    await stream.aclose()


@pytest.mark.asyncio
async def test_requests_are_serialized_and_queue_cancellation_is_local(engine):
    entered = threading.Event()
    release = threading.Event()
    count = 0

    def run(ids, options, stops, event, publish):
        nonlocal count
        count += 1
        entered.set()
        release.wait(5)
        publish(GenerationOutput(text="done", completion_tokens=1))

    engine._run = run
    first = asyncio.create_task(engine.generate("first"))
    await asyncio.to_thread(entered.wait, 2)
    second = asyncio.create_task(engine.generate("second"))
    await asyncio.sleep(0.01)
    second.cancel()
    release.set()
    assert (await asyncio.wait_for(first, 2)).text == "done"
    with pytest.raises(asyncio.CancelledError):
        await second
    assert count == 1
    assert not engine.has_active_requests()


@pytest.mark.parametrize(
    "options",
    [
        {"response_format": {"type": "json_object"}},
        {"structured_outputs": {"json_schema": {"type": "object"}}},
        {"guided_grammar": 'root ::= "ok"'},
    ],
)
def test_uno_grammar_requests_never_degrade_to_prompt_injection(engine, options):
    from omlx.server import _reject_diffusion_structured_outputs

    with pytest.raises(InvalidRequestError, match="grammar-constrained"):
        _reject_diffusion_structured_outputs(engine, **options)


@pytest.mark.asyncio
async def test_pool_dispatch_and_override_preserve_uno(cache):
    from omlx.engine_pool import EnginePool

    root, _, _ = cache
    pool = EnginePool()
    pool.discover_models(str(root))
    settings = MagicMock()
    settings.get_settings.return_value.model_type_override = "llm"
    pool.apply_settings_overrides(settings)
    entry = pool.get_entry("IFM--K2-Horizon-0.9B-Uno")
    assert entry.engine_type == "uno"
    mock_engine = MagicMock()
    mock_engine.start = AsyncMock()
    mock_engine.stop = AsyncMock()
    with (
        patch("omlx.engine.uno.UnoEngine", return_value=mock_engine),
        patch("omlx.engine_pool.BatchedEngine") as ar,
    ):
        pool._raise_if_model_path_missing_locked(entry.model_id, entry)
        await pool._load_engine(entry.model_id)
        assert entry.engine is mock_engine
        ar.assert_not_called()

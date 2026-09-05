# SPDX-License-Identifier: Apache-2.0
"""Exercise cached K2 bases or Uno through production HTTP protocol routes.

The ASGI transport uses the real pool and engines without touching a running
server. All requests and responses, including unsuccessful checks, are saved.
"""

import argparse
import asyncio
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path

import httpx

from omlx.engine_pool import EnginePool
from omlx.scheduler import SchedulerConfig


def sse_events(text):
    return [
        json.loads(line[6:])
        for line in text.splitlines()
        if line.startswith("data: ") and line[6:] != "[DONE]"
    ]


def chat_stream(events):
    text, reasoning, calls = "", "", {}
    finish, usage = None, None
    for event in events:
        if event.get("usage"):
            usage = event["usage"]
        for choice in event.get("choices", []):
            delta = choice.get("delta", {})
            text += delta.get("content") or ""
            reasoning += delta.get("reasoning_content") or ""
            finish = choice.get("finish_reason") or finish
            for call in delta.get("tool_calls", []):
                entry = calls.setdefault(call["index"], {"name": "", "arguments": ""})
                for key in entry:
                    entry[key] += call.get("function", {}).get(key) or ""
    return dict(text=text, reasoning=reasoning, calls=calls, finish=finish, usage=usage)


async def probe(args):
    import omlx.server as server

    if args.prefix_cache_only and args.ssd_cache_dir is None:
        raise ValueError("--prefix-cache-only requires an isolated --ssd-cache-dir")
    pool = EnginePool(
        scheduler_config=SchedulerConfig(
            paged_ssd_cache_dir=str(args.ssd_cache_dir) if args.ssd_cache_dir else None,
            paged_ssd_cache_max_size=1024**3,
        )
    )
    pool.discover_models(str(args.cache))
    previous = server._server_state
    server._server_state = server.ServerState(
        engine_pool=pool, default_model=args.model
    )
    report = {
        "model": args.model,
        "transport": "in-process HTTP ASGI, production pool and engines",
        "platform": platform.platform(),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "versions": {
            name: importlib.metadata.version(name) for name in ("mlx", "mlx-lm")
        },
        "requests": [],
        "checks": {},
        "passed": False,
    }
    tool = {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app), base_url="http://localhost"
        ) as client:

            async def post(label, route, body, expected=200):
                request = {"model": args.model, "temperature": 0, **body}
                started = time.perf_counter()
                response = await client.post(route, json=request)
                payload = (
                    sse_events(response.text)
                    if response.headers.get("content-type", "").startswith(
                        "text/event-stream"
                    )
                    else response.json()
                )
                row = {
                    "label": label,
                    "route": route,
                    "request": request,
                    "status": response.status_code,
                    "elapsed_seconds": time.perf_counter() - started,
                    "response": payload,
                }
                report["requests"].append(row)
                print(json.dumps(row), flush=True)
                assert response.status_code == expected, row
                return payload

            models = await client.get("/v1/models")
            models.raise_for_status()
            assert args.model in [item["id"] for item in models.json()["data"]]
            if args.prefix_cache_only:
                if pool.get_entry(args.model).engine_type == "uno":
                    raise ValueError("Uno deliberately has no persistent prefix cache")
                body = {
                    "max_tokens": 64,
                    "reasoning_effort": "low",
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "Reference notes:\n"
                                + "Canberra is the capital of Australia.\n" * 100
                                + "\nUsing those notes, name the capital of Australia. Keep the answer short."
                            ),
                        }
                    ],
                }
                cold = await post("prefix-cold", "/v1/chat/completions", body)
                # Shutdown drains the cache writer. Reload must restore persisted
                # blocks into a new engine rather than reuse the old live KV.
                await pool._unload_engine(args.model)
                restored = await post(
                    "prefix-ssd-restore", "/v1/chat/completions", body
                )
                assert cold["choices"] == restored["choices"]
                assert restored["choices"][0]["finish_reason"] == "stop"
                assert "Canberra" in restored["choices"][0]["message"]["content"]
                assert (
                    restored["usage"]["prompt_tokens_details"]["cached_tokens"] >= 256
                )
                report["checks"]["persistent_prefix_cache_restore"] = True
                report["passed"] = True
                return
            prompts = [
                "What is the capital of Australia? Answer in one sentence.",
                "Write a Python function that adds two integers. Keep it short.",
                "I have 3 boxes with 4 apples each and give away 5 apples. How many remain? Explain briefly.",
            ]
            for index, prompt in enumerate(prompts):
                body = {
                    "messages": [{"role": "user", "content": prompt}],
                    "reasoning_effort": "low",
                    "max_tokens": 256,
                    "seed": 42,
                }
                first = await post(f"chat-{index}", "/v1/chat/completions", body)
                warm = await post(f"chat-{index}-warm", "/v1/chat/completions", body)
                streamed = chat_stream(
                    await post(
                        f"chat-{index}-stream",
                        "/v1/chat/completions",
                        {
                            **body,
                            "stream": True,
                            "stream_options": {"include_usage": True},
                        },
                    )
                )
                choice = first["choices"][0]
                assert choice["message"].get("content")
                assert choice["finish_reason"] == "stop", "Output budget exhausted"
                assert first["choices"] == warm["choices"]
                assert streamed["text"].strip() == choice["message"]["content"].strip()
                assert streamed["finish"] == choice["finish_reason"]
                assert (
                    streamed["usage"]["completion_tokens"]
                    == first["usage"]["completion_tokens"]
                )
            report["checks"]["chat_cold_warm_stream_three_prompts"] = True

            completion = await post(
                "completion",
                "/v1/completions",
                {"prompt": "The capital of Australia is", "max_tokens": 16},
            )
            assert completion["choices"][0]["text"].strip()
            history = await post(
                "assistant-history",
                "/v1/chat/completions",
                {
                    "max_tokens": 128,
                    "reasoning_effort": "low",
                    "messages": [
                        {"role": "user", "content": "Remember the code word: ORCHID."},
                        {"role": "assistant", "content": "The code word is ORCHID."},
                        {"role": "user", "content": "What was the code word?"},
                    ],
                },
            )
            assert "ORCHID" in history["choices"][0]["message"]["content"].upper()
            report["checks"]["completion_and_assistant_history"] = True

            for effort in ("medium", "high"):
                answer = await post(
                    f"reasoning-{effort}",
                    "/v1/chat/completions",
                    {
                        "messages": [{"role": "user", "content": "What is 2 + 2?"}],
                        "max_tokens": 384,
                        "reasoning_effort": effort,
                    },
                )
                assert answer["choices"][0]["finish_reason"] == "stop"
                assert "4" in answer["choices"][0]["message"]["content"]
            report["checks"]["reasoning_efforts"] = True

            weather_prompt = "Use get_weather to check the weather in Paris."
            events = await post(
                "stream-tools",
                "/v1/chat/completions",
                {
                    "messages": [{"role": "user", "content": weather_prompt}],
                    "tools": [{"type": "function", "function": tool}],
                    "max_tokens": 192,
                    "reasoning_effort": "low",
                    "stream": True,
                },
            )
            streamed = chat_stream(events)
            assert streamed["finish"] == "tool_calls"
            assert len(streamed["calls"]) == 1
            call = next(iter(streamed["calls"].values()))
            assert call["name"] == "get_weather"
            assert json.loads(call["arguments"])["city"] == "Paris"
            report["checks"]["openai_stream_tools"] = True

            for stream in (False, True):
                response = await post(
                    f"responses-{stream}",
                    "/v1/responses",
                    {
                        "input": prompts[0],
                        "max_output_tokens": 192,
                        "reasoning": {"effort": "low"},
                        "stream": stream,
                    },
                )
                if stream:
                    completed = [
                        e for e in response if e.get("type") == "response.completed"
                    ]
                    assert len(completed) == 1
                    response = completed[0]["response"]
                assert response["status"] == "completed"
                assert response["output"] and response["usage"]["output_tokens"] > 0
                response = await post(
                    f"anthropic-{stream}",
                    "/v1/messages",
                    {
                        "messages": [{"role": "user", "content": prompts[0]}],
                        "max_tokens": 192,
                        "stream": stream,
                        "chat_template_kwargs": {"reasoning_effort": "low"},
                    },
                )
                if stream:
                    assert response[-1]["type"] == "message_stop"
                    assert any(e.get("type") == "content_block_delta" for e in response)
                else:
                    assert response["stop_reason"] == "end_turn"
                    assert any(b.get("text") for b in response["content"])
            report["checks"]["responses_and_anthropic_stream_nonstream"] = True

            if getattr(pool.get_entry(args.model).engine, "is_uno_model", False):
                for options in (
                    {"top_logprobs": 0},
                    {"include": ["message.output_text.logprobs"]},
                ):
                    await post(
                        f"responses-logprobs-rejected-{next(iter(options))}",
                        "/v1/responses",
                        {
                            "input": "Hello",
                            "max_output_tokens": 16,
                            "stream": True,
                            **options,
                        },
                        400,
                    )
                await post(
                    "responses-grammar-rejected",
                    "/v1/responses",
                    {
                        "input": "Hello",
                        "max_output_tokens": 16,
                        "stream": True,
                        "text": {"format": {"type": "json_object"}},
                    },
                    400,
                )
                await post(
                    "anthropic-budget-rejected",
                    "/v1/messages",
                    {
                        "messages": [{"role": "user", "content": "Hello"}],
                        "max_tokens": 2048,
                        "stream": True,
                        "thinking": {"type": "enabled", "budget_tokens": 1024},
                    },
                    400,
                )
                report["checks"]["alternate_api_preflight_rejections"] = True
            report["passed"] = True
    finally:
        await pool.shutdown()
        server._server_state = previous
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ssd-cache-dir", type=Path)
    parser.add_argument("--prefix-cache-only", action="store_true")
    asyncio.run(probe(parser.parse_args()))


if __name__ == "__main__":
    main()

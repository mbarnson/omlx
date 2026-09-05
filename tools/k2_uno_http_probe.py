# SPDX-License-Identifier: Apache-2.0
"""Run production oMLX HTTP routes against a cached Uno engine, in process."""

import argparse
import asyncio
import importlib.metadata
import json
import platform
import socket
import subprocess
from pathlib import Path

import httpx

from omlx.engine_pool import EnginePool


async def probe(args):
    import omlx.server as server

    pool = EnginePool()
    pool.discover_models(str(args.cache))
    state = server.ServerState(engine_pool=pool, default_model=args.model)
    previous = server._server_state
    server._server_state = state
    listener = http_server = http_task = None
    client_options = {
        "transport": httpx.ASGITransport(app=server.app),
        "base_url": "http://localhost",
        "timeout": 120,
    }
    report = {
        "transport": "ASGI HTTP, production routes",
        "model": args.model,
        "requests": [],
        "platform": platform.platform(),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("mlx", "mlx-lm", "transformers")
        },
    }
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    try:
        if args.tcp:
            import uvicorn

            listener = socket.socket()
            listener.bind(("127.0.0.1", 0))
            listener.listen(128)
            http_server = uvicorn.Server(
                uvicorn.Config(server.app, lifespan="off", log_level="warning")
            )
            http_task = asyncio.create_task(http_server.serve(sockets=[listener]))
            async with asyncio.timeout(30):
                while not http_server.started:
                    if http_task.done():
                        await http_task
                        raise RuntimeError("HTTP listener exited before starting")
                    await asyncio.sleep(0.01)
            client_options = {
                "base_url": f"http://127.0.0.1:{listener.getsockname()[1]}",
                "timeout": 120,
            }
            report["transport"] = "loopback TCP, uvicorn, production routes/pool"
        async with httpx.AsyncClient(**client_options) as client:
            models = await client.get("/v1/models")
            models.raise_for_status()
            assert args.model in [item["id"] for item in models.json()["data"]]
            report["listed"] = True
            requests = [
                (
                    "/v1/chat/completions",
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": "What is the capital of Australia? Answer in one sentence.",
                            }
                        ]
                    },
                ),
                (
                    "/v1/chat/completions",
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": "What is the capital of Australia? Answer in one sentence.",
                            }
                        ],
                        "stream": True,
                        "stream_options": {"include_usage": True},
                    },
                ),
                (
                    "/v1/completions",
                    {"prompt": "The capital of Australia is", "max_tokens": 16},
                ),
                (
                    "/v1/chat/completions",
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": "Use get_weather to check the weather in Paris.",
                            }
                        ],
                        "tools": tools,
                    },
                ),
                (
                    "/v1/chat/completions",
                    {
                        "messages": [{"role": "user", "content": "Hello"}],
                        "min_p": 0.1,
                        "stream": True,
                    },
                ),
                (
                    "/v1/chat/completions",
                    {
                        "messages": [{"role": "user", "content": "Hello"}],
                        "response_format": {"type": "json_object"},
                    },
                ),
            ]
            requests.extend(
                [
                    (
                        "/v1/completions",
                        {"prompt": "Hello", "min_p": 0.1, "stream": True},
                    ),
                    (
                        "/v1/chat/completions",
                        {
                            "model": "IFM--K2-Horizon-0.9B",
                            "messages": [
                                {
                                    "role": "user",
                                    "content": "What is the capital of Australia? Answer in one sentence.",
                                }
                            ],
                        },
                    ),
                    (
                        "/v1/chat/completions",
                        {
                            "messages": [
                                {
                                    "role": "user",
                                    "content": "What is the capital of Australia? Answer in one sentence.",
                                }
                            ],
                            "stop": ["Canberra"],
                        },
                    ),
                ]
            )
            for route, request in requests:
                body = {
                    "model": args.model,
                    "temperature": 0,
                    "seed": 42,
                    "max_tokens": 192,
                    **request,
                }
                if "messages" in body:
                    body["reasoning_effort"] = "low"
                response = await client.post(route, json=body)
                row = {"route": route, "request": body, "status": response.status_code}
                expected = 400 if "min_p" in body or "response_format" in body else 200
                row["response"] = (
                    response.text
                    if body.get("stream") and expected == 200
                    else response.json()
                )
                report["requests"].append(row)
                print(json.dumps(row), flush=True)
                assert response.status_code == expected
            call_response = report["requests"][3]["response"]
            calls = call_response["choices"][0]["message"].get("tool_calls")
            assert calls and calls[0]["function"]["name"] == "get_weather"
            assert json.loads(calls[0]["function"]["arguments"])["city"] == "Paris"
            followup = {
                "model": args.model,
                "temperature": 0,
                "max_tokens": 192,
                "reasoning_effort": "low",
                "tools": tools,
                "messages": [
                    {
                        "role": "user",
                        "content": "Use get_weather to check the weather in Paris.",
                    },
                    {"role": "assistant", "tool_calls": calls},
                    {
                        "role": "tool",
                        "tool_call_id": calls[0]["id"],
                        "content": '{"temperature_c":18,"condition":"sunny"}',
                    },
                ],
            }
            response = await client.post("/v1/chat/completions", json=followup)
            report["tool_followup"] = {
                "status": response.status_code,
                "response": response.json(),
            }
            response.raise_for_status()
            assert "18" in response.json()["choices"][0]["message"]["content"]
            stopped = report["requests"][-1]["response"]["choices"][0]
            assert stopped["finish_reason"] == "stop"
            assert "Canberra" not in stopped["message"].get("content", "")
            # Independently decoded requests must not share RNG or adapter state.
            stochastic = {
                "model": args.model,
                "temperature": 0.7,
                "top_p": 0.9,
                "seed": 123,
                "max_tokens": 64,
                "reasoning_effort": "low",
                "messages": [{"role": "user", "content": "Give me a short greeting."}],
            }
            samples = [
                (await client.post("/v1/chat/completions", json=stochastic)).json()
                for _ in range(2)
            ]
            assert samples[0]["choices"] == samples[1]["choices"]
            report["stochastic_reproducible"] = samples
            # Exercise pool unload/reload, not just engine.stop directly.
            await pool._unload_engine(args.model)
            reloaded = await client.post(
                "/v1/chat/completions", json=report["requests"][0]["request"]
            )
            reloaded.raise_for_status()
            assert (
                reloaded.json()["choices"]
                == report["requests"][0]["response"]["choices"]
            )
            report["reload_matches"] = True
            if args.tcp:
                engine = pool.get_entry(args.model).engine
                cancellation_request = {
                    "model": args.model,
                    "prompt": "Write a detailed tutorial with many Python examples:",
                    "temperature": 0,
                    "max_tokens": 2048,
                    "stream": True,
                }
                observed = []
                active_at_close = False
                async with client.stream(
                    "POST", "/v1/completions", json=cancellation_request
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        assert (
                            line != "data: [DONE]"
                        ), "Generation ended before disconnect"
                        chunk = json.loads(line[6:])
                        observed.append(chunk)
                        if any(c.get("text") for c in chunk.get("choices", [])):
                            active_at_close = bool(engine._events)
                            break
                assert active_at_close, "Disconnect must interrupt an active producer"
                async with asyncio.timeout(10):
                    while engine._events:
                        await asyncio.sleep(0.01)
                after_cancel = await client.post(
                    "/v1/chat/completions", json=report["requests"][0]["request"]
                )
                after_cancel.raise_for_status()
                assert after_cancel.json()["choices"] == reloaded.json()["choices"]
                report["disconnect"] = {
                    "request": cancellation_request,
                    "received_chunks": observed,
                    "producer_active_at_close": active_at_close,
                    "producer_drained": not engine._events,
                    "next_request_matches": True,
                }
            report["engine_stats"] = pool.get_entry(args.model).engine.get_stats()
            report["passed"] = True
    finally:
        if http_server is not None:
            http_server.should_exit = True
        if http_task is not None:
            await http_task
        if listener is not None:
            listener.close()
        await pool.shutdown()
        server._server_state = previous
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--model", default="IFM--K2-Horizon-0.9B-Uno")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--tcp", action="store_true", help="Also prove real disconnects"
    )
    asyncio.run(probe(parser.parse_args()))


if __name__ == "__main__":
    main()

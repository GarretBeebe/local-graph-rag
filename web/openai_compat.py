"""OpenAI-compatible response and SSE formatting."""

import json
import time
import uuid
from typing import Any


def make_stream_chunk(
    request_id: str,
    created: int,
    model: str,
    *,
    content: str | None = None,
    finish_reason: str | None = None,
) -> str:
    delta: dict[str, str] = {"content": content} if content is not None else {}
    chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(chunk)}\n\n"


def build_chat_response(answer: str, model: str) -> dict[str, Any]:
    """Build an OpenAI-compatible chat completion response object."""
    answer = answer or "No response generated."
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def model_entry(model_id: str, server_start: int) -> dict[str, Any]:
    return {"id": model_id, "object": "model", "created": server_start, "owned_by": "ollama"}

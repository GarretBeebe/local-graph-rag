"""Request/response Pydantic models and chat request validation."""

from typing import Literal

from fastapi import HTTPException
from pydantic import BaseModel, Field

from local_graph_rag.settings import (
    MAX_CHAT_CONTENT_ITEMS,
    MAX_CHAT_MESSAGE_CHARS,
    MAX_CHAT_MESSAGES,
    MAX_CHAT_QUESTION_CHARS,
    MAX_CHAT_TOTAL_CHARS,
    MAX_MODEL_NAME_CHARS,
)

GraphMode = Literal["auto", "local", "global"]


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=128)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system", "tool"]
    content: str | list[dict[str, str]]


class ChatRequest(BaseModel):
    model: str = Field(min_length=1, max_length=MAX_MODEL_NAME_CHARS)
    messages: list[ChatMessage] = Field(min_length=1, max_length=MAX_CHAT_MESSAGES)
    stream: bool | None = False
    graph_mode: GraphMode = "auto"


def _message_size(content: str | list[dict[str, str]]) -> int:
    if isinstance(content, str):
        return len(content)
    return sum(len(str(v)) for item in content if isinstance(item, dict) for v in item.values())


def validate_chat_request(req: ChatRequest) -> None:
    total_chars = 0
    for message in req.messages:
        size = _message_size(message.content)
        if size > MAX_CHAT_MESSAGE_CHARS:
            raise HTTPException(status_code=400, detail="Chat message exceeds size limit")
        total_chars += size
        if total_chars > MAX_CHAT_TOTAL_CHARS:
            raise HTTPException(status_code=400, detail="Chat request exceeds total size limit")
        if isinstance(message.content, list) and len(message.content) > MAX_CHAT_CONTENT_ITEMS:
            raise HTTPException(status_code=400, detail="Structured message has too many items")


def extract_question_from_messages(messages: list[ChatMessage]) -> str:
    """Extract the user question from the latest user message, handling OpenAI formats."""
    for message in reversed(messages):
        if message.role == "user":
            content = message.content
            break
    else:
        raise HTTPException(status_code=400, detail="Chat request must include a user message")

    if isinstance(content, str):
        question = content
    else:
        question = " ".join(item.get("text", "") for item in content if isinstance(item, dict))

    question = question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Last user message content is empty")
    if len(question) > MAX_CHAT_QUESTION_CHARS:
        raise HTTPException(status_code=400, detail="Question exceeds size limit")
    return question

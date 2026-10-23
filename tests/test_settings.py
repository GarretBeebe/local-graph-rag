"""Tests for settings validation."""

import importlib

import pytest

import local_graph_rag.settings as settings


@pytest.fixture(autouse=True)
def reload_settings_after_env_patch():
    yield
    importlib.reload(settings)


def test_short_api_key_is_rejected(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("API_KEY", "too-short")
    with pytest.raises(ValueError, match="API_KEY must be at least 32 characters"):
        importlib.reload(settings)


def test_32_character_api_key_is_accepted(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("API_KEY", "a" * 32)
    reloaded = importlib.reload(settings)
    assert reloaded.API_KEY == "a" * 32


def test_insecure_localonly_with_wildcard_cors_is_rejected(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ALLOW_INSECURE_LOCALONLY", "true")
    monkeypatch.setenv("CORS_ORIGINS", "*")
    with pytest.raises(
        ValueError,
        match=r"CORS_ORIGINS=\* is not allowed with ALLOW_INSECURE_LOCALONLY",
    ):
        importlib.reload(settings)


def test_insecure_localonly_with_specific_cors_origin_is_accepted(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ALLOW_INSECURE_LOCALONLY", "true")
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:3000")
    reloaded = importlib.reload(settings)
    assert reloaded.CORS_ORIGINS == ["http://localhost:3000"]


def test_executor_workers_below_generation_concurrency_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("GENERATION_CONCURRENCY_LIMIT", "2")
    monkeypatch.setenv("RAG_EXECUTOR_WORKERS", "1")
    with pytest.raises(ValueError, match="RAG_EXECUTOR_WORKERS must be >= GENERATION_CONCURRENCY"):
        importlib.reload(settings)


def test_executor_workers_equal_to_generation_concurrency_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("GENERATION_CONCURRENCY_LIMIT", "2")
    monkeypatch.setenv("RAG_EXECUTOR_WORKERS", "2")
    reloaded = importlib.reload(settings)
    assert reloaded.RAG_EXECUTOR_WORKERS == 2


def test_chat_models_default_to_gen_model(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GEN_MODEL", "big")
    monkeypatch.delenv("CHAT_MODELS", raising=False)
    reloaded = importlib.reload(settings)
    assert reloaded.CHAT_MODELS == ["big"]


def test_chat_models_list_gen_model_first_without_duplicates(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GEN_MODEL", "big")
    monkeypatch.setenv("CHAT_MODELS", " small , big,,small ")
    reloaded = importlib.reload(settings)
    assert reloaded.CHAT_MODELS == ["big", "small"]

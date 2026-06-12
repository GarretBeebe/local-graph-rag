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
    with pytest.raises(ValueError, match=r"CORS_ORIGINS=\* is not allowed with ALLOW_INSECURE_LOCALONLY"):
        importlib.reload(settings)


def test_insecure_localonly_with_specific_cors_origin_is_accepted(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ALLOW_INSECURE_LOCALONLY", "true")
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:3000")
    reloaded = importlib.reload(settings)
    assert reloaded.CORS_ORIGINS == ["http://localhost:3000"]

"""Chat model factory — Groq or OpenAI, chosen once per session."""

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

PROVIDERS = {
    "groq": {
        "label": "Groq (GPT-OSS 120B)",
        "model": "groq:openai/gpt-oss-120b",
        "env_key": "GROQ_API_KEY",
    },
    "openai": {
        "label": "OpenAI (GPT-4o mini)",
        "model": "openai:gpt-4o-mini",
        "env_key": "OPENAI_API_KEY",
    },
}

_models: dict[str, BaseChatModel] = {}


def get_model(provider: str) -> BaseChatModel:
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}")

    if provider not in _models:
        _models[provider] = init_chat_model(PROVIDERS[provider]["model"])

    return _models[provider]

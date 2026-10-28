"""OpenRouter chat helper — ključ iz openrouter.env, zadani model, opcionalni override."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Union

from openrouter import OpenRouter, components

_ROOT = Path(__file__).resolve().parent
_ENV_FILE = _ROOT / "openrouter.env"

# OpenRouter model id (npr. openai/gpt-4o-mini); override u openrouter.env ili parametrom model=
DEFAULT_MODEL = "openai/gpt-4o-mini"
# Decision modeli (Mammoth/Elephant) koriste alpha.decisions, ne chat/completions
DEFAULT_DECISIONS_MODEL = "baseten/mammoth-1:free"

Questions = Union[
    Mapping[str, components.Questions],
    Mapping[str, components.QuestionsTypedDict],
]
State = Union[str, components.State, components.StateTypedDict, Mapping[str, Any]]

Message = Union[
    components.ChatMessages,
    components.ChatMessagesTypedDict,
    Mapping[str, str],
]


def _parse_env_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if "=" not in line:
        return None
    key, _, value = line.partition("=")
    key, value = key.strip(), value.strip().strip('"').strip("'")
    if not key:
        return None
    return key, value


def load_openrouter_env(path: Path | None = None) -> dict[str, str]:
    """Učitaj openrouter.env (ne prepisuje postojeće os.environ varijable)."""
    path = path or _ENV_FILE
    loaded: dict[str, str] = {}
    if not path.is_file():
        return loaded
    for line in path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_env_line(line)
        if not parsed:
            continue
        key, value = parsed
        loaded[key] = value
        if key not in os.environ:
            os.environ[key] = value
    return loaded


def get_openrouter_api_key() -> str:
    load_openrouter_env()
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            f"OPENROUTER_API_KEY nije postavljen. Dodaj ključ u {_ENV_FILE.name}."
        )
    return key


def get_default_model() -> str:
    load_openrouter_env()
    return (
        os.getenv("OPENROUTER_DEFAULT_MODEL", "").strip()
        or DEFAULT_MODEL
    )


def get_default_decisions_model() -> str:
    load_openrouter_env()
    return (
        os.getenv("OPENROUTER_DECISIONS_MODEL", "").strip()
        or DEFAULT_DECISIONS_MODEL
    )


def _normalize_messages(
    messages: Iterable[Message],
) -> list[components.ChatMessagesTypedDict]:
    out: list[components.ChatMessagesTypedDict] = []
    for m in messages:
        if isinstance(m, Mapping):
            out.append({"role": str(m["role"]), "content": str(m["content"])})
        else:
            out.append(m)  # type: ignore[arg-type]
    return out


def chat_send(
    messages: Iterable[Message],
    *,
    model: str | None = None,
    stream: bool = False,
    **kwargs: Any,
) -> components.ChatResult:
    """
    Pošalji chat completion (stream=False).

    :param model: OpenRouter model id; ako nije zadan, koristi get_default_model().
    :param kwargs: proslijedi u OpenRouter.chat.send (temperature, max_tokens, …).
    """
    if stream:
        raise ValueError("Za stream=True koristi chat_send_stream().")

    model = (model or get_default_model()).strip()
    msgs = _normalize_messages(messages)
    api_key = get_openrouter_api_key()

    with OpenRouter(api_key=api_key) as client:
        return client.chat.send(
            messages=msgs,
            model=model,
            stream=False,
            **kwargs,
        )


def chat_send_stream(
    messages: Iterable[Message],
    *,
    model: str | None = None,
    **kwargs: Any,
):
    """Stream chat completion — vraća iterator chunkova iz SDK-a."""
    model = (model or get_default_model()).strip()
    msgs = _normalize_messages(messages)
    api_key = get_openrouter_api_key()

    with OpenRouter(api_key=api_key) as client:
        return client.chat.send(
            messages=msgs,
            model=model,
            stream=True,
            **kwargs,
        )


def chat_text(
    prompt: str,
    *,
    system: str | None = None,
    model: str | None = None,
    **kwargs: Any,
) -> str:
    """Jednostavan user prompt → tekst odgovora asistenta."""
    messages: list[components.ChatMessagesTypedDict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    result = chat_send(messages, model=model, **kwargs)
    return message_content(result)


def message_content(result: components.ChatResult) -> str:
    if not result.choices:
        return ""
    msg = result.choices[0].message
    if msg is None:
        return ""
    content = msg.content
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    # multimodal: spoji tekstualne dijelove
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, Mapping) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
        elif hasattr(block, "text"):
            parts.append(str(getattr(block, "text", "")))
    return "".join(parts)


def decisions_create(
    state: State,
    questions: Questions,
    *,
    model: str | None = None,
    **kwargs: Any,
) -> components.DecisionsResponse:
    """
    Decisions API (npr. baseten/mammoth-1:free) — vraća vjerojatnosti, ne chat tekst.
    """
    model = (model or get_default_decisions_model()).strip()
    api_key = get_openrouter_api_key()
    with OpenRouter(api_key=api_key) as client:
        return client.alpha.decisions.create(
            model=model,
            state=state,
            questions=questions,
            **kwargs,
        )


def _main() -> None:
    import json
    import sys

    model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DECISIONS_MODEL
    print(f"Testing decisions model: {model}")

    res = decisions_create(
        state="What is the capital of France?",
        questions={
            "capital": {
                "type": "choice",
                "instructions": "Which city is the capital of France?",
                "criteria": {
                    "paris": "Paris, France",
                    "london": "London, United Kingdom",
                    "berlin": "Berlin, Germany",
                },
            },
        },
        model=model,
    )
    print(json.dumps(res.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    _main()

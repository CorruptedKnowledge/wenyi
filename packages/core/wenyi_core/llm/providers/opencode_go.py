"""OpenCode Go OpenAI-compatible endpoint with required identity headers.

OpenCode Go expects clients to identify themselves and send a stable per-connection
``x-opencode-session`` header. See https://opencode.ai/docs/go/ and the turygo/wenyi
compatibility notes.
"""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..transport import Messages, ResolvedModel
from ._openai_compatible import (
    OpenAICompatibleBaseClient,
    base_request_kwargs,
    deep_merge,
)

DEFAULT_BASE_URL = "https://opencode.ai/zen/go/v1"
DEFAULT_API_KEY_ENV = "OPENCODE_API_KEY"
USER_AGENT = "wenyi"


class OpenCodeGoOptions(BaseModel):
    """OpenCode Go model options (OpenAI-compatible body with optional extras)."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    # None leaves reasoning to the gateway; False disables it where the model allows.
    thinking: bool | None = None
    reasoning_effort: str = "high"
    extra_body: dict[str, Any] = Field(default_factory=dict)


def _is_deepseek(model: str) -> bool:
    """Match DeepSeek models, including vendor-prefixed gateway names."""
    return model.rsplit("/", 1)[-1].lower().startswith("deepseek")


def build_request_kwargs(
    model_config: ResolvedModel[OpenCodeGoOptions],
    messages: Messages,
    *,
    json_mode: bool = False,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """Build Chat Completions arguments for the OpenCode Go gateway."""
    kwargs = base_request_kwargs(model_config.model, messages, json_mode=json_mode)
    extra_body = dict(model_config.options.extra_body)
    thinking = model_config.options.thinking
    if thinking:
        extra_body.setdefault("reasoning_effort", model_config.options.reasoning_effort)
    elif (
        thinking is False
        and _is_deepseek(model_config.model)
        and "reasoning_effort" not in extra_body
    ):
        # The gateway enables DeepSeek reasoning when a request omits a control, so an
        # explicit False must disable it. Thinking-only models such as GLM reject the flag.
        extra_body.setdefault("thinking", {"type": "disabled"})
    if extra_body:
        kwargs["extra_body"] = deep_merge({}, extra_body)
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return kwargs


class OpenCodeGoClient(OpenAICompatibleBaseClient[OpenCodeGoOptions]):
    """OpenCode Go transport with Wenyi User-Agent and a stable session id."""

    default_base_url = DEFAULT_BASE_URL
    default_api_key_env = DEFAULT_API_KEY_ENV
    requires_api_key = True

    def __init__(self, cfg, *, credentials: tuple[str | None] | None = None):
        super().__init__(cfg, credentials=credentials)
        # One session id per connection/client lifetime (stable across retries in a run).
        self._session_id = str(uuid.uuid4())

    def _ensure_client(self) -> Any:
        """Create the SDK client with OpenCode Go identity headers only for this provider."""
        with self._client_lock:
            if self._client is None:
                from openai import OpenAI

                self.validate_credentials()
                api_key = self.api_key()
                self._client = OpenAI(
                    api_key=api_key or "no-key",
                    base_url=self.base_url,
                    timeout=self.cfg.timeout,
                    max_retries=0,
                    default_headers={
                        "User-Agent": USER_AGENT,
                        "x-opencode-session": self._session_id,
                    },
                )
        return self._client

    def _build_request_kwargs(
        self,
        model_config: ResolvedModel[OpenCodeGoOptions],
        messages: Messages,
        *,
        json_mode: bool,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        return build_request_kwargs(
            model_config,
            messages,
            json_mode=json_mode,
            max_tokens=max_tokens,
        )

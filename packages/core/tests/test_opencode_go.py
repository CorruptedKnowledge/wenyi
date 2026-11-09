"""OpenCode Go provider identity and session header tests."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from wenyi_core.config import Config
from wenyi_core.llm.configuration import ProviderConfig
from wenyi_core.llm.providers.opencode_go import (
    USER_AGENT,
    OpenCodeGoClient,
    OpenCodeGoOptions,
    build_request_kwargs,
)
from wenyi_core.llm.registry import provider_spec
from wenyi_core.llm.transport import ResolvedModel


class TestOpenCodeGoProvider(unittest.TestCase):
    def test_provider_is_registered(self):
        self.assertEqual(provider_spec("opencode-go").kind, "opencode-go")
        self.assertEqual(provider_spec("opencode-go").client_class, "OpenCodeGoClient")

    def test_opencode_go_has_no_builtin_preset(self):
        with self.assertRaisesRegex(ValueError, "has no preset"):
            provider_spec("opencode-go").preset()

    def test_config_accepts_opencode_go_connection(self):
        cfg = Config.from_dict(
            {
                "llm": {
                    "providers": {"go": {"kind": "opencode-go"}},
                    "models": {
                        "writer": {
                            "provider": "go",
                            "model": "deepseek-v4-flash",
                        }
                    },
                    "tiers": {"strong": "writer", "cheap": "writer", "fast": "writer"},
                }
            }
        )
        self.assertEqual(cfg.llm.providers["go"].kind, "opencode-go")

    def test_ensure_client_passes_opencode_go_headers_only(self):
        client = OpenCodeGoClient(ProviderConfig(kind="opencode-go"))
        with (
            patch.dict("os.environ", {"OPENCODE_API_KEY": "test-key"}),
            patch("openai.OpenAI") as openai_ctor,
        ):
            client._ensure_client()
            client._ensure_client()  # reuse the same session id

        self.assertEqual(openai_ctor.call_count, 1)
        kwargs = openai_ctor.call_args.kwargs
        self.assertEqual(kwargs["base_url"], "https://opencode.ai/zen/go/v1")
        self.assertEqual(kwargs["api_key"], "test-key")
        headers = kwargs["default_headers"]
        self.assertEqual(headers["User-Agent"], USER_AGENT)
        self.assertEqual(headers["User-Agent"], "wenyi")
        self.assertRegex(
            headers["x-opencode-session"],
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
        )
        self.assertEqual(headers["x-opencode-session"], client._session_id)


class TestOpenCodeGoThinking(unittest.TestCase):
    messages = [{"role": "user", "content": "hi"}]

    def kwargs(self, model: str, **options):
        return build_request_kwargs(
            ResolvedModel(model=model, options=OpenCodeGoOptions(**options)), self.messages
        )

    def test_unset_thinking_keeps_the_gateway_default(self):
        for model in ("deepseek-v4.1-flash", "glm-5.3"):
            with self.subTest(model=model):
                self.assertNotIn("extra_body", self.kwargs(model))

    def test_explicit_false_disables_deepseek_reasoning(self):
        # Without a control the gateway reasons for DeepSeek, so false must say so explicitly.
        for model in ("deepseek-v4.1-flash", "deepseek-v4-pro", "DeepSeek-V4.1-Flash"):
            with self.subTest(model=model):
                kwargs = self.kwargs(model, thinking=False)
                self.assertEqual(kwargs["extra_body"], {"thinking": {"type": "disabled"}})

    def test_explicit_false_leaves_other_models_alone(self):
        # Thinking-only models such as GLM reject a disable flag.
        for model in ("glm-5.3", "minimax-m3", "kimi-k3"):
            with self.subTest(model=model):
                self.assertNotIn("extra_body", self.kwargs(model, thinking=False))

    def test_true_requests_the_configured_effort(self):
        kwargs = self.kwargs("deepseek-v4.1-flash", thinking=True, reasoning_effort="low")
        self.assertEqual(kwargs["extra_body"], {"reasoning_effort": "low"})

    def test_extra_body_overrides_win(self):
        for extra in ({"thinking": {"type": "enabled"}}, {"reasoning_effort": "low"}):
            with self.subTest(extra=extra):
                kwargs = self.kwargs("deepseek-v4.1-flash", thinking=False, extra_body=extra)
                self.assertEqual(kwargs["extra_body"], extra)


if __name__ == "__main__":
    unittest.main()

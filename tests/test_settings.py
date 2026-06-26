import unittest

from config import settings


class SettingsTests(unittest.TestCase):
    def test_resolve_max_output_tokens_is_capped_to_safe_range(self) -> None:
        original = settings.LLM_MAX_OUTPUT_TOKENS
        try:
            settings.LLM_MAX_OUTPUT_TOKENS = 65535
            self.assertEqual(settings._resolve_max_output_tokens(), 8192)

            settings.LLM_MAX_OUTPUT_TOKENS = 128
            self.assertEqual(settings._resolve_max_output_tokens(), 256)
        finally:
            settings.LLM_MAX_OUTPUT_TOKENS = original

    def test_build_openrouter_llm_kwargs_sets_max_tokens_for_openai_models(self) -> None:
        kwargs = settings._build_openrouter_llm_kwargs(
            model_name="openai/gpt-4.1-nano",
            temperature=0,
            streaming=False,
        )

        self.assertEqual(kwargs["max_tokens"], 4096)
        self.assertNotIn("model_kwargs", kwargs)

    def test_build_openrouter_llm_kwargs_uses_same_supported_key_for_gemini(self) -> None:
        kwargs = settings._build_openrouter_llm_kwargs(
            model_name="google/gemini-2.5-flash",
            temperature=0,
            streaming=True,
        )

        self.assertEqual(kwargs["max_tokens"], 4096)
        self.assertNotIn("model_kwargs", kwargs)


if __name__ == "__main__":
    unittest.main()

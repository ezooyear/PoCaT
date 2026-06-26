import importlib
import os
import unittest
from unittest.mock import patch

import config.settings as settings


class LLMSettingsTest(unittest.TestCase):
    def test_process_env_model_has_priority(self):
        with patch.dict(os.environ, {"LLM_MODEL": "openai/gpt-4.1-mini", "OPENROUTER_API_KEY": "test-key"}, clear=False):
            importlib.reload(settings)
            resolved = settings.get_resolved_llm_model()
            llm_settings = settings.get_llm_settings()

        self.assertEqual(resolved, "openai/gpt-4.1-mini")
        self.assertEqual(llm_settings["resolved_model"], "openai/gpt-4.1-mini")
        self.assertEqual(llm_settings["env_source_priority"], "process_env_over_dotenv")

    def test_resolved_model_updates_without_stale_global_cache(self):
        with patch.dict(os.environ, {"LLM_MODEL": "openai/first-model", "OPENROUTER_API_KEY": "test-key"}, clear=False):
            importlib.reload(settings)
            first = settings.get_resolved_llm_model()

        with patch.dict(os.environ, {"LLM_MODEL": "openai/second-model", "OPENROUTER_API_KEY": "test-key"}, clear=False):
            second = settings.get_resolved_llm_model()

        self.assertEqual(first, "openai/first-model")
        self.assertEqual(second, "openai/second-model")


if __name__ == "__main__":
    unittest.main()

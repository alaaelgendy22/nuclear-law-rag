import unittest

from prompter import build_grounded_prompt


class PromptTests(unittest.TestCase):
    def test_prompt_requires_sources_and_refuses_invention(self):
        prompt = build_grounded_prompt("What is strict liability?", "[Source 1] Example")
        self.assertIn("[Source N]", prompt)
        self.assertIn("Do not invent", prompt)
        self.assertIn("What is strict liability?", prompt)
        self.assertIn("[Source 1] Example", prompt)


if __name__ == "__main__":
    unittest.main()

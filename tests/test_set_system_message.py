import unittest
from unittest.mock import patch

import definers


class TestSetSystemMessage(unittest.TestCase):

    def setUp(self):
        self.original_system_message = definers.SYSTEM_MESSAGE

    def tearDown(self):
        definers.SYSTEM_MESSAGE = self.original_system_message

    def test_default_message(self):
        definers.set_system_message()
        self.assertIn("You are a helpful AI assistant.", definers.SYSTEM_MESSAGE)

    def test_with_role_and_name(self):
        definers.set_system_message(role="a code assistant", name="Definer")
        self.assertIn("You are a code assistant.", definers.SYSTEM_MESSAGE)
        self.assertIn("Your name is Definer.", definers.SYSTEM_MESSAGE)

    def test_with_style_instructions(self):
        definers.set_system_message(tone="friendly", chattiness="concise", interaction_style="ask questions")
        self.assertIn("Your tone should be friendly.", definers.SYSTEM_MESSAGE)
        self.assertIn("In terms of verbosity, concise.", definers.SYSTEM_MESSAGE)
        self.assertIn("When interacting, ask questions.", definers.SYSTEM_MESSAGE)

    def test_with_persona_data(self):
        persona = {"creator": "John Doe", "version": "1.0"}
        definers.set_system_message(persona_data=persona)
        self.assertIn("creator is John Doe", definers.SYSTEM_MESSAGE)
        self.assertIn("version is 1.0", definers.SYSTEM_MESSAGE)

    def test_with_goals(self):
        goals = ["answer questions", "be helpful"]
        definers.set_system_message(goals=goals)
        self.assertIn("answer questions; be helpful.", definers.SYSTEM_MESSAGE)

    def test_with_task_rules_and_output_format(self):
        rules = ["Do not mention you are an AI."]
        output_format = "JSON"
        definers.set_system_message(task_rules=rules, output_format=output_format)
        self.assertIn("You must strictly follow these rules:", definers.SYSTEM_MESSAGE)
        self.assertIn("1. Do not mention you are an AI.", definers.SYSTEM_MESSAGE)
        self.assertIn("2. Your final output must be exclusively in the following format: JSON.", definers.SYSTEM_MESSAGE)

    @patch('definers.log')
    def test_log_is_called(self, mock_log):
        definers.set_system_message(name="Tester")
        mock_log.assert_called_once_with("System Message Updated", definers.SYSTEM_MESSAGE)

    def test_all_parameters(self):
        definers.set_system_message(
            name="ChatBot",
            role="a friendly guide",
            tone="encouraging",
            goals=["guide the user"],
            chattiness="detailed",
            persona_data={"language": "Python"},
            task_rules=["Always be positive"],
            interaction_style="offer examples",
            output_format="Markdown"
        )
        self.assertIn("You are a friendly guide.", definers.SYSTEM_MESSAGE)
        self.assertIn("Your name is ChatBot.", definers.SYSTEM_MESSAGE)
        self.assertIn("Your tone should be encouraging.", definers.SYSTEM_MESSAGE)
        self.assertIn("guide the user.", definers.SYSTEM_MESSAGE)
        self.assertIn("In terms of verbosity, detailed.", definers.SYSTEM_MESSAGE)
        self.assertIn("language is Python", definers.SYSTEM_MESSAGE)
        self.assertIn("1. Always be positive", definers.SYSTEM_MESSAGE)
        self.assertIn("When interacting, offer examples.", definers.SYSTEM_MESSAGE)
        self.assertIn("format: Markdown", definers.SYSTEM_MESSAGE)


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import patch
from definers import set_system_message

class TestSetSystemMessage(unittest.TestCase):

    def setUp(self):
        self.original_system_message = definers._system_message

    def tearDown(self):
        definers._system_message = self.original_system_message

    def test_default_message(self):
        expected_message = "You are a helpful AI assistant."
        result = set_system_message()
        self.assertEqual(result, expected_message)

    def test_with_role_and_name(self):
        result = set_system_message(role="a code assistant", name="Definer")
        self.assertIn("You are a code assistant.", result)
        self.assertIn("Your name is Definer.", result)

    def test_with_style_instructions(self):
        result = set_system_message(tone="friendly", chattiness="be concise", interaction_style="ask clarifying questions")
        self.assertIn("Your tone should be friendly.", result)
        self.assertIn("In terms of verbosity, be concise.", result)
        self.assertIn("When interacting, ask clarifying questions.", result)

    def test_with_persona_data(self):
        persona = {"creator": "John Doe", "purpose": "to assist"}
        result = set_system_message(persona_data=persona)
        self.assertIn("Here is some information for you to learn and remember: creator is John Doe; purpose is to assist.", result)

    def test_with_goals(self):
        goals = ["answer user questions", "be accurate"]
        result = set_system_message(goals=goals)
        self.assertIn("answer user questions; be accurate.", result)

    def test_with_task_rules_and_output_format(self):
        rules = ["Do not use emojis.", "Provide code in Python."]
        output_format = "JSON"
        result = set_system_message(task_rules=rules, output_format=output_format)
        self.assertIn("You must strictly follow these rules:", result)
        self.assertIn("1. Do not use emojis.", result)
        self.assertIn("2. Provide code in Python.", result)
        self.assertIn("3. Your final output must be exclusively in the following format: JSON.", result)

    def test_all_parameters(self):
        result = set_system_message(
            name="Tester",
            role="a test bot",
            tone="formal",
            goals=["validate outputs"],
            chattiness="to the point",
            persona_data={"version": "1.0"},
            task_rules=["Only respond with pass or fail."],
            interaction_style="do not ask questions",
            output_format="a single word"
        )
        self.assertIn("You are a test bot.", result)
        self.assertIn("Your name is Tester.", result)
        self.assertIn("Your tone should be formal.", result)
        self.assertIn("validate outputs.", result)
        self.assertIn("to the point", result)
        self.assertIn("version is 1.0", result)
        self.assertIn("Only respond with pass or fail.", result)
        self.assertIn("do not ask questions", result)
        self.assertIn("a single word", result)

    @patch('definers.log')
    def test_log_is_called(self, mock_log):
        set_system_message(role="test")
        mock_log.assert_called_once()
        args, _ = mock_log.call_args
        self.assertEqual(args[0], "System Message Updated")
        self.assertEqual(args[1], "You are a test bot.")

if __name__ == '__main__':
    # Add definers to path to access the global variable for tests
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import definers
    unittest.main()

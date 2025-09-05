import unittest
from unittest.mock import patch
from definers import set_system_message

class TestSetSystemMessage(unittest.TestCase):

    def setUp(self):
        from definers import _system_message
        self.original_system_message = _system_message

    def tearDown(self):
        import definers
        definers._system_message = self.original_system_message

    @patch('definers.log')
    def test_default_message(self, mock_log):
        import definers
        definers._system_message = ""  # Reset for this test
        returned_message = set_system_message()
        self.assertIn("You are a helpful AI assistant.", returned_message)
        self.assertEqual(returned_message, definers._system_message)
        mock_log.assert_called_once()

    @patch('definers.log')
    def test_with_role_and_name(self, mock_log):
        message = set_system_message(role="a code assistant", name="Definer")
        self.assertIn("You are a code assistant.", message)
        self.assertIn("Your name is Definer.", message)

    @patch('definers.log')
    def test_with_style_instructions(self, mock_log):
        message = set_system_message(tone="friendly", chattiness="concise", interaction_style="ask clarifying questions")
        self.assertIn("Your tone should be friendly.", message)
        self.assertIn("In terms of verbosity, concise.", message)
        self.assertIn("When interacting, ask clarifying questions.", message)

    @patch('definers.log')
    def test_with_persona_data(self, mock_log):
        message = set_system_message(persona_data={"creator": "John Doe"})
        self.assertIn("creator is John Doe", message)

    @patch('definers.log')
    def test_with_goals(self, mock_log):
        message = set_system_message(goals=["answer questions", "be helpful"])
        self.assertIn("answer questions; be helpful.", message)

    @patch('definers.log')
    def test_with_task_rules_and_output_format(self, mock_log):
        message = set_system_message(task_rules=["Do not mention you are an AI."], output_format="JSON")
        self.assertIn("You must strictly follow these rules:", message)
        self.assertIn("1. Do not mention you are an AI.", message)
        self.assertIn("2. Your final output must be exclusively in the following format: JSON.", message)

    @patch('definers.log')
    def test_all_parameters(self, mock_log):
        message = set_system_message(
            role="a travel guide",
            name="Wanderer",
            tone="enthusiastic",
            chattiness="detailed",
            persona_data={"specialty": "Europe"},
            goals=["provide travel tips"],
            task_rules=["Avoid tourist traps"],
            output_format="Markdown"
        )
        self.assertIn("You are a travel guide.", message)
        self.assertIn("Your name is Wanderer.", message)
        self.assertIn("Your tone should be enthusiastic.", message)
        self.assertIn("specialty is Europe", message)
        self.assertIn("provide travel tips.", message)
        self.assertIn("1. Avoid tourist traps", message)
        self.assertIn("2. Your final output must be exclusively in the following format: Markdown.", message)

    @patch('definers.log')
    def test_log_is_called(self, mock_log):
        set_system_message()
        mock_log.assert_called()

if __name__ == '__main__':
    unittest.main()


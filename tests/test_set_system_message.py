import sys
import unittest
from unittest.mock import patch

import definers.constants as constants
from definers.application_text.system_messages import set_system_message


class TestSetSystemMessage(unittest.TestCase):
    def setUp(self):
        self.definers_package = sys.modules["definers"]
        self.original_constants_system_message = constants.SYSTEM_MESSAGE
        self.original_package_system_message = getattr(
            self.definers_package,
            "SYSTEM_MESSAGE",
            None,
        )

    def call_set_system_message(self, **kwargs):
        with (
            patch.object(
                constants,
                "SYSTEM_MESSAGE",
                self.original_constants_system_message,
            ),
            patch.object(
                self.definers_package,
                "SYSTEM_MESSAGE",
                self.original_package_system_message,
                create=True,
            ),
        ):
            set_system_message(**kwargs)
            self.assertEqual(
                constants.SYSTEM_MESSAGE,
                self.definers_package.SYSTEM_MESSAGE,
            )
            return constants.SYSTEM_MESSAGE

    def test_default_message(self):
        message = self.call_set_system_message()
        self.assertIn("You are a helpful AI assistant.", message)

    def test_with_role_and_name(self):
        message = self.call_set_system_message(
            role="a code assistant",
            name="Definer",
        )
        self.assertIn("You are a code assistant.", message)
        self.assertIn("Your name is Definer.", message)

    def test_with_style_instructions(self):
        message = self.call_set_system_message(
            tone="friendly",
            chattiness="concise",
            interaction_style="ask questions",
        )
        self.assertIn("Your tone should be friendly.", message)
        self.assertIn("In terms of verbosity, concise.", message)
        self.assertIn("When interacting, ask questions.", message)

    def test_with_persona_data(self):
        persona = {"creator": "John Doe", "version": "1.0"}
        message = self.call_set_system_message(persona_data=persona)
        self.assertIn("creator is John Doe", message)
        self.assertIn("version is 1.0", message)

    def test_with_goals(self):
        goals = ["answer questions", "be helpful"]
        message = self.call_set_system_message(goals=goals)
        self.assertIn("answer questions; be helpful.", message)

    def test_with_task_rules_and_output_format(self):
        rules = ["Do not mention you are an AI."]
        output_format = "JSON"
        message = self.call_set_system_message(
            task_rules=rules, output_format=output_format
        )
        self.assertIn("You must strictly follow these rules:", message)
        self.assertIn("1. Do not mention you are an AI.", message)
        self.assertIn(
            "2. Your final output must be exclusively in the following format: JSON.",
            message,
        )

    def test_all_parameters(self):
        message = self.call_set_system_message(
            name="ChatBot",
            role="a friendly guide",
            tone="encouraging",
            goals=["guide the user"],
            chattiness="detailed",
            persona_data={"language": "Python"},
            task_rules=["Always be positive"],
            interaction_style="offer examples",
            output_format="Markdown",
        )
        self.assertIn("You are a friendly guide.", message)
        self.assertIn("Your name is ChatBot.", message)
        self.assertIn("Your tone should be encouraging.", message)
        self.assertIn("guide the user.", message)
        self.assertIn("In terms of verbosity, detailed.", message)
        self.assertIn("language is Python", message)
        self.assertIn("1. Always be positive", message)
        self.assertIn("When interacting, offer examples.", message)
        self.assertIn("format: Markdown", message)


if __name__ == "__main__":
    unittest.main()

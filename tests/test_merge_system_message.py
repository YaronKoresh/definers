import unittest

from definers import merge_system_message


class TestMergeSystemMessage(unittest.TestCase):

    def test_simple_string_values(self):
        data = {
            "message1": "Hello World",
            "message2": "This is a test.",
        }
        expected_output = "<|system|>\nHello World\n<|end|>\n<|system|>\nThis is a test.\n<|end|>\n"
        self.assertEqual(merge_system_message(data), expected_output)

    def test_list_of_dictionaries(self):
        data = {
            "users": [
                {"name": "Alice", "role": "admin"},
                {"name": "Bob", "role": "user"},
            ]
        }
        expected_output = (
            "<|system|>\nname: Alice\nrole: admin\n<|end|>\n"
            "<|system|>\nname: Bob\nrole: user\n<|end|>\n"
        )
        self.assertEqual(merge_system_message(data), expected_output)

    def test_list_with_nested_dictionaries(self):
        data = {
            "translations": [
                {"greetings": {"en": "Hello", "fr": "Bonjour"}}
            ]
        }
        expected_output = (
            "<|system|>\nen: Hello\nfr: Bonjour\n<|end|>\n"
        )
        self.assertEqual(merge_system_message(data), expected_output)

    def test_mixed_content(self):
        data = {
            "initial_prompt": "System initialized.",
            "rules": [
                {"rule_id": "001", "description": "Be helpful."},
                {"rule_id": "002", "description": "Be concise."},
            ],
            "final_prompt": "Awaiting user input.",
        }
        expected_output = (
            "<|system|>\nSystem initialized.\n<|end|>\n"
            "<|system|>\nrule_id: 001\ndescription: Be helpful.\n<|end|>\n"
            "<|system|>\nrule_id: 002\ndescription: Be concise.\n<|end|>\n"
            "<|system|>\nAwaiting user input.\n<|end|>\n"
        )
        self.assertEqual(merge_system_message(data), expected_output)

    def test_empty_dictionary(self):
        data = {}
        expected_output = ""
        self.assertEqual(merge_system_message(data), expected_output)

    def test_dictionary_with_other_types(self):
        data = {
            "message": "A string",
            "number": 123,
            "boolean": True,
            "none_value": None,
        }
        expected_output = "<|system|>\nA string\n<|end|>\n"
        self.assertEqual(merge_system_message(data), expected_output)


if __name__ == "__main__":
    unittest.main()

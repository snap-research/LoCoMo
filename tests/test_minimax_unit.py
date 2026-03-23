"""Unit tests for MiniMax provider integration in LoCoMo."""

import sys
import os
import json
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestRunMinimax(unittest.TestCase):
    """Test the run_minimax function in global_methods."""

    @patch('global_methods.httpx')
    def test_run_minimax_basic(self, mock_httpx):
        """Test basic MiniMax API call."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Paris"}}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_httpx.post.return_value = mock_response

        from global_methods import run_minimax
        os.environ['MINIMAX_API_KEY'] = 'test-key'
        result = run_minimax("What is the capital of France?", 32, "minimax-m2.5")

        self.assertEqual(result, "Paris")
        mock_httpx.post.assert_called_once()
        call_args = mock_httpx.post.call_args
        payload = call_args[1]['json']
        self.assertEqual(payload['model'], 'MiniMax-M2.5')
        self.assertEqual(payload['max_tokens'], 32)

    @patch('global_methods.httpx')
    def test_run_minimax_model_mapping(self, mock_httpx):
        """Test model name mapping for different MiniMax models."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_httpx.post.return_value = mock_response

        from global_methods import run_minimax
        os.environ['MINIMAX_API_KEY'] = 'test-key'

        test_cases = [
            ('minimax-m2.5', 'MiniMax-M2.5'),
            ('minimax-m2.5-highspeed', 'MiniMax-M2.5-highspeed'),
            ('minimax-m2.7', 'MiniMax-M2.7'),
        ]

        for input_name, expected_api_name in test_cases:
            mock_httpx.post.reset_mock()
            run_minimax("test", 32, input_name)
            payload = mock_httpx.post.call_args[1]['json']
            self.assertEqual(payload['model'], expected_api_name,
                             f"Model {input_name} should map to {expected_api_name}")

    @patch('global_methods.httpx')
    def test_run_minimax_temperature_clamping(self, mock_httpx):
        """Test that temperature is clamped to valid range."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_httpx.post.return_value = mock_response

        from global_methods import run_minimax
        os.environ['MINIMAX_API_KEY'] = 'test-key'

        # temperature=0 should be clamped to 0.01
        run_minimax("test", 32, "minimax-m2.5", temperature=0)
        payload = mock_httpx.post.call_args[1]['json']
        self.assertEqual(payload['temperature'], 0.01)

        # temperature=0.5 should pass through
        run_minimax("test", 32, "minimax-m2.5", temperature=0.5)
        payload = mock_httpx.post.call_args[1]['json']
        self.assertEqual(payload['temperature'], 0.5)

        # temperature > 1.0 should be clamped to 1.0
        run_minimax("test", 32, "minimax-m2.5", temperature=1.5)
        payload = mock_httpx.post.call_args[1]['json']
        self.assertEqual(payload['temperature'], 1.0)

    @patch('global_methods.httpx')
    def test_run_minimax_strips_think_tags(self, mock_httpx):
        """Test that thinking tags from M2.5 models are stripped."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "<think>Let me think about this...</think>Paris"}}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_httpx.post.return_value = mock_response

        from global_methods import run_minimax
        os.environ['MINIMAX_API_KEY'] = 'test-key'
        result = run_minimax("test", 32, "minimax-m2.5")
        self.assertEqual(result, "Paris")

    @patch('global_methods.httpx')
    def test_run_minimax_api_url(self, mock_httpx):
        """Test that the correct API URL is used."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_httpx.post.return_value = mock_response

        from global_methods import run_minimax
        os.environ['MINIMAX_API_KEY'] = 'test-key'
        run_minimax("test", 32, "minimax-m2.5")

        call_args = mock_httpx.post.call_args
        self.assertEqual(call_args[0][0], "https://api.minimax.io/v1/chat/completions")

    @patch('global_methods.httpx')
    def test_run_minimax_auth_header(self, mock_httpx):
        """Test that the authorization header includes the API key."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_httpx.post.return_value = mock_response

        from global_methods import run_minimax
        os.environ['MINIMAX_API_KEY'] = 'my-secret-key'
        run_minimax("test", 32, "minimax-m2.5")

        call_args = mock_httpx.post.call_args
        headers = call_args[1]['headers']
        self.assertEqual(headers['Authorization'], 'Bearer my-secret-key')


class TestSetMinimaxKey(unittest.TestCase):
    """Test the set_minimax_key function."""

    def test_set_minimax_key_no_error(self):
        """Test that set_minimax_key runs without error."""
        from global_methods import set_minimax_key
        set_minimax_key()  # Should not raise


class TestMinimaxUtils(unittest.TestCase):
    """Test the minimax_utils module."""

    def test_process_output_valid_json(self):
        """Test process_ouput with valid JSON."""
        from task_eval.minimax_utils import process_ouput

        result = process_ouput('{"0": "Paris", "1": "London"}')
        self.assertEqual(result, {"0": "Paris", "1": "London"})

    def test_process_output_with_prefix(self):
        """Test process_ouput strips text before JSON."""
        from task_eval.minimax_utils import process_ouput

        result = process_ouput('Here is the answer: {"0": "Paris"}')
        self.assertEqual(result, {"0": "Paris"})

    def test_get_cat_5_answer_single_char_a(self):
        """Test adversarial answer parsing: single character 'a'."""
        from task_eval.minimax_utils import get_cat_5_answer

        answer_key = {'a': 'Not mentioned', 'b': 'Paris'}
        result = get_cat_5_answer('a', answer_key)
        self.assertEqual(result, 'Not mentioned')

    def test_get_cat_5_answer_single_char_b(self):
        """Test adversarial answer parsing: single character 'b'."""
        from task_eval.minimax_utils import get_cat_5_answer

        answer_key = {'a': 'Not mentioned', 'b': 'Paris'}
        result = get_cat_5_answer('b', answer_key)
        self.assertEqual(result, 'Paris')

    def test_get_cat_5_answer_parenthesized(self):
        """Test adversarial answer parsing: parenthesized."""
        from task_eval.minimax_utils import get_cat_5_answer

        answer_key = {'a': 'Not mentioned', 'b': 'Paris'}
        result = get_cat_5_answer('(a)', answer_key)
        self.assertEqual(result, 'Not mentioned')

    def test_get_cat_5_answer_freeform(self):
        """Test adversarial answer parsing: free-form text returned as-is (lowercased)."""
        from task_eval.minimax_utils import get_cat_5_answer

        answer_key = {'a': 'Not mentioned', 'b': 'Paris'}
        # The function lowercases the input, then returns it when length > 3
        result = get_cat_5_answer('The answer is Paris', answer_key)
        self.assertEqual(result, 'the answer is paris')

    def test_get_input_context(self):
        """Test conversation context extraction."""
        from task_eval.minimax_utils import get_input_context

        data = {
            'session_1': [
                {'speaker': 'Alice', 'text': 'Hello Bob!', 'dia_id': 'd1'},
                {'speaker': 'Bob', 'text': 'Hi Alice!', 'dia_id': 'd2'},
            ],
            'session_1_date_time': '2024-01-01',
        }
        args = MagicMock()
        result = get_input_context(data, 100, None, args)
        self.assertIn('Alice', result)
        self.assertIn('Bob', result)
        self.assertIn('2024-01-01', result)

    def test_get_input_context_multimodal(self):
        """Test that blip captions are included in context."""
        from task_eval.minimax_utils import get_input_context

        data = {
            'session_1': [
                {'speaker': 'Alice', 'text': 'Look at this!', 'dia_id': 'd1',
                 'blip_caption': 'a photo of a sunset'},
            ],
            'session_1_date_time': '2024-01-01',
        }
        args = MagicMock()
        result = get_input_context(data, 100, None, args)
        self.assertIn('a photo of a sunset', result)

    def test_max_length_definitions(self):
        """Test that MAX_LENGTH contains correct MiniMax model entries."""
        from task_eval.minimax_utils import MAX_LENGTH

        self.assertIn('minimax-m2.5', MAX_LENGTH)
        self.assertIn('minimax-m2.5-highspeed', MAX_LENGTH)
        self.assertIn('minimax-m2.7', MAX_LENGTH)
        self.assertEqual(MAX_LENGTH['minimax-m2.5'], 204000)
        self.assertEqual(MAX_LENGTH['minimax-m2.7'], 1000000)

    def test_prompts_defined(self):
        """Test that required prompts are defined in minimax_utils."""
        from task_eval import minimax_utils

        self.assertTrue(hasattr(minimax_utils, 'QA_PROMPT'))
        self.assertTrue(hasattr(minimax_utils, 'QA_PROMPT_CAT_5'))
        self.assertTrue(hasattr(minimax_utils, 'QA_PROMPT_BATCH'))
        self.assertTrue(hasattr(minimax_utils, 'CONV_START_PROMPT'))


class TestEvaluateQADispatch(unittest.TestCase):
    """Test that evaluate_qa.py correctly dispatches to MiniMax."""

    def test_minimax_import(self):
        """Test that minimax_utils can be imported from evaluate_qa context."""
        from task_eval.minimax_utils import get_minimax_answers
        self.assertTrue(callable(get_minimax_answers))

    def test_set_minimax_key_import(self):
        """Test that set_minimax_key can be imported from global_methods."""
        from global_methods import set_minimax_key
        self.assertTrue(callable(set_minimax_key))

    @patch('global_methods.httpx')
    def test_get_minimax_answers_single_batch(self, mock_httpx):
        """Test get_minimax_answers with batch_size=1."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Paris"}}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_httpx.post.return_value = mock_response

        from task_eval.minimax_utils import get_minimax_answers

        os.environ['MINIMAX_API_KEY'] = 'test-key'
        in_data = {
            'conversation': {
                'session_1': [
                    {'speaker': 'Alice', 'text': 'I live in Paris.', 'dia_id': 'd1'},
                    {'speaker': 'Bob', 'text': 'Nice!', 'dia_id': 'd2'},
                ],
                'session_1_date_time': '2024-01-01',
            },
            'qa': [
                {'question': 'Where does Alice live?', 'answer': 'Paris', 'category': 1, 'evidence': []}
            ]
        }
        out_data = {
            'qa': [
                {'question': 'Where does Alice live?', 'answer': 'Paris', 'category': 1, 'evidence': []}
            ]
        }

        args = MagicMock()
        args.model = 'minimax-m2.5'
        args.batch_size = 1
        args.use_rag = False
        args.rag_mode = ''
        args.overwrite = True

        result = get_minimax_answers(in_data, out_data, 'minimax-m2.5_prediction', args)

        self.assertIn('minimax-m2.5_prediction', result['qa'][0])
        self.assertEqual(result['qa'][0]['minimax-m2.5_prediction'], 'Paris')

    @patch('global_methods.httpx')
    def test_get_minimax_answers_cat_5(self, mock_httpx):
        """Test get_minimax_answers with adversarial category 5 question."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "(a)"}}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_httpx.post.return_value = mock_response

        from task_eval.minimax_utils import get_minimax_answers

        os.environ['MINIMAX_API_KEY'] = 'test-key'
        in_data = {
            'conversation': {
                'session_1': [
                    {'speaker': 'Alice', 'text': 'Hello Bob', 'dia_id': 'd1'},
                    {'speaker': 'Bob', 'text': 'Hi Alice', 'dia_id': 'd2'},
                ],
                'session_1_date_time': '2024-01-01',
            },
            'qa': [
                {'question': 'Does Alice own a car?', 'answer': 'Not mentioned in the conversation', 'category': 5, 'evidence': []}
            ]
        }
        out_data = {
            'qa': [
                {'question': 'Does Alice own a car?', 'answer': 'Not mentioned in the conversation', 'category': 5, 'evidence': []}
            ]
        }

        args = MagicMock()
        args.model = 'minimax-m2.5'
        args.batch_size = 1
        args.use_rag = False
        args.rag_mode = ''
        args.overwrite = True

        result = get_minimax_answers(in_data, out_data, 'minimax-m2.5_prediction', args)
        self.assertIn('minimax-m2.5_prediction', result['qa'][0])

    @patch('global_methods.httpx')
    def test_get_minimax_answers_skip_existing(self, mock_httpx):
        """Test that existing predictions are skipped without overwrite."""
        from task_eval.minimax_utils import get_minimax_answers

        in_data = {
            'conversation': {
                'session_1': [
                    {'speaker': 'Alice', 'text': 'Hi', 'dia_id': 'd1'},
                    {'speaker': 'Bob', 'text': 'Hello', 'dia_id': 'd2'},
                ],
                'session_1_date_time': '2024-01-01',
            },
            'qa': [
                {'question': 'Q?', 'answer': 'A', 'category': 1, 'evidence': []}
            ]
        }
        out_data = {
            'qa': [
                {'question': 'Q?', 'answer': 'A', 'category': 1, 'evidence': [],
                 'minimax-m2.5_prediction': 'existing_answer'}
            ]
        }

        args = MagicMock()
        args.model = 'minimax-m2.5'
        args.batch_size = 1
        args.use_rag = False
        args.rag_mode = ''
        args.overwrite = False

        result = get_minimax_answers(in_data, out_data, 'minimax-m2.5_prediction', args)
        # Should not have called the API
        mock_httpx.post.assert_not_called()
        self.assertEqual(result['qa'][0]['minimax-m2.5_prediction'], 'existing_answer')


class TestEvaluateMinimaxShellScript(unittest.TestCase):
    """Test that the evaluation shell script exists and has correct content."""

    def test_script_exists(self):
        """Test that evaluate_minimax.sh exists."""
        script_path = Path(__file__).parent.parent / 'scripts' / 'evaluate_minimax.sh'
        self.assertTrue(script_path.exists())

    def test_script_sources_env(self):
        """Test that the script sources env.sh."""
        script_path = Path(__file__).parent.parent / 'scripts' / 'evaluate_minimax.sh'
        content = script_path.read_text()
        self.assertIn('source scripts/env.sh', content)

    def test_script_runs_m2_5(self):
        """Test that the script evaluates minimax-m2.5."""
        script_path = Path(__file__).parent.parent / 'scripts' / 'evaluate_minimax.sh'
        content = script_path.read_text()
        self.assertIn('minimax-m2.5', content)

    def test_script_runs_highspeed(self):
        """Test that the script evaluates minimax-m2.5-highspeed."""
        script_path = Path(__file__).parent.parent / 'scripts' / 'evaluate_minimax.sh'
        content = script_path.read_text()
        self.assertIn('minimax-m2.5-highspeed', content)


class TestEnvScript(unittest.TestCase):
    """Test that env.sh includes MINIMAX_API_KEY."""

    def test_env_has_minimax_key(self):
        """Test that env.sh contains MINIMAX_API_KEY export."""
        env_path = Path(__file__).parent.parent / 'scripts' / 'env.sh'
        content = env_path.read_text()
        self.assertIn('MINIMAX_API_KEY', content)


if __name__ == '__main__':
    unittest.main()

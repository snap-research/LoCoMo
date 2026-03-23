"""Integration tests for MiniMax provider in LoCoMo.

These tests verify the full pipeline works with the MiniMax API.
They require MINIMAX_API_KEY to be set in the environment.
"""

import sys
import os
import json
import unittest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

MINIMAX_API_KEY = os.environ.get('MINIMAX_API_KEY', '')
SKIP_REASON = "MINIMAX_API_KEY not set; skipping integration tests"


@unittest.skipUnless(MINIMAX_API_KEY, SKIP_REASON)
class TestMinimaxAPIIntegration(unittest.TestCase):
    """Integration tests that call the real MiniMax API."""

    def test_run_minimax_m2_5(self):
        """Test a real API call to MiniMax-M2.5."""
        from global_methods import run_minimax

        # M2.5 uses thinking tokens that count against max_tokens,
        # so we need a larger budget
        result = run_minimax(
            "What is 2 + 2? Answer with just the number.",
            1024,
            "minimax-m2.5"
        )
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)
        self.assertIn('4', result)

    def test_run_minimax_m2_5_highspeed(self):
        """Test a real API call to MiniMax-M2.5-highspeed."""
        from global_methods import run_minimax

        result = run_minimax(
            "Name a color of the rainbow. Answer with just one word.",
            1024,
            "minimax-m2.5-highspeed"
        )
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_minimax_qa_pipeline(self):
        """Test the full MiniMax QA pipeline with a minimal conversation."""
        from task_eval.minimax_utils import get_minimax_answers
        from unittest.mock import MagicMock

        in_data = {
            'conversation': {
                'session_1': [
                    {'speaker': 'Alice', 'text': 'I just adopted a golden retriever named Max!', 'dia_id': 'd1'},
                    {'speaker': 'Bob', 'text': 'That is wonderful! What breed is he?', 'dia_id': 'd2'},
                    {'speaker': 'Alice', 'text': 'He is a golden retriever, about 2 years old.', 'dia_id': 'd3'},
                ],
                'session_1_date_time': '2024-03-15',
            },
            'qa': [
                {'question': "What is Alice's dog's name?", 'answer': 'Max', 'category': 1, 'evidence': ['d1']}
            ]
        }
        out_data = {
            'qa': [
                {'question': "What is Alice's dog's name?", 'answer': 'Max', 'category': 1, 'evidence': ['d1']}
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
        prediction = result['qa'][0]['minimax-m2.5_prediction'].lower()
        self.assertIn('max', prediction)


if __name__ == '__main__':
    unittest.main()

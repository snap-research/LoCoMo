"""Regression test for category-5 (adversarial) scoring.

The released data/locomo10.json stores the ground truth for category-5
(adversarial) questions under the key 'adversarial_answer', not 'answer'.
eval_question_answering previously read line['answer'] unconditionally and
raised KeyError on every category-5 question, crashing the whole pipeline.

Run with: python -m task_eval.test_cat5_eval   (or pytest task_eval/test_cat5_eval.py)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from task_eval.evaluation import eval_question_answering


def test_cat5_scores_without_answer_key():
    # A category-5 record exactly as released: no 'answer', only
    # 'adversarial_answer'. Correct model behaviour is "not mentioned".
    qas = [{
        "question": "What did Caroline realize after her charity race?",
        "evidence": ["D2:3"],
        "category": 5,
        "adversarial_answer": "self-care is important",
        "model_prediction": "Not mentioned in the conversation",
    }]
    scores, _lens, _recall = eval_question_answering(qas, 'model_prediction')
    assert scores == [1], scores


if __name__ == "__main__":
    test_cat5_scores_without_answer_key()
    print("OK: category-5 scoring works on released-format data")

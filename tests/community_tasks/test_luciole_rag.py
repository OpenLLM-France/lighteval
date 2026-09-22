# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import logging
from types import SimpleNamespace

import pytest

from community_tasks import luciole_rag


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ('{"refused": true, "justification": "missing evidence"}', True),
        ('```json\n{"refused": false, "justification": "answered"}\n```', False),
        ('Result: {"refused": "yes", "justification": "missing evidence"}', True),
        ('{"refused": false, "justification": "the literal {x} is harmless"}', False),
    ],
)
def test_parse_refusal_judge_response_variants(response, expected):
    assert luciole_rag.parse_refusal_judge_response(response) is expected


@pytest.mark.parametrize(
    "response",
    [
        "The assistant refused to answer.",
        '{"unknown": true}',
        '{"refused": "maybe"}',
        None,
    ],
)
def test_parse_refusal_judge_response_rejects_ambiguous_output(response):
    assert luciole_rag.parse_refusal_judge_response(response) is None


def test_parse_failure_logs_raw_response(caplog):
    luciole_rag._judge_parse_failure_counts.clear()
    with caplog.at_level(logging.WARNING, logger=luciole_rag.__name__):
        assert luciole_rag.parse_refusal_judge_response("not JSON") is None
    assert "Raw response #1: 'not JSON'" in caplog.text


def test_factual_parser_keeps_false_refusal_populated():
    assert luciole_rag.parse_factual_judge_response(
        '{"score": 4, "refused": false, "justification": "mostly correct"}'
    ) == {"score": 4, "refused": False}


def test_false_refusal_and_refusal_recall_keep_false_as_zero():
    class FakeFactualJudge:
        def evaluate_answer_batch(self, **kwargs):
            return [{"score": 4, "refused": False}], [], []

    class FakeRefusalJudge:
        def evaluate_answer_batch(self, **kwargs):
            return [False], [], []

    response = SimpleNamespace(final_text=["A substantive answer"])
    answerable_doc = SimpleNamespace(
        query="Answerable?",
        specific={
            "is_unanswerable": False,
            "reference_answer": "answer",
            "supporting_facts_titles": ["source"],
            "context": "[source] evidence",
        },
    )
    unanswerable_doc = SimpleNamespace(query="Unanswerable?", specific={"is_unanswerable": True})

    factual_metric = object.__new__(luciole_rag.LucioleRagFactualJudge)
    factual_metric.judge = FakeFactualJudge()
    refusal_metric = object.__new__(luciole_rag.LucioleRagRefusalJudge)
    refusal_metric.judge = FakeRefusalJudge()

    factual = factual_metric.compute([response], [answerable_doc])[0]
    refusal = refusal_metric.compute([response], [unanswerable_doc])[0]

    assert factual["factual_judge_accuracy_ge_5"] == 0.0
    assert factual["false_refusal_judge"] == 0.0
    assert refusal["refusal_judge_recall"] == 0.0
    assert luciole_rag._nanmean_skip_none_with_stderr("metric")([None, 0.0])["metric"] == 0.0

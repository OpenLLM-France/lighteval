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


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ('{"score": 0, "refused": true, "justification": "refused"}', {"score": 1, "refused": True}),
        ('{"score": null, "refused": true}', {"score": 1, "refused": True}),
        ('{"refused": true}', {"score": 1, "refused": True}),
        ('{"score": 0, "refused": false}', None),
        ('{"score": 6, "refused": false}', None),
    ],
)
def test_factual_parser_keeps_refusals_scored_off_scale(response, expected):
    assert luciole_rag.parse_factual_judge_response(response) == expected


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


@pytest.mark.parametrize(
    ("response", "gold", "expected"),
    [
        ("The provided documents do not allow me to answer this question.", "no", 0.0),
        ("Revenue rose from 80 million to 85 million.", "5", 0.0),
        ("No, they were not.", "no", 1.0),
        ("Net sales were <quote>157,133</quote>.", "157,133", 1.0),
        ('Paid when amounts fall due<ref name="source_1">quote</ref>.', "when amounts fall due", 1.0),
        ("Accrued revenue was $5,824,000.", "5,824", 0.0),
        ("The grant was 5 million dollars.", "5,000,000", 1.0),
        ("The adjustment was $500,000.", "$0.5 million", 1.0),
        ("The ratio was 12.3.", "1.23", 0.0),
        ("The adjustment was $5.8 million.", "5,824", 0.0),
        ("Nonvested awards were $0.", "67.6", 0.0),
        ("The change was 0%.", "0", 1.0),
    ],
)
def test_answer_em_matches_whole_words_only(response, gold, expected):
    assert luciole_rag.compute_answer_em(response, gold) == expected


@pytest.mark.parametrize(
    ("response", "gold", "language", "expected"),
    [
        ("There were 5000 people.", "5", "en", 0.0),
        ("The amount was 5 dollars.", "5 million", "en", 0.0),
        ("The amount was $5,824,000.", "5.824 million", "en", 1.0),
        ("The amount was $5,824,000.", "5,824 thousand", "en", 1.0),
        ("The change was +5%.", "-5", "en", 1.0),
        ("The change was -5%.", "5", "en", 1.0),
        ("The change was −5%.", "-5", "en", 1.0),
        ("The loss was -$5 million.", "-5000000", "en", 1.0),
        ("The loss was $-5 million.", "-5000000", "en", 1.0),
        ("The loss was ($5 million).", "-5000000", "en", 1.0),
        ("The amount was $5 million.", "(5000000)", "en", 1.0),
        ("The value is 5.2.", "2", "en", 0.0),
        ("La valeur est 1,23.", "23", "fr", 0.0),
        ("La valeur est 1,23.", "1.23", "fr", 1.0),
        ("La valeur est 1,234.", "1.234", "fr", 1.0),
        ("The value is 1,234.", "1234", "en", 1.0),
        ("Le montant est 1 234,56 €.", "1234.56", "fr", 1.0),
        ("Le montant est 1\u00a0234,56 €.", "1234.56", "fr", 1.0),
        ("Le montant est 1\u202f234,56 €.", "1234.56", "fr", 1.0),
        ("Le montant est 1.234,56 €.", "1234.56", "fr", 1.0),
        ("The amount is $1,234.56.", "1234.56", "en", 1.0),
        ("Le montant est 0,5 million €.", "500000", "fr", 1.0),
        ("Le montant est 2 milliards €.", "2000000000", "fr", 1.0),
        ("La variation est -1,23 %.", "1,23", "fr", 1.0),
        ("La variation est -1,23 %.", "-1,23", "fr", 1.0),
        ("The amount is 0.", "0.000000001", "en", 0.0),
        ("The Minimum Pension Liability balance in 2018 was $10.1 million.", "(10.1)", "en", 1.0),
        ("Accumulated depreciation for 2019 is $2,378.2 million.", "2,378.2", "en", 1.0),
        ("Other financial income decreased by €23 million.", "-23", "en", 1.0),
        ("Net cash used was $415,772 thousand.", "(415,772)", "en", 1.0),
        ("The amount was 5 million.", "5 billion", "en", 0.0),
    ],
)
def test_answer_em_numeric_values(response, gold, language, expected):
    assert luciole_rag.compute_answer_em(response, gold, language=language) == expected


def test_sample_metrics_use_document_language_for_numbers():
    doc = SimpleNamespace(specific={"reference_answer": "1.234", "language": "fr"})
    response = SimpleNamespace(final_text=["La valeur est 1,234."])
    assert luciole_rag.LucioleRagSampleMetrics().compute(response, doc)["answer_em"] == 1.0


def test_citation_f1_is_zero_when_expected_sources_are_not_cited():
    assert luciole_rag.evaluate_citations([], ["source"]) == (None, 0.0, 0.0)
    assert luciole_rag.evaluate_citations([], []) == (None, None, None)


def test_answer_em_stays_fast_on_degenerate_markup_loops():
    import time

    response = "A decrease of $4,001,000" + " <br>" * 2000
    start = time.perf_counter()
    assert luciole_rag.compute_answer_em(response, "4,001,000") == 1.0
    assert time.perf_counter() - start < 1.0


def test_every_subset_has_a_source():
    assert set(luciole_rag.DATASET_SUBSETS) == set(luciole_rag.DATASET_SOURCES)
    tasks = {t.name: t for t in luciole_rag.TASKS_TABLE}
    openrag = tasks["luciole_rag:openrag_bench"]
    assert (openrag.hf_repo, openrag.hf_subset) == ("OpenLLM-France/OpenRAG-Bench", "luciole")
    assert openrag.hf_revision


def test_convert_openrag_row():
    from community_tasks.convert_openrag_to_luciole import convert_row

    def chunk(text, filename, role):
        return {"text": text, "filename": filename, "role": role}

    row = {
        "id": "q1",
        "question": "What was the revenue?",
        "answer": "5",
        "answerable": True,
        "contexts": [
            chunk("a", "report.pdf", "distractor"),
            chunk("b", "report.pdf", "gold"),
            chunk("c", "other.pdf", "gold_alternative"),
            chunk("d", "report.pdf", "topic"),
        ],
    }
    assert convert_row(row) == {
        "id": "q1",
        "query": "What was the revenue?",
        "retrieved_documents": ["a", "b", "c", "d"],
        "titles": ["report.pdf", "report.pdf [2]", "other.pdf", "report.pdf [3]"],
        "supporting_index": [1, 2],
        "answer": "5",
    }
    unanswerable = convert_row({**row, "answerable": False, "contexts": [chunk("a", "x.pdf", "distractor")]})
    assert (unanswerable["supporting_index"], unanswerable["answer"]) == ([], "")

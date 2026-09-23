# MIT License

# Copyright (c) 2026 OpenLLM-France

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

"""
name:
BLEnD (short-answer, multilingual)

dataset:
uilab/BLEnD

abstract:
BLEnD evaluates everyday cultural knowledge with short-answer questions written in the local
language. This implements the open-ended short-answer task over the SemEval locales (each a
Language_Country split of the ``semeval-annotations`` config, incl. French): the model answers a
question in the target language and is scored against the set of human-provided answers.

languages:
17 SemEval locales (incl. French — France), 500 questions each.

tags:
multilingual, cultural, commonsense, short-answer, generative

paper:
https://arxiv.org/abs/2406.09948

--------------------------------------------------------------------------------------------------
How to run -- task strings are ``community|blend_<code>|0`` (pass ``--custom-tasks <this file>``):

    French:  community|blend_fr|0
    others:  community|blend_ar_eg  blend_ar_ma  blend_ar_sa  blend_eu  blend_bg  blend_en_au
             blend_ga  blend_ja  blend_ms_sg  blend_zh_sg  blend_zh_tw  blend_es_ec  blend_sv
             blend_tl  blend_ta_sg  blend_ta_lk

Each locale has 500 questions (evaluation-only -> run 0-shot). The gold set for a question is the
union of all human answers (target-language ``answers`` + their ``en_answers``); a prediction is
counted correct if any gold answer appears in it (accent/case/punctuation-insensitive, word-bounded),
which mirrors BLEnD's rule-based short-answer matching. Metric: ``blend_acc``.
"""

import re
import unicodedata

import numpy as np

from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.metrics.utils.metric_utils import SampleLevelMetric, SamplingMethod
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


HF_REPO = "uilab/BLEnD"
HF_CONFIG = "semeval-annotations"  # self-contained: target-language `question` + human `annotations`

# short code (task-name suffix) -> Language_Country split of the `semeval-annotations` config.
LANGUAGES = {
    "fr": "French_France",
    "ar_eg": "Arabic_Egypt",
    "ar_ma": "Arabic_Morocco",
    "ar_sa": "Arabic_SaudiArabia",
    "eu": "Basque_BasqueCountry",
    "bg": "Bulgarian_Bulgaria",
    "en_au": "English_Australia",
    "ga": "Irish_Ireland",
    "ja": "Japanese_Japan",
    "ms_sg": "Malay_Singapore",
    "zh_sg": "Mandarin_Singapore",
    "zh_tw": "Mandarin_Taiwan",
    "es_ec": "Spanish_Ecuador",
    "sv": "Swedish_Sweden",
    "tl": "Tagalog_Philippines",
    "ta_sg": "Tamil_Singapore",
    "ta_lk": "Tamil_SriLanka",
}


def _norm(text) -> str:
    """Lowercase, strip accents/diacritics and punctuation, collapse whitespace (script-agnostic)."""
    text = unicodedata.normalize("NFD", str(text))
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")  # drop combining marks
    text = re.sub(r"[^\w\s]", " ", text.lower(), flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _gold_answers(line) -> list:
    """Union of human answers for a question (target-language + English variants), deduplicated."""
    golds = []
    for group in line.get("annotations") or []:
        if (group.get("count") or 0) < 1:
            continue
        for key in ("answers", "en_answers"):
            golds += [a for a in (group.get(key) or []) if a and str(a).strip()]
    return list(dict.fromkeys(golds))


def blend_prompt_fn(line, task_name: str = None):
    golds = _gold_answers(line)
    if not golds:  # only 'I don't know' answers were collected -> skip (nothing to score against)
        return None
    return Doc(
        task_name=task_name,
        query=line["question"].strip(),
        choices=golds,
        gold_index=list(range(len(golds))),
    )


class BlendShortAnswerMatch(SampleLevelComputation):
    """Correct if any gold answer occurs in the generation (normalized, word-bounded) or matches it."""

    def compute(self, model_response, doc, **kwargs) -> float:
        pred = _norm(model_response.text[0]) if getattr(model_response, "text", None) else ""
        if not pred:
            return 0.0
        for gold in doc.choices:
            g = _norm(gold)
            if not g:
                continue
            if g == pred or re.search(rf"(?<!\w){re.escape(g)}(?!\w)", pred):
                return 1.0
        return 0.0


blend_metric = SampleLevelMetric(
    metric_name="blend_acc",
    sample_level_fn=BlendShortAnswerMatch(),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)


def _make_task(code, split):
    return LightevalTaskConfig(
        name=f"blend_{code}",
        suite=["community"],
        prompt_function=blend_prompt_fn,
        hf_repo=HF_REPO,
        hf_subset=HF_CONFIG,
        hf_avail_splits=[split],
        evaluation_splits=[split],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=64,
        metrics=[blend_metric],
        stop_sequence=["\n"],
        version=0,
    )


TASKS_TABLE = [_make_task(code, split) for code, split in LANGUAGES.items()]

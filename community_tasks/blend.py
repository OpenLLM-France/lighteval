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

--------------------------------------------------------------------------------------------------
Scoring — follows the official repo (github.com/nlee0212/BLEnD, ``evaluation/exact_match.py``,
``soft_exact_match``):
  * A question is SKIPPED (not counted) when ``no-answer + not-applicable >= 3`` or ``idk >= 5``
    annotators, or when it has no answers (prompt fn returns None -> excluded from the eval set,
    so the corpus mean is over valid questions, as in the paper).
  * A prediction is CORRECT if any human answer is contained in it, after lowercasing and removing
    accents (BLEnD's rule; hyphen/space variants collapse under normalization).
  * Two scores are reported, exactly like the paper:
      - ``blend_acc``      : binary (any human answer matched).
      - ``blend_weighted`` : vote-weighted -- the matched answer group's ``count / max_count``
                             (reward for hitting the answer most annotators agreed on).
  * English answer variants (``en_answers``) are included in the gold set (lenient toward a model
    answering in English).

KNOWN APPROXIMATION vs the official code: BLEnD additionally applies per-language lemmatizers/stemmers
(konlpy, jieba, hazm, qalsadi, indic-nlp, spark-nlp, spaCy, ...) as a fallback when plain containment
fails, to catch morphological variants. Those heavy per-language dependencies are NOT reproduced here;
we use containment over accent-normalized text plus the multiple human-provided variants. Note the
official repo has *no French branch* at all, so for French the effective rule there is exactly this
containment check -- i.e. French is faithful; scores for morphologically rich languages (Arabic,
Tamil, ...) may read slightly lower than the official lemmatized numbers.
The prompt is the bare question (the paper's persona / instruction-wrapper variants are not applied).
"""

import re
import unicodedata

import numpy as np

from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.metrics.utils.metric_utils import SampleLevelMetricGrouping, SamplingMethod
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


def _answer_groups(line):
    """[(normalized answers, vote count)] for each human answer group with >=1 vote (else empty)."""
    groups = []
    for group in line.get("annotations") or []:
        count = group.get("count") or 0
        if count < 1:
            continue
        answers = [_norm(a) for a in (group.get("answers") or []) + (group.get("en_answers") or [])]
        answers = [a for a in answers if a]
        if answers:
            groups.append((answers, count))
    return groups


def _skip(line) -> bool:
    """Official BLEnD skip rule: too many annotators could not / would not answer."""
    idks = line.get("idks") or {}
    return (idks.get("no-answer", 0) + idks.get("not-applicable", 0) >= 3) or (idks.get("idk", 0) >= 5)


def blend_prompt_fn(line, task_name: str = None):
    if _skip(line):
        return None
    groups = _answer_groups(line)
    if not groups:
        return None
    flat = [a for answers, _ in groups for a in answers]
    return Doc(
        task_name=task_name,
        query=line["question"].strip(),
        choices=list(dict.fromkeys(flat)),
        gold_index=list(range(len(set(flat)))),
        # Store groups as a list of dicts (a clean struct list), NOT (list, int) tuples: with
        # --save-details, pyarrow types each tuple as a sequence and chokes on the list `answers`
        # next to the int `count` ("cannot mix list and non-list, non-null values").
        specific={
            "groups": [{"answers": answers, "count": count} for answers, count in groups],
            "max_count": max(count for _, count in groups),
        },
    )


class BlendShortAnswer(SampleLevelComputation):
    """BLEnD soft exact match: any human answer contained in the (accent-normalized) generation.

    Returns the binary score and the vote-weighted score (matched group's count / max_count),
    mirroring ``soft_exact_match`` in the official repo.
    """

    def compute(self, model_response, doc, **kwargs) -> dict:
        pred = _norm(model_response.text[0]) if getattr(model_response, "text", None) else ""
        groups = (doc.specific or {}).get("groups", [])
        max_count = (doc.specific or {}).get("max_count", 1) or 1
        if pred:
            # highest-vote matching group first, as in the official implementation
            for group in sorted(groups, key=lambda g: g["count"], reverse=True):
                if any(answer in pred for answer in group["answers"]):
                    return {"blend_acc": 1.0, "blend_weighted": group["count"] / max_count}
        return {"blend_acc": 0.0, "blend_weighted": 0.0}


blend_metric = SampleLevelMetricGrouping(
    metric_name=["blend_acc", "blend_weighted"],
    sample_level_fn=BlendShortAnswer(),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn={"blend_acc": np.mean, "blend_weighted": np.mean},
    higher_is_better={"blend_acc": True, "blend_weighted": True},
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

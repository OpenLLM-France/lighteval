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
uilab/BLEnD (config ``semeval-annotations`` -- the SemEval-2026 Task 7 variant, incl. French)

abstract:
BLEnD evaluates everyday cultural knowledge with short-answer questions written in the local
language. This implements the open-ended short-answer task over the SemEval locales (each a
Language_Country split): the model answers a question in the target language and is scored
against the set of human-provided answers.

languages:
17 SemEval locales (incl. French — France), ~500 questions each.

tags:
multilingual, cultural, commonsense, short-answer, generative

paper:
https://arxiv.org/abs/2406.09948

--------------------------------------------------------------------------------------------------
How to run -- task strings are ``community|blend_<code>|<n_fewshot>`` (pass ``--custom-tasks <this file>``):

    0-shot French:  community|blend_fr|0
    5-shot French:  community|blend_fr|5
    others:  community|blend_ar_eg  blend_ar_ma  blend_ar_sa  blend_eu  blend_bg  blend_en_au
             blend_ga  blend_ja  blend_ms_sg  blend_zh_sg  blend_zh_tw  blend_es_ec  blend_sv
             blend_tl  blend_ta_sg  blend_ta_lk

--------------------------------------------------------------------------------------------------
Prompt — follows the paper's short-answer instruction template (Appendix C.2.1, ``inst``):
"Read the following question and provide a single answer without any explanations. Question: {q}
Answer:". The SemEval config ships NO prompt file, so the instruction + ``Question:/Réponse:``
labels are provided here, per locale, in the local language (see ``_INSTRUCTIONS`` / ``_LABELS``).

  * The instruction is carried on ``Doc.instruction`` (emitted once by lighteval) and, for
    chat/instruct models, as a SYSTEM message (``specific["instruction_as_system"]``); base
    (completion) models get it as a plain-text header. The query is ``"{Q} : {question}\n{A} :"``.
  * The explicit ``Answer:`` marker is the key difference from a bare-question prompt: without it,
    base models in a few-shot context tend to HALLUCINATE the numeric-answer instruction that the
    dataset attaches to count questions ("(Répondez en chiffres arabes...)") and reply with a bare
    number on word questions. The marker + instruction give a clear answer slot, which fixes this
    for both 0-shot and few-shot.
  * The same task runs 0-shot (``|0``) or few-shot (``|5``); few-shot demonstrations show the raw,
    accented human answer (the most-agreed one).

Scoring — follows the official repo (github.com/nlee0212/BLEnD, ``evaluation/exact_match.py``,
``soft_exact_match`` -> SEM-B / SEM-W):
  * A question is SKIPPED (not counted) when ``no-answer + not-applicable >= 3`` or ``idk >= 5``
    annotators, or when it has no answers (prompt fn returns None -> excluded, as in the paper).
  * A prediction is CORRECT if any human answer appears in it with WORD BOUNDARIES, after
    lowercasing and removing accents. Word-boundary matching (not raw substring) mirrors the
    official tokenize-then-match and avoids spurious hits such as gold "1" matching "18"/"2018".
  * Two scores, as in the paper:
      - ``blend_acc``      (SEM-B): binary (any human answer matched).
      - ``blend_weighted`` (SEM-W): vote-weighted -- matched group's ``count / max_count``.
  * English answer variants (``en_answers``) are included in the gold set (lenient toward a model
    answering in English).

KNOWN APPROXIMATION vs the official code: BLEnD additionally applies per-language lemmatizers/stemmers
(konlpy, jieba, hazm, qalsadi, indic-nlp, spark-nlp, spaCy, ...) as a fallback when matching fails,
to catch morphological variants. Those heavy per-language dependencies are NOT reproduced here; we
use word-boundary matching over accent-normalized text plus the multiple human-provided variants
(the dataset usually lists singular/plural forms separately, which covers most of the gap). Scores
for morphologically rich languages (Arabic, Tamil, ...) may read slightly lower than the official
lemmatized numbers.
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

# Short-answer instruction (paper's ``inst`` template) per locale, in the local language. The
# SemEval config provides no prompt, so these are supplied here. French and English are
# authoritative; the others are best-effort translations and welcome review by native speakers.
_INSTRUCTIONS = {
    "fr": "Répondez à la question suivante en donnant une seule réponse, sans explication.",
    "en_au": "Answer the following question with a single answer, without any explanation.",
    "es_ec": "Responde la siguiente pregunta con una sola respuesta, sin ninguna explicación.",
    "ar_eg": "أجب عن السؤال التالي بإجابة واحدة فقط، دون أي شرح.",
    "ar_ma": "أجب عن السؤال التالي بإجابة واحدة فقط، دون أي شرح.",
    "ar_sa": "أجب عن السؤال التالي بإجابة واحدة فقط، دون أي شرح.",
    "eu": "Erantzun hurrengo galderari erantzun bakar batekin, azalpenik gabe.",
    "bg": "Отговорете на следния въпрос с един-единствен отговор, без никакво обяснение.",
    "ga": "Freagair an cheist seo a leanas le freagra amháin, gan aon mhíniú.",
    "ja": "次の質問に、説明なしで答えを一つだけ書いてください。",
    "ms_sg": "Jawab soalan berikut dengan satu jawapan sahaja, tanpa sebarang penjelasan.",
    "zh_sg": "请用一个答案回答以下问题，不要作任何解释。",
    "zh_tw": "請用一個答案回答以下問題，不要作任何解釋。",
    "sv": "Besvara följande fråga med ett enda svar, utan någon förklaring.",
    "tl": "Sagutin ang sumusunod na tanong nang may iisang sagot, nang walang anumang paliwanag.",
    "ta_sg": "பின்வரும் கேள்விக்கு எந்த விளக்கமும் இல்லாமல் ஒரே ஒரு பதில் மட்டும் கொடுங்கள்.",
    "ta_lk": "பின்வரும் கேள்விக்கு எந்த விளக்கமும் இல்லாமல் ஒரே ஒரு பதில் மட்டும் கொடுங்கள்.",
}

# (question_label, answer_label) per locale, in the local language.
_LABELS = {
    "fr": ("Question", "Réponse"),
    "en_au": ("Question", "Answer"),
    "es_ec": ("Pregunta", "Respuesta"),
    "ar_eg": ("السؤال", "الإجابة"),
    "ar_ma": ("السؤال", "الإجابة"),
    "ar_sa": ("السؤال", "الإجابة"),
    "eu": ("Galdera", "Erantzuna"),
    "bg": ("Въпрос", "Отговор"),
    "ga": ("Ceist", "Freagra"),
    "ja": ("質問", "回答"),
    "ms_sg": ("Soalan", "Jawapan"),
    "zh_sg": ("问题", "答案"),
    "zh_tw": ("問題", "答案"),
    "sv": ("Fråga", "Svar"),
    "tl": ("Tanong", "Sagot"),
    "ta_sg": ("கேள்வி", "பதில்"),
    "ta_lk": ("கேள்வி", "பதில்"),
}


def _norm(text) -> str:
    """Lowercase, strip accents/diacritics and punctuation, collapse whitespace (script-agnostic)."""
    text = unicodedata.normalize("NFD", str(text))
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")  # drop combining marks
    text = re.sub(r"[^\w\s]", " ", text.lower(), flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _answer_groups(line):
    """[(raw answers, normalized answers, vote count)] for each human group with >=1 vote.

    ``raw`` keeps the original (accented) strings for few-shot display; ``norm`` is used for
    scoring. French answers come before the English (``en_answers``) variants.
    """
    groups = []
    for group in line.get("annotations") or []:
        count = group.get("count") or 0
        if count < 1:
            continue
        raw = [a for a in (group.get("answers") or []) + (group.get("en_answers") or []) if a and str(a).strip()]
        norm = [n for n in (_norm(a) for a in raw) if n]
        if norm:
            groups.append((raw, norm, count))
    return groups


def _skip(line) -> bool:
    """Official BLEnD skip rule: too many annotators could not / would not answer."""
    idks = line.get("idks") or {}
    return (idks.get("no-answer", 0) + idks.get("not-applicable", 0) >= 3) or (idks.get("idk", 0) >= 5)


def _make_prompt_fn(code):
    """Build the prompt function for one locale (instruction + ``Question:/Réponse:`` scaffold)."""
    instruction = _INSTRUCTIONS.get(code) or _INSTRUCTIONS["en_au"]
    q_label, a_label = _LABELS.get(code, ("Question", "Answer"))

    def prompt_fn(line, task_name: str = None):
        if _skip(line):
            return None
        groups = _answer_groups(line)
        if not groups:
            return None
        # Few-shot demonstrations show the raw, accented answer of the most-agreed group first.
        groups_by_votes = sorted(groups, key=lambda g: g[2], reverse=True)
        raw_flat = [a for raw, _, _ in groups_by_votes for a in raw]
        choices = list(dict.fromkeys(raw_flat))
        return Doc(
            task_name=task_name,
            instruction=instruction + "\n",  # emitted once by lighteval (system msg for chat models)
            query=f"{q_label} : {line['question'].strip()}\n{a_label} :",
            choices=choices,
            gold_index=list(range(len(choices))),
            specific={
                # Normalized groups for scoring (a clean struct list, not (list, int) tuples, so
                # --save-details' pyarrow typing does not choke on mixing list and int).
                "groups": [{"answers": norm, "count": count} for _, norm, count in groups],
                "max_count": max(count for _, _, count in groups),
                # Deliver the instruction as a system message for chat/instruct models (ignored by
                # completion/base models, which receive it as a plain-text header).
                "instruction_as_system": True,
            },
        )

    return prompt_fn


class BlendShortAnswer(SampleLevelComputation):
    """BLEnD soft exact match: any human answer appears in the (accent-normalized) generation,
    matched at WORD BOUNDARIES (so gold "1" does not match "18").

    Returns the binary score (SEM-B) and the vote-weighted score (SEM-W: matched group's
    count / max_count), mirroring ``soft_exact_match`` in the official repo.
    """

    def compute(self, model_response, doc, **kwargs) -> dict:
        pred = _norm(model_response.text[0]) if getattr(model_response, "text", None) else ""
        groups = (doc.specific or {}).get("groups", [])
        max_count = (doc.specific or {}).get("max_count", 1) or 1
        if pred:
            padded = f" {pred} "  # word-boundary matching over space-separated normalized tokens
            # highest-vote matching group first, as in the official implementation
            for group in sorted(groups, key=lambda g: g["count"], reverse=True):
                if any(f" {answer} " in padded for answer in group["answers"]):
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
        prompt_function=_make_prompt_fn(code),
        hf_repo=HF_REPO,
        hf_subset=HF_CONFIG,
        hf_avail_splits=[split],
        evaluation_splits=[split],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=64,
        metrics=[blend_metric],
        stop_sequence=["\n"],
        version=1,
    )


TASKS_TABLE = [_make_task(code, split) for code, split in LANGUAGES.items()]

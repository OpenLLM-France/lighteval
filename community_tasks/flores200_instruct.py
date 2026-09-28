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
FLORES-200 (instruction-tuned / chat variant)

dataset:
facebook/flores

abstract:
An instruction-style variant of the ``flores200`` translation benchmark, meant for chat / instruct
(and reasoning) models rather than base models. It differs from ``flores200`` in two ways:

  1. Prompt: instead of the bare ``EN: <src> FR:`` continuation, it prepends an explicit instruction
     ("Translate the following text from <source> to <target>. Output only the translation."), the
     standard instruction-tuned translation prompt (cf. AfroBench flores ``prompt_2``, and Hendy et
     al. 2023 / Vilar et al. 2022 / Zhang et al. 2023, arXiv:2301.07069).
  2. Scoring: the translation is *extracted* from the model output before scoring, so reasoning
     traces and trailing commentary do not pollute BLEU / chrF++ / COMET / MetricX. ``flores200``
     relies on a ``\\n`` stop sequence to cut the answer, which cannot be used with reasoning models
     (their thinking is full of newlines); this variant drops that stop and extracts instead.

Same dataset, splits, language handling and metrics as ``flores200`` — only the prompt and the
answer extraction change.

languages:
Configurable list of pairs (see ``LANGUAGE_PAIRS`` below; English<->French by default).

tags:
multilingual, translation, generative, instruction-tuned, reasoning

paper:
https://arxiv.org/abs/2207.04672 (FLORES-200 / NLLB)

--------------------------------------------------------------------------------------------------
How to run (pass ``--custom-tasks <this file>``); task strings are ``community|flores200_instruct:<src>-<tgt>|<k>``:

    community|flores200_instruct:eng_Latn-fra_Latn|0
    community|flores200_instruct:fra_Latn-eng_Latn|5

To add a language pair, add it to ``LANGUAGE_PAIRS`` and make sure both codes are in ``LANGUAGE_NAMES``.

NOTE on reasoning models and generation budget: the answer extraction fixes the *format* (reasoning
tags and commentary are removed), but a reasoning model still needs enough tokens to finish thinking
*and* produce the translation. On backends with two-phase thinking generation, the reasoning gets its
own budget and ``GENERATION_SIZE`` counts only the answer. On backends without it (async vLLM, remote
endpoints), raise ``GENERATION_SIZE`` and/or pass a provider-native reasoning budget, otherwise the
reasoning can consume the whole budget and leave no translation to extract.
"""

import copy
import re

from lighteval.metrics.metrics import Metrics
from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.metrics.sample_preparator import Preparator
from lighteval.metrics.utils.metric_utils import Metric
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.templates.translation import get_translation_prompt_function
from lighteval.tasks.templates.utils.formulation import CFFormulation
from lighteval.utils.language import Language, manage_duplicate_language_codes
from lighteval.utils.utils import remove_reasoning_tags


# Language pairs to build (FLORES-200 codes, e.g. "eng_Latn"). One per line so they are easy to
# add/remove. Both codes of each pair must have an entry in LANGUAGE_NAMES below.
LANGUAGE_PAIRS = [
    ("eng_Latn", "fra_Latn"),
    ("fra_Latn", "eng_Latn"),
]

# Human-readable names used in the instruction ("Translate ... from <name> to <name>").
LANGUAGE_NAMES = {
    "eng_Latn": "English",
    "fra_Latn": "French",
    "deu_Latn": "German",
    "spa_Latn": "Spanish",
    "ita_Latn": "Italian",
    "por_Latn": "Portuguese",
}

# Generation budget for the answer (no "\n" stop: reasoning models emit newlines, so we extract the
# translation from the output instead of cutting at the first newline). See the module docstring for
# the interaction with reasoning budgets.
GENERATION_SIZE = 1024


# ---------------------------------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------------------------------

# A leading label some models prefix to the translation ("Translation:", "Traduction :", ...).
_LEADIN_RE = re.compile(
    r"^\s*(?:here(?:'?s| is)? the translation|the translation is|voici la traduction|"
    r"translation|traduction|réponse|answer)\s*[:：]\s*",
    re.IGNORECASE,
)
# Start of trailing commentary on a new line (markdown emphasis, notes, explanations).
_COMMENTARY_RE = re.compile(r"\n\s*(?:\*\*|\*\(|note\b|explanation\b|explication)", re.IGNORECASE)


def extract_translation(text: str) -> str:
    """Extract just the translation from a (possibly chatty / reasoning) model output.

    Steps: drop reasoning traces (``<think>``/``[THINK]``, closed or primed), keep the first
    paragraph (models append commentary after a blank line), cut inline markdown/explanation
    markers, drop a leading label, and strip surrounding quotes/emphasis.
    """
    if not text:
        return ""
    text = remove_reasoning_tags(text).strip()
    text = re.split(r"\n\s*\n", text, maxsplit=1)[0]  # first paragraph (commentary follows a blank line)
    text = _COMMENTARY_RE.split(text, maxsplit=1)[0]  # or an inline commentary marker
    text = _LEADIN_RE.sub("", text, count=1)
    return text.strip().strip('*"“”\'' + " ")


def _response_with_extracted_text(model_response: ModelResponse) -> ModelResponse:
    """Shallow copy of the response whose text view (``final_text``) is the extracted translation."""
    extracted = copy.copy(model_response)
    extracted.text_post_processed = [extract_translation(t) for t in model_response.final_text]
    return extracted


class _ExtractionPreparator(Preparator):
    """Wrap a preparator (BLEU/chrF++/COMET/MetricX) so it prepares the extracted translation."""

    def __init__(self, inner: Preparator):
        self._inner = inner

    def prepare(self, doc, model_response, **kwargs):
        return self._inner.prepare(doc=doc, model_response=_response_with_extracted_text(model_response), **kwargs)


class _ExtractionComputation(SampleLevelComputation):
    """Wrap a sample-level computation (bleu_1/bleu_4) so it scores the extracted translation."""

    def __init__(self, inner: SampleLevelComputation):
        self._inner = inner

    def compute(self, doc, model_response, **kwargs):
        return self._inner.compute(doc=doc, model_response=_response_with_extracted_text(model_response), **kwargs)


def _with_extraction(base: Metric) -> Metric:
    """Clone a metric so it scores the extracted translation, reusing the (heavy) corpus fn instance."""
    inner = base.sample_level_fn
    wrapped = _ExtractionPreparator(inner) if isinstance(inner, Preparator) else _ExtractionComputation(inner)
    return type(base)(
        metric_name=base.metric_name,
        higher_is_better=base.higher_is_better,
        category=base.category,
        sample_level_fn=wrapped,
        corpus_level_fn=base.corpus_level_fn,  # shared: COMET/MetricX model + cache loaded once
        batched_compute=base.batched_compute,
    )


# Same metrics as flores200, each scoring the extracted translation.
METRICS = [
    _with_extraction(m.value)
    for m in (Metrics.chrf_plus, Metrics.bleu, Metrics.bleu_1, Metrics.bleu_4, Metrics.comet, Metrics.metricx)
]


# ---------------------------------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------------------------------


def _instruct_adapter(lang1: str, lang2: str):
    instruction = (
        f"Translate the following text from {LANGUAGE_NAMES[lang1]} to {LANGUAGE_NAMES[lang2]}. "
        "Output only the translation."
    )
    return lambda line: {
        "source_text": line[f"sentence_{lang1}"],
        "target_text": line[f"sentence_{lang2}"],
        "instruction": instruction,
    }


def _make_task(lang1: str, lang2: str) -> LightevalTaskConfig:
    return LightevalTaskConfig(
        name=f"flores200_instruct:{lang1}-{lang2}",
        prompt_function=get_translation_prompt_function(
            source_language=Language(manage_duplicate_language_codes(lang1.split("_")[0])),
            target_language=Language(manage_duplicate_language_codes(lang2.split("_")[0])),
            adapter=_instruct_adapter(lang1, lang2),
            formulation=CFFormulation(),
        ),
        suite=["community"],
        hf_repo="facebook/flores",
        hf_subset=f"{lang1}-{lang2}",
        hf_avail_splits=["dev", "devtest"],
        evaluation_splits=["devtest"],
        few_shots_split="dev",
        few_shots_select=None,
        generation_size=GENERATION_SIZE,
        metrics=METRICS,
        stop_sequence=None,  # no "\n" stop: extract the translation instead (reasoning-model safe)
        version=0,
    )


TASKS_TABLE = [_make_task(lang1, lang2) for lang1, lang2 in LANGUAGE_PAIRS]

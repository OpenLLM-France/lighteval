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
INCLUDE (multilingual)

dataset:
CohereLabs/include-base-44

abstract:
INCLUDE is a multilingual benchmark of regional/cultural knowledge multiple-choice questions
(4 options) collected from local exams and sources across 44 languages. This implementation
evaluates each language with the CF / MCF / Hybrid multiple-choice formulations and log-likelihood
scoring, per-language and in the language's own script (correct translation literals).

languages:
44 languages (incl. French, German, Spanish, Italian, Portuguese, Dutch, Basque, Chinese, Arabic,
Hindi, Japanese, Korean, Russian, ...)

tags:
multilingual, regional-knowledge, cultural, multiple-choice

paper:
https://arxiv.org/abs/2411.19799

--------------------------------------------------------------------------------------------------
Merged from two prior implementations:
  - OpenLLM-BPI custom_benchmarks/include.py  -> multilingual coverage, CF/MCF/Hybrid formulations,
    the `domain` and `regional_feature` dimensions, `CohereLabs/include-base-44`.
  - community_tasks/filipino_evals.py         -> clean lighteval community style, PMI normalization,
    and passing the *correct* per-language `Language` to the template.
Best of both, with two fixes:
  * OpenLLM's version passed `Language.FRENCH` for every language (wrong translation literals in the
    MCF/Hybrid templates); here each config uses its own `Language`.
  * OpenLLM's version exploded into language x domain x formulation x regional_feature tasks. Here the
    `domain`, `subject`, `regional_feature`, `country` and `level` fields are attached to each sample
    (``doc.specific``) so those breakdowns are recovered offline from the details, without thousands
    of tasks.

--------------------------------------------------------------------------------------------------
How to run -- task strings are ``community|include_<code>_<formulation>|<n_fewshot>`` where <code> is
the ISO 639-1 language code (fr, de, es, ...); pass ``--custom-tasks <this file>``:

    French:  community|include_fr_mcf|0   community|include_fr_cf|0   community|include_fr_hybrid|0
    others:  community|include_de_mcf|0   community|include_es_mcf|0   community|include_ar_mcf|0  ...

  Formulation: mcf  -> letters (A/B/C/D) scored by log-likelihood (the INCLUDE-standard setting)
               cf   -> answer texts scored by log-likelihood
               hybrid
  Few-shot examples are drawn from the `validation` split; evaluation on `test`. Metrics: `acc`
  (+ token/char/PMI-normalized variants). Per-sample `doc.specific` carries `domain`, `subject`,
  `regional_feature` (agnostic / region implicit / region explicit / culture), `country`, `level`
  for offline breakdowns.
"""

from lighteval.metrics.dynamic_metrics import LogLikelihoodAccMetric
from lighteval.metrics.normalizations import LogProbCharNorm, LogProbPMINorm, LogProbTokenNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.multilingual.utils.task_utils import get_metrics_for_formulation
from lighteval.tasks.requests import Doc
from lighteval.tasks.templates.multichoice import get_mcq_prompt_function
from lighteval.tasks.templates.utils.formulation import (
    CFFormulation,
    HybridFormulation,
    MCFFormulation,
)
from lighteval.utils.language import Language


HF_REPO = "CohereLabs/include-base-44"

FORMULATIONS = [MCFFormulation(), CFFormulation(), HybridFormulation()]

# dataset config name (language) -> (short ISO 639-1 code used in the task name, lighteval Language).
# The Language drives the template's translation literals; the short code gives names like `include_fr`.
LANGUAGES = {
    "Albanian": ("sq", Language.ALBANIAN),
    "Arabic": ("ar", Language.ARABIC),
    "Armenian": ("hy", Language.ARMENIAN),
    "Azerbaijani": ("az", Language.AZERBAIJANI),
    "Basque": ("eu", Language.BASQUE),
    "Belarusian": ("be", Language.BELARUSIAN),
    "Bengali": ("bn", Language.BENGALI),
    "Bulgarian": ("bg", Language.BULGARIAN),
    "Chinese": ("zh", Language.CHINESE),
    "Croatian": ("hr", Language.CROATIAN),
    "Dutch": ("nl", Language.DUTCH),
    "Estonian": ("et", Language.ESTONIAN),
    "Finnish": ("fi", Language.FINNISH),
    "French": ("fr", Language.FRENCH),
    "Georgian": ("ka", Language.GEORGIAN),
    "German": ("de", Language.GERMAN),
    "Greek": ("el", Language.GREEK),
    "Hebrew": ("he", Language.HEBREW),
    "Hindi": ("hi", Language.HINDI),
    "Hungarian": ("hu", Language.HUNGARIAN),
    "Indonesian": ("id", Language.INDONESIAN),
    "Italian": ("it", Language.ITALIAN),
    "Japanese": ("ja", Language.JAPANESE),
    "Kazakh": ("kk", Language.KAZAKH),
    "Korean": ("ko", Language.KOREAN),
    "Lithuanian": ("lt", Language.LITHUANIAN),
    "Malay": ("ms", Language.MALAY),
    "Malayalam": ("ml", Language.MALAYALAM),
    "Nepali": ("ne", Language.NEPALI),
    "North Macedonian": ("mk", Language.MACEDONIAN),
    "Persian": ("fa", Language.PERSIAN),
    "Polish": ("pl", Language.POLISH),
    "Portuguese": ("pt", Language.PORTUGUESE),
    "Russian": ("ru", Language.RUSSIAN),
    "Serbian": ("sr", Language.SERBIAN),
    "Spanish": ("es", Language.SPANISH),
    "Tagalog": ("tl", Language.TAGALOG),
    "Tamil": ("ta", Language.TAMIL),
    "Telugu": ("te", Language.TELUGU),
    "Turkish": ("tr", Language.TURKISH),
    "Ukrainian": ("uk", Language.UKRAINIAN),
    "Urdu": ("ur", Language.URDU),
    "Uzbek": ("uz", Language.UZBEK),
    "Vietnamese": ("vi", Language.VIETNAMESE),
}

# Extra per-sample fields kept on `doc.specific` for offline breakdowns (domain / cultural analysis).
_META_FIELDS = ("domain", "subject", "regional_feature", "country", "level")


def _adapter(line):
    return {
        "question": line["question"].strip(),
        "choices": [line["option_a"], line["option_b"], line["option_c"], line["option_d"]],
        "gold_idx": int(line["answer"]),  # `answer` is already a 0-based index into the options
    }


def _make_prompt_fn(language, formulation):
    base_fn = get_mcq_prompt_function(language, _adapter, formulation=formulation)

    def prompt_fn(line, task_name: str = None):
        doc = base_fn(line, task_name)
        if isinstance(doc, Doc):
            doc.specific = {**(doc.specific or {}), **{field: line.get(field) for field in _META_FIELDS}}
        return doc

    return prompt_fn


def _make_task(language_name, code, language, formulation):
    return LightevalTaskConfig(
        name=f"include_{code}_{formulation.name.lower()}",
        suite=["community"],
        prompt_function=_make_prompt_fn(language, formulation),
        hf_repo=HF_REPO,
        hf_subset=language_name,
        hf_avail_splits=["test", "validation"],
        evaluation_splits=["test"],
        few_shots_split="validation",
        few_shots_select="sequential",
        generation_size=-1,
        metrics=get_metrics_for_formulation(
            formulation,
            [
                LogLikelihoodAccMetric(normalization=LogProbTokenNorm()),
                LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
                LogLikelihoodAccMetric(normalization=LogProbPMINorm()),
            ],
        ),
        stop_sequence=["\n"],
        version=0,
    )


TASKS_TABLE = [
    _make_task(language_name, code, language, formulation)
    for language_name, (code, language) in LANGUAGES.items()
    for formulation in FORMULATIONS
]

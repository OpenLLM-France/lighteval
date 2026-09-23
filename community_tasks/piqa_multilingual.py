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
Global-PIQA (multilingual PIQA)

dataset:
mrlbenchmarks/global-piqa-nonparallel

abstract:
Multilingual physical-commonsense reasoning (PIQA-style: a goal + two candidate solutions, pick the
physically sensible one). Global-PIQA (MRL 2025 shared task) provides natively-authored,
culturally-grounded items per language. This mirrors the English ``lighteval|piqa`` task (same
``Question:/Answer:`` log-likelihood scoring), one task per language -- French included.

languages:
multilingual (incl. French: France and Canada)

tags:
commonsense, physical-reasoning, multiple-choice, multilingual

paper:

--------------------------------------------------------------------------------------------------
Task names (task string ``community|piqa-<code>|0``, pass ``--custom-tasks <this file>``):
    French:  community|piqa-fr        (France)      community|piqa-fr-ca   (Canada / Québec)
    e.g.:    community|piqa-en  piqa-de  piqa-es  piqa-it  piqa-pt  piqa-zh  piqa-ru  piqa-ar ...

Each item: ``prompt`` (goal) + ``solution0``/``solution1`` + ``label`` (0/1). Scored exactly like
``lighteval|piqa``: the model's per-solution log-likelihood, reported as ``acc`` and ``acc_norm``
(character-length normalized). Eval split ``test`` (100 items per language; the benchmark is
evaluation-only, so these run 0-shot).

Uses the *non-parallel* set (each language authored independently -> culturally authentic). For a
translated, cross-lingually comparable set, point ``HF_REPO`` at ``mrlbenchmarks/global-piqa-parallel``
(same configs and schema).
"""

from lighteval.metrics.metrics import Metrics
from lighteval.metrics.normalizations import LogProbCharNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


HF_REPO = "mrlbenchmarks/global-piqa-nonparallel"

# Short code -> Global-PIQA config (ISO 639-3 + script [+ region]). A curated multilingual set;
# extend it with any of the ~130 configs the dataset ships. French comes in two regional variants.
LANGUAGES = {
    "fr": "fra_latn_fran",
    "fr-ca": "fra_latn_cana",
    "en": "eng_latn",
    "de": "deu_latn",
    "es": "spa_latn_spai",
    "it": "ita_latn",
    "pt": "por_latn_port",
    "pt-br": "por_latn_braz",
    "nl": "nld_latn",
    "ru": "rus_cyrl",
    "zh": "cmn_hans",
    "zh-hant": "cmn_hant",
    "ja": "jpn_jpan",
    "ko": "kor_hang",
    "ar": "arb_arab",
    "hi": "hin_deva",
    "tr": "tur_latn",
    "pl": "pol_latn",
    "vi": "vie_latn",
    "id": "ind_latn",
    "fa": "pes_arab",
    "uk": "ukr_cyrl",
    "el": "ell_grek",
    "he": "heb_hebr",
    "ro": "ron_latn",
    "sv": "swe_latn",
    "fi": "fin_latn",
    "cs": "ces_latn",
    "hu": "hun_latn",
    "bn": "ben_beng",
    "ta": "tam_taml",
    "th": "tha_thai",
    "sw": "swh_latn",
}


def piqa_prompt_fn(line, task_name: str = None):
    """Same format as lighteval's ``piqa_harness``, over Global-PIQA fields."""
    return Doc(
        task_name=task_name,
        query=f"Question: {line['prompt']}\nAnswer:",
        choices=[f" {line['solution0']}", f" {line['solution1']}"],
        gold_index=int(line["label"]),
    )


def _make_task(code, hf_subset):
    return LightevalTaskConfig(
        name=f"piqa-{code}",
        suite=["community"],
        prompt_function=piqa_prompt_fn,
        hf_repo=HF_REPO,
        hf_subset=hf_subset,
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=-1,
        metrics=[
            Metrics.loglikelihood_acc,
            Metrics.loglikelihood_acc(
                sample_params={"logprob_normalization": LogProbCharNorm(ignore_first_space=True)}
            ),
        ],
        stop_sequence=["\n"],
        version=0,
    )


TASKS_TABLE = [_make_task(code, hf_subset) for code, hf_subset in LANGUAGES.items()]

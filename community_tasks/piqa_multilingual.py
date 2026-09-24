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
Multilingual physical-commonsense reasoning: a goal + two candidate solutions, pick the physically
sensible one. Global-PIQA (Global PIQA, EMNLP 2025 MRL shared task) provides natively-authored,
culturally-grounded items in 100+ language varieties, incl. French (France & Canada).

languages:
33 curated language varieties (incl. French: France and Canada).

tags:
commonsense, physical-reasoning, multiple-choice, multilingual

paper:
https://arxiv.org/abs/2510.24081

--------------------------------------------------------------------------------------------------
Two task variants per language, matching the paper's two evaluation modes (both zero-shot, prompt in
the target language; metric = accuracy):

  * ``piqa_<code>``      -- for BASE (pretrained-only) models: the log-probability of each candidate
                           solution given the goal is compared (``acc``, plus length-normalized
                           ``acc_norm``). This is the paper's base-model protocol.
  * ``piqa_<code>_gen``  -- for INSTRUCT models: the goal + the two options (A/B) are shown, the model
                           generates an answer, and it is scored by string matching (``piqa_gen_acc``).
                           This is the paper's instruction-tuned-model protocol.

How to run (pass ``--custom-tasks <this file>``):
    French, base:     community|piqa_fr|0        French, instruct:  community|piqa_fr_gen|0
    Canada:           community|piqa_fr_ca|0 / community|piqa_fr_ca_gen|0
    others:           community|piqa_de|0  piqa_es_gen  piqa_zh  piqa_ar_gen  ...

KNOWN APPROXIMATIONS vs the official code:
  - The paper normalizes the base-model log-prob by the solution's length in *bytes*; lighteval offers
    character-length normalization (used here as ``acc_norm``), which equals bytes for Latin scripts
    and differs slightly for multi-byte scripts.
  - The instruct-model prompt's fixed wording ("Answer with A or B.") is in English (the paper's exact
    template wording per language is not published); the goal and options are in the target language.
  - Uses the *non-parallel* set (natively authored). Point ``HF_REPO`` at
    ``mrlbenchmarks/global-piqa-parallel`` for the translated, cross-lingually comparable set.
"""

import re
import unicodedata

import numpy as np

from lighteval.metrics.metrics import Metrics
from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.metrics.normalizations import LogProbCharNorm
from lighteval.metrics.utils.metric_utils import SampleLevelMetric, SamplingMethod
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


HF_REPO = "mrlbenchmarks/global-piqa-nonparallel"

# Short code (task-name suffix) -> Global-PIQA config. Extend with any of the ~130 configs shipped.
LANGUAGES = {
    "fr": "fra_latn_fran",
    "fr_ca": "fra_latn_cana",
    "en": "eng_latn",
    "de": "deu_latn",
    "es": "spa_latn_spai",
    "it": "ita_latn",
    "pt": "por_latn_port",
    "pt_br": "por_latn_braz",
    "nl": "nld_latn",
    "ru": "rus_cyrl",
    "zh": "cmn_hans",
    "zh_hant": "cmn_hant",
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


# --------------------------------------------------------------------------------------------------
# Base-model variant: log-probability of each solution given the goal (paper's base-model protocol)
# --------------------------------------------------------------------------------------------------
def piqa_loglikelihood_prompt_fn(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=line["prompt"].strip(),
        choices=[f" {line['solution0'].strip()}", f" {line['solution1'].strip()}"],
        gold_index=int(line["label"]),
    )


# --------------------------------------------------------------------------------------------------
# Instruct-model variant: show A/B options, generate, score by string matching (paper's IT protocol)
# --------------------------------------------------------------------------------------------------
def piqa_generative_prompt_fn(line, task_name: str = None):
    sol0, sol1 = line["solution0"].strip(), line["solution1"].strip()
    query = f"{line['prompt'].strip()}\n\nA. {sol0}\nB. {sol1}\n\nAnswer with A or B."
    return Doc(
        task_name=task_name,
        query=query,
        choices=[sol0, sol1],
        gold_index=int(line["label"]),
    )


def _norm(text) -> str:
    text = unicodedata.normalize("NFD", str(text))
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"[^\w\s]", " ", text.lower(), flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


class PiqaGenerativeMatch(SampleLevelComputation):
    """String-match the generated answer against the gold option (by A/B letter, else by solution text)."""

    def compute(self, model_response, doc, **kwargs) -> float:
        text = ""
        for attr in ("final_text", "text"):
            seq = getattr(model_response, attr, None)
            if seq:
                text = seq[0]
                break
        label = doc.gold_index if isinstance(doc.gold_index, int) else doc.gold_index[0]
        gold_letter = "A" if label == 0 else "B"
        # 1) an explicit A/B choice (take the last standalone letter = the model's conclusion)
        letters = re.findall(r"(?<![A-Za-z])([ABab])(?![A-Za-z])", text)
        if letters:
            return 1.0 if letters[-1].upper() == gold_letter else 0.0
        # 2) fallback: the gold solution text is present and the distractor is not
        pred = _norm(text)
        gold, other = _norm(doc.choices[label]), _norm(doc.choices[1 - label])
        if gold and gold in pred and not (other and other in pred):
            return 1.0
        return 0.0


piqa_generative_metric = SampleLevelMetric(
    metric_name="piqa_gen_acc",
    sample_level_fn=PiqaGenerativeMatch(),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)


def _make_loglikelihood_task(code, hf_subset):
    return LightevalTaskConfig(
        name=f"piqa_{code}",
        suite=["community"],
        prompt_function=piqa_loglikelihood_prompt_fn,
        hf_repo=HF_REPO,
        hf_subset=hf_subset,
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=-1,
        metrics=[
            Metrics.loglikelihood_acc,
            # NOTE: the paper normalizes by BYTE length; lighteval only offers character-length
            # normalization. This is exact for Latin scripts but a byte-length LogProbNormalization
            # should be implemented (and used here) for non-Latin languages (CJK, Arabic, Cyrillic, ...).
            Metrics.loglikelihood_acc(
                sample_params={"logprob_normalization": LogProbCharNorm(ignore_first_space=True)}
            ),
        ],
        stop_sequence=["\n"],
        version=0,
    )


def _make_generative_task(code, hf_subset):
    return LightevalTaskConfig(
        name=f"piqa_{code}_gen",
        suite=["community"],
        prompt_function=piqa_generative_prompt_fn,
        hf_repo=HF_REPO,
        hf_subset=hf_subset,
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=32,
        metrics=[piqa_generative_metric],
        stop_sequence=["\n"],
        version=0,
    )


TASKS_TABLE = [_make_loglikelihood_task(code, hf_subset) for code, hf_subset in LANGUAGES.items()] + [
    _make_generative_task(code, hf_subset) for code, hf_subset in LANGUAGES.items()
]

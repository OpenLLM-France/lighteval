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
CARTE (Culturally Anchored Regional-Territorial Evaluation)

dataset:
ScarAlcar/CARTE

abstract:
CARTE is a fully French multiple-choice benchmark (2,431 questions) evaluating region-specific and
intra-national cultural knowledge across the 13 metropolitan regions of France and 14 thematic
domains (culture, language, demographics, economy, environment, mobility, ...). CARTE-LV is a
233-question subset (flag ``is_carte_lv``) targeting regional linguistic variation.

languages:
french

tags:
culture, france, regional, question-answering, multiple-choice, abstention

paper:
https://arxiv.org/abs/2606.01995

--------------------------------------------------------------------------------------------------
How to run -- task strings are ``community|<name>|<num_fewshot>`` (pass ``--custom-tasks <this file>``):

  Benchmark prefix:  carte      -> full CARTE (2421 questions)
                     carte_lv   -> CARTE-LV linguistic-variation subset (223 questions; run 0-shot)

  Formulation:       _mcf        -> multiple-choice, answer LETTERS scored by log-likelihood
                                    (this is the paper's setting -- read the `acc` metric)
                     _cf         -> cloze: answer TEXTS scored by log-likelihood
                     _hybrid     -> hybrid of the two
                     _generative -> free generation of a letter (with reasoning); extractive-match
                                    `acc` (pass@1) + macro-F1 over the A-E classes

  Prompt suffix:     (none)      -> no system prompt  [DEFAULT, closest to the paper]
                     _frprompt   -> French "French-knowledge expert" system prompt
                     _enprompt   -> the same in English

  >>> Reproduce the paper (accuracy):  community|carte_mcf|0   (also |1 and |3)
                                        community|carte_lv_mcf|0
      then read the `acc` metric.

  All task names (each runnable at |0, |1 or |3):
      full CARTE                       CARTE-LV
      ----------                       --------
      carte_mcf                        carte_lv_mcf
      carte_cf                         carte_lv_cf
      carte_hybrid                     carte_lv_hybrid
      carte_generative                 carte_lv_generative
      carte_mcf_frprompt               carte_lv_mcf_frprompt
      carte_cf_frprompt                carte_lv_cf_frprompt
      carte_hybrid_frprompt            carte_lv_hybrid_frprompt
      carte_generative_frprompt        carte_lv_generative_frprompt
      carte_mcf_enprompt               carte_lv_mcf_enprompt
      carte_cf_enprompt                carte_lv_cf_enprompt
      carte_hybrid_enprompt            carte_lv_hybrid_enprompt
      carte_generative_enprompt        carte_lv_generative_enprompt

  Metrics: `acc` is the headline (paper). Extras (our additions): abstention_reward, reliability,
  selective_acc, coverage, abstention_rate, pick_A..pick_E (answer/position bias), and macro_f1
  (generative only). Region / Topic / is_carte_lv are stored per sample in ``doc.specific`` for
  offline breakdowns.

--------------------------------------------------------------------------------------------------
Dataset shape (from the HF card):
  - FR_Question:    the question (French).
  - Options:        dict {A,B,C,D,E}, each value carries its own letter prefix, e.g. "A. Nantes".
  - Correct_Answer: the gold letter ("A".."E").
  - Topic:          one of the 14 thematic domains.
  - Region:         one of the 13 metropolitan regions.
  - is_carte_lv:    True for the CARTE-LV linguistic subset.
  - Explanation:    rationale (not used for scoring).
Each item has exactly 5 options = 1 correct + 3 distractors + 1 "Je ne sais pas" ("I don't know").
The correct answer AND the "Je ne sais pas" option are shuffled across positions A-E (verified in
the data), so the abstention option is detected *dynamically* per question, not assumed at "E".

--------------------------------------------------------------------------------------------------
What the paper does (Almeida Carneiro et al., 2026), for reference:
  - Primary metric: plain **accuracy** (proportion answered correctly).
  - Protocol: **0-shot, 1-shot and 3-shot**, in-context examples drawn uniformly at random
    (stratified by region) from the dataset itself, avoiding the evaluated question.
  - Breakdowns reported by difficulty (Easy/Med/Hard), by Region and by Topic; CARTE vs CARTE-LV.
  - The "Je ne sais pas" option is described only as a *valid response that "reduces the penalty
    for uncertainty" and discourages forced guessing* -- the paper gives it NO partial credit; in
    its accuracy metric, choosing it simply counts as not-correct (i.e. 0, like a wrong answer).

Abstention scoring here is therefore an **extension** beyond the paper, motivated by the benchmark's
stated intent. State of the art for scoring an explicit "I don't know" option:
  - Selective prediction / risk-coverage (El-Yaniv & Wiener 2010; Geifman & El-Yaniv 2017):
    report accuracy on the *answered* subset (selective accuracy) together with coverage.
  - Formula scoring / negative marking (standardized tests): correct +1, wrong -1/(k-1), blank 0.
  - Effective reliability (Whitehead et al. 2022): correct +1, wrong -1, abstain 0.
  - Confidence-threshold proper scoring (OpenAI, "Why Language Models Hallucinate", 2025): answer
    only if confident; correct +1, wrong -t/(1-t), IDK 0 (the wrong-penalty encodes a threshold t).
  - Bounded partial credit: correct 1, IDK a small positive c, wrong 0 (interpretable in [0,1]).
We report several of these so an abstaining model is penalized less than a wrong one:
  - abstention_rate, coverage        -- descriptive (how often the model opts out / answers).
  - selective_acc                    -- accuracy among *answered* questions (selective prediction).
  - abstention_reward                -- bounded partial credit: 1 / IDK_PARTIAL_REWARD / 0.
  - reliability                      -- effective reliability: +1 correct, 0 abstain, -1 wrong.
"""

import re

import numpy as np

from lighteval.metrics.dynamic_metrics import (
    LogLikelihoodAccMetric,
    MultilingualExtractiveMatchMetric,
)
from lighteval.metrics.metrics_corpus import CorpusLevelF1Score
from lighteval.metrics.metrics_sample import PassAtK, SampleLevelComputation
from lighteval.metrics.normalizations import LogProbCharNorm, LogProbTokenNorm, normalize_log_probs
from lighteval.metrics.sample_preparator import GenerativeCorpusMetricInput, Preparator
from lighteval.metrics.utils.extractive_match_utils import IndicesExtractionConfig
from lighteval.metrics.utils.metric_utils import (
    CorpusLevelMetric,
    SampleLevelMetric,
    SampleLevelMetricGrouping,
    SamplingMethod,
)
from lighteval.tasks.default_prompts import LETTER_INDICES
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
from lighteval.utils.utils import as_list


HF_REPO = "ScarAlcar/CARTE"
IDK_TEXT = "je ne sais pas"
IDK_PARTIAL_REWARD = 0.25  # bounded partial credit for choosing "Je ne sais pas" (see docstring)

FORMULATIONS = [MCFFormulation(), CFFormulation(), HybridFormulation()]

# System-prompt variants (mirrors community_tasks/mathalea.py). "noprompt" is the default (no added
# system prompt, closest to the paper's evaluation) and carries no name suffix.
PROMPT_CONFIGS = {
    "noprompt": None,
    "frprompt": ("Vous êtes un expert des connaissances culturelles, géographiques et régionales de la France.\n\n"),
    "enprompt": ("You are an expert in the cultural, geographic and regional knowledge of France.\n\n"),
}


def _prompt_suffix(prompt_key: str) -> str:
    """No suffix for the default ('noprompt'); '_frprompt'/'_enprompt' for the prompted variants."""
    return "" if prompt_key == "noprompt" else f"_{prompt_key}"


# (benchmark name, hf_filter): the full CARTE and the standalone CARTE-LV linguistic-variation subset.
BENCHMARKS = [
    ("carte", None),
    ("carte_lv", lambda row: bool(row.get("is_carte_lv"))),
]


# --------------------------------------------------------------------------------------------------
# Row parsing
# --------------------------------------------------------------------------------------------------
def _clean_option(text: str) -> str:
    """Strip the leading 'X. ' / 'X) ' letter prefix baked into each option string."""
    return re.sub(r"^\s*[A-Za-z]\s*[.)]\s*", "", str(text)).strip()


def _parse_line(line) -> dict:
    """Return {question, choices (clean, ordered A..E), gold_idx, idk_idx}."""
    options = line["Options"]
    letters = sorted(options.keys())  # ['A','B','C','D','E']
    choices = [_clean_option(options[k]) for k in letters]
    correct = str(line["Correct_Answer"]).strip()
    gold_idx = letters.index(correct) if correct in letters else 0
    idk_idx = next(
        (i for i, c in enumerate(choices) if c.strip().lower().rstrip(".") == IDK_TEXT),
        None,
    )
    return {"question": str(line["FR_Question"]).strip(), "choices": choices, "gold_idx": gold_idx, "idk_idx": idk_idx}


# --------------------------------------------------------------------------------------------------
# Abstention-aware scoring (shared by the logprob and generative flavors)
#
# Also reports a per-letter pick rate ("pick_A".."pick_E"): the fraction of questions for which the
# model chose each position. The paper (App. E) flags that some models favor specific options/
# positions; since the correct answer and the "Je ne sais pas" option are shuffled across A-E, a
# skewed pick distribution exposes that label/position bias (distinct from abstention_rate, which is
# the content-level rate of choosing "Je ne sais pas" wherever it sits).
# --------------------------------------------------------------------------------------------------
_PICK_LETTERS = list("ABCDE")  # CARTE items always have exactly 5 options A-E
_PICK_NAMES = [f"pick_{letter}" for letter in _PICK_LETTERS]

_ABSTENTION_NAMES = ["abstention_rate", "coverage", "selective_acc", "abstention_reward", "reliability"] + _PICK_NAMES
_ABSTENTION_CORPUS = {
    "abstention_rate": np.mean,
    "coverage": np.mean,
    "selective_acc": np.nanmean,  # mean over *answered* questions only (NaN = abstained/no answer)
    "abstention_reward": np.mean,
    "reliability": np.mean,
    **dict.fromkeys(_PICK_NAMES, np.mean),
}
_ABSTENTION_HIGHER = {
    "abstention_rate": False,
    "coverage": True,
    "selective_acc": True,
    "abstention_reward": True,
    "reliability": True,
    **dict.fromkeys(_PICK_NAMES, False),  # diagnostics, not optimization targets
}


def _pick_scores(pred_idx) -> dict:
    """One-hot per-letter pick indicator (all zeros if no answer was parsed)."""
    return {name: (1.0 if pred_idx == i else 0.0) for i, name in enumerate(_PICK_NAMES)}


def _abstention_scores(pred_idx, gold_idx, idk_idx, idk_reward) -> dict:
    """Turn a predicted choice index into the abstention-aware metric dict.

    ``pred_idx is None`` means no parsable answer (generative): counted as answered-and-wrong, not
    as an abstention (only the explicit "Je ne sais pas" option counts as abstaining).
    """
    if pred_idx is None:
        return {
            "abstention_rate": 0.0,
            "coverage": 1.0,
            "selective_acc": 0.0,
            "abstention_reward": 0.0,
            "reliability": -1.0,
            **_pick_scores(None),
        }
    correct = pred_idx == gold_idx
    abstained = idk_idx is not None and pred_idx == idk_idx
    answered = not abstained
    return {
        "abstention_rate": float(abstained),
        "coverage": float(answered),
        "selective_acc": float(correct) if answered else float("nan"),
        "abstention_reward": 1.0 if correct else (idk_reward if abstained else 0.0),
        "reliability": 1.0 if correct else (0.0 if abstained else -1.0),
        **_pick_scores(pred_idx),
    }


def _argmax_logprob_choice(doc: Doc, model_response) -> int:
    """Predicted choice index = argmax of char-length-normalized choice logprobs.

    Char normalization keeps the short "Je ne sais pas" option from being unfairly favored over long
    distractors. For the MCF (single-letter) formulation all choices have length 1, so this reduces
    to the raw argmax used by the headline accuracy.
    """
    n = len(doc.choices)
    choices_logprobs = model_response.logprobs[:n]
    unconditioned = model_response.logprobs[n : 2 * n] if len(model_response.logprobs) == 2 * n else None
    choices_tokens = model_response.output_tokens[:n] if model_response.output_tokens else None
    normalized = normalize_log_probs(LogProbCharNorm(), choices_logprobs, unconditioned, doc.choices, choices_tokens)
    return int(np.argmax(normalized))


class CarteAbstentionLogprob(SampleLevelComputation):
    """Abstention-aware scoring for the logprob (CF/MCF/Hybrid) MCQ tasks."""

    def __init__(self, idk_reward: float = IDK_PARTIAL_REWARD):
        self.idk_reward = idk_reward

    def compute(self, doc: Doc, model_response, **kwargs) -> dict:
        pred = _argmax_logprob_choice(doc, model_response)
        gold = as_list(doc.gold_index)[0]
        idk = (doc.specific or {}).get("idk_index")
        return _abstention_scores(pred, gold, idk, self.idk_reward)


def _extract_pred_letter(text: str, n_choices: int, idk_idx):
    """Recover the answer letter index from a generative output, else None."""
    if not text:
        return None
    valid = "".join(LETTER_INDICES[:n_choices])
    low = text.lower()
    letter_re = rf"\b([{valid}{valid.lower()}])\b"
    for cue in (
        "réponse finale",
        "la réponse est",
        "réponse :",
        "réponse:",
        "réponse est",
        "the answer is",
        "answer:",
    ):
        i = low.rfind(cue)
        if i != -1:
            m = re.search(letter_re, text[i + len(cue) :])
            if m:
                return valid.index(m.group(1).upper())
    if idk_idx is not None and IDK_TEXT in low:  # wrote out "je ne sais pas" instead of its letter
        return idk_idx
    matches = re.findall(letter_re, text)  # fallback: last standalone valid letter
    return valid.index(matches[-1].upper()) if matches else None


class CarteAbstentionGenerative(SampleLevelComputation):
    """Abstention-aware scoring for the generative MCQ task."""

    def __init__(self, idk_reward: float = IDK_PARTIAL_REWARD):
        self.idk_reward = idk_reward

    def compute(self, doc: Doc, model_response, **kwargs) -> dict:
        text = _response_text(model_response)
        idk = (doc.specific or {}).get("idk_index")
        pred = _extract_pred_letter(text, len(doc.choices), idk)
        gold = as_list(doc.gold_index)[0]
        return _abstention_scores(pred, gold, idk, self.idk_reward)


def _response_text(model_response) -> str:
    for attr in ("final_text", "text"):
        seq = getattr(model_response, attr, None)
        if seq:
            return seq[0]
    return ""


def _abstention_metric(sample_fn, category):
    return SampleLevelMetricGrouping(
        metric_name=_ABSTENTION_NAMES,
        sample_level_fn=sample_fn,
        category=category,
        corpus_level_fn=_ABSTENTION_CORPUS,
        higher_is_better=_ABSTENTION_HIGHER,
    )


# --------------------------------------------------------------------------------------------------
# Generative macro-F1 over the answer letters (an extra lens, robust to class imbalance)
# --------------------------------------------------------------------------------------------------
class CarteLetterPreparator(Preparator):
    """Emit the (gold_letter, predicted_letter) pair for corpus-level F1 over the A..E classes."""

    def prepare(self, doc: Doc, model_response, **kwargs) -> GenerativeCorpusMetricInput:
        letters = LETTER_INDICES[: len(doc.choices)]
        idk = (doc.specific or {}).get("idk_index")
        pred = _extract_pred_letter(_response_text(model_response), len(doc.choices), idk)
        gold_letter = letters[as_list(doc.gold_index)[0]]
        pred_letter = letters[pred] if pred is not None else "?"
        return GenerativeCorpusMetricInput(golds=[gold_letter], preds=[pred_letter])


carte_macro_f1_metric = CorpusLevelMetric(
    metric_name="macro_f1",
    sample_level_fn=CarteLetterPreparator(),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=CorpusLevelF1Score(average="macro"),
    higher_is_better=True,
)


# --------------------------------------------------------------------------------------------------
# Generative accuracy (extractive match on the answer letter), mirrors mathalea.py
# --------------------------------------------------------------------------------------------------
carte_generative_acc = SampleLevelMetric(
    metric_name="carte_pass@1",
    sample_level_fn=PassAtK(
        sample_scoring_function=MultilingualExtractiveMatchMetric(
            language=Language.FRENCH,
            gold_extraction_target=[
                IndicesExtractionConfig(prefix_for_extraction="NativeLetters", try_extract_without_anchor=True)
            ],
            pred_extraction_target=[
                IndicesExtractionConfig(prefix_for_extraction="NativeLetters", try_extract_without_anchor=True)
            ],
            precision=6,
        ),
        k=1,
    ),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)


# --------------------------------------------------------------------------------------------------
# Prompt builders
# --------------------------------------------------------------------------------------------------
def _doc_specific(line, idk_idx) -> dict:
    """Per-sample metadata stashed on ``doc.specific``: the abstention (IDK) index plus the row's
    Region / Topic / is_carte_lv, so results can be broken down along those axes from the details."""
    return {
        "idk_index": idk_idx,
        "region": line.get("Region"),
        "topic": line.get("Topic"),
        "is_carte_lv": bool(line.get("is_carte_lv")),
    }


def _attach_idk(prompt_fn):
    """Wrap a template prompt function so the IDK index and Region/Topic metadata ride along in
    ``doc.specific``.

    The MCQ templates keep the choice order, so the index found on the raw row is valid for the Doc.
    """

    def wrapped(line, task_name: str = None):
        doc = prompt_fn(line, task_name)
        if isinstance(doc, Doc):
            doc.specific = {**(doc.specific or {}), **_doc_specific(line, _parse_line(line)["idk_idx"])}
        return doc

    return wrapped


def _mcq_prompt(formulation, instruction):
    def adapter(line):
        parsed = _parse_line(line)
        out = {"question": parsed["question"], "choices": parsed["choices"], "gold_idx": parsed["gold_idx"]}
        if instruction:
            out["instruction"] = instruction
        return out

    return _attach_idk(get_mcq_prompt_function(Language.FRENCH, adapter, formulation=formulation))


def _generative_prompt(instruction):
    prefix = instruction or ""

    def prompt_fn(line, task_name: str = None):
        parsed = _parse_line(line)
        choices = parsed["choices"]
        valid_letters = "".join(LETTER_INDICES[: len(choices)])
        task_instruction = (
            "Répondez à la question à choix multiple suivante. L'une des options est « Je ne sais "
            "pas » : choisissez-la si vous n'êtes pas sûr plutôt que de deviner. La dernière ligne "
            "de votre réponse doit être au format suivant : 'Réponse : $LETTER' (sans les "
            f"guillemets) où LETTER est l'une des lettres {valid_letters}. Réfléchissez étape par "
            "étape avant de répondre."
        )
        choices_str = "\n".join(f"{letter}) {choice}" for letter, choice in zip(LETTER_INDICES, choices))
        query = f"{prefix}{task_instruction}\n\n{parsed['question']}\n\n{choices_str}"
        return Doc(
            task_name=task_name,
            query=query,
            choices=LETTER_INDICES[: len(choices)],
            gold_index=parsed["gold_idx"],
            instruction=prefix + task_instruction,
            specific=_doc_specific(line, parsed["idk_idx"]),
        )

    return prompt_fn


# --------------------------------------------------------------------------------------------------
# Task factories
#   Only a single 'default' config with a 'train' split exists, so few-shot examples are drawn from
#   the same split (the sampler excludes the evaluated question). Run at |0, |1 or |3 as in the paper.
# --------------------------------------------------------------------------------------------------
def _make_mcq_task(benchmark, formulation, prompt_key, hf_filter):
    instruction = PROMPT_CONFIGS[prompt_key]
    metrics = get_metrics_for_formulation(
        formulation,
        [
            LogLikelihoodAccMetric(normalization=LogProbTokenNorm()),
            LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
        ],
    ) + [_abstention_metric(CarteAbstentionLogprob(), SamplingMethod.LOGPROBS)]
    return LightevalTaskConfig(
        name=f"{benchmark}_{formulation.name.lower()}{_prompt_suffix(prompt_key)}",
        prompt_function=_mcq_prompt(formulation, instruction),
        suite=["community"],
        hf_repo=HF_REPO,
        hf_subset="default",
        hf_filter=hf_filter,
        hf_avail_splits=["train"],
        evaluation_splits=["train"],
        few_shots_split="train",
        few_shots_select="random_sampling",
        generation_size=-1,
        metrics=metrics,
        stop_sequence=["\n"],
        version=0,
    )


def _make_generative_task(benchmark, prompt_key, hf_filter):
    instruction = PROMPT_CONFIGS[prompt_key]
    return LightevalTaskConfig(
        name=f"{benchmark}_generative{_prompt_suffix(prompt_key)}",
        prompt_function=_generative_prompt(instruction),
        suite=["community"],
        hf_repo=HF_REPO,
        hf_subset="default",
        hf_filter=hf_filter,
        hf_avail_splits=["train"],
        evaluation_splits=["train"],
        few_shots_split="train",
        few_shots_select="random_sampling",
        generation_size=4096,
        metrics=[
            carte_generative_acc,
            _abstention_metric(CarteAbstentionGenerative(), SamplingMethod.GENERATIVE),
            carte_macro_f1_metric,
        ],
        stop_sequence=[],
        version=0,
    )


TASKS_TABLE = [
    _make_mcq_task(benchmark, formulation, prompt_key, hf_filter)
    for benchmark, hf_filter in BENCHMARKS
    for formulation in FORMULATIONS
    for prompt_key in PROMPT_CONFIGS
] + [
    _make_generative_task(benchmark, prompt_key, hf_filter)
    for benchmark, hf_filter in BENCHMARKS
    for prompt_key in PROMPT_CONFIGS
]

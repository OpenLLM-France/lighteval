"""
name:
Exo7

dataset:
OpenLLM-BPI/Exo7MCQ

abstract:
Exo7 is a dataset of multi-label multiple-choice math questions for French undergraduate
students, sourced from http://exo7.emath.fr/. Many items have more than one correct answer.
Two scoring paths are exposed, both zero-shot: a logprob path (MCF, Hybrid) using a
TruthfulQA MC2-style probability-mass metric, and a generative path that asks the model to
emit "Réponse : A, C" and scores with set-F1 and exact-set-match.

languages:
french

tags:
math, question-answering, multiple-choice, multi-label

paper:

"""

import re

import numpy as np

from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.metrics.utils.metric_utils import SampleLevelMetric
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod
from lighteval.tasks.templates.multichoice import get_mcq_prompt_function
from lighteval.tasks.templates.utils.formulation import (
    HybridFormulation,
    MCFFormulation,
)
from lighteval.utils.language import Language


# --- Custom logprob mass metric ---


class Exo7MCMetric(SampleLevelComputation):
    """Probability mass metric for multi-label multiple choice.

    Converts log-likelihoods to probabilities, normalizes them, and returns
    the total probability mass on the correct answers.
    """

    def compute(self, model_response: ModelResponse, doc: Doc, **kwargs):
        logprobs = model_response.logprobs
        probs = np.exp(np.array(logprobs))
        probs_norm = probs / np.sum(probs)

        labels = np.array(doc.specific["labels"])
        return float(np.sum(probs_norm[labels == 1]))


exo7_mc_metric = SampleLevelMetric(
    metric_name="acc",
    sample_level_fn=Exo7MCMetric(),
    category=SamplingMethod.LOGPROBS,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)


# --- Generative metrics (multi-letter answer) ---


_RESPONSE_RE = re.compile(r"(?:^|\n)\s*[Rr][ée]ponse\s*:?\s*([^\n]*)")
_LETTER_RE = re.compile(r"\b[A-Z]\b")


def _extract_letters(text: str, valid: set) -> set:
    """Extract the set of answer letters from a generative response.

    Prefers the last line starting with "Réponse :" (the instructed format);
    otherwise falls back to the last non-empty line. Keeps only letters in
    the valid set for this question. Uses word boundaries so isolated
    capitals (e.g. "A, C") match but letters inside words ("Aucune", "Vrai")
    do not.
    """
    if not text:
        return set()
    matches = list(_RESPONSE_RE.finditer(text))
    if matches:
        target = matches[-1].group(1)
    else:
        lines = [line for line in text.strip().splitlines() if line.strip()]
        target = lines[-1] if lines else ""
    return {c for c in _LETTER_RE.findall(target) if c in valid}


class Exo7GenerativeF1(SampleLevelComputation):
    """Set-F1 between predicted and gold letter sets."""

    def compute(self, model_response: ModelResponse, doc: Doc, **kwargs):
        pred_text = model_response.text[0] if model_response.text else ""
        valid = set(doc.choices)
        gold = set(doc.specific["correct_letters"])
        pred = _extract_letters(pred_text, valid)
        if not gold and not pred:
            return 1.0
        if not gold or not pred:
            return 0.0
        tp = len(pred & gold)
        if tp == 0:
            return 0.0
        precision = tp / len(pred)
        recall = tp / len(gold)
        return 2 * precision * recall / (precision + recall)


class Exo7GenerativeExactMatch(SampleLevelComputation):
    """1.0 iff the predicted letter set exactly matches the gold set."""

    def compute(self, model_response: ModelResponse, doc: Doc, **kwargs):
        pred_text = model_response.text[0] if model_response.text else ""
        valid = set(doc.choices)
        gold = set(doc.specific["correct_letters"])
        pred = _extract_letters(pred_text, valid)
        return float(pred == gold)


exo7_generative_f1_metric = SampleLevelMetric(
    metric_name="f1",
    sample_level_fn=Exo7GenerativeF1(),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)

exo7_generative_exact_metric = SampleLevelMetric(
    metric_name="exact_match",
    sample_level_fn=Exo7GenerativeExactMatch(),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)


# --- Prompt function ---

INSTRUCTION = (
    "Pour la question suivante, une ou plusieurs propositions peuvent être correctes. "
    "Évaluez chaque proposition."
)


def _make_prompt_fn(formulation):
    base_fn = get_mcq_prompt_function(
        Language.FRENCH,
        lambda line: {
            "question": line["question"],
            "choices": line["targets"]["choices"],
            "gold_idx": [i for i, label in enumerate(line["targets"]["labels"]) if label == 1],
            "instruction": INSTRUCTION,
        },
        formulation=formulation,
    )

    def prompt_fn(line, task_name: str = None):
        doc = base_fn(line, task_name)
        doc.specific = {"labels": line["targets"]["labels"]}
        return doc

    return prompt_fn


GENERATIVE_INSTRUCTION_TEMPLATE = (
    "Pour la question suivante, une ou plusieurs propositions peuvent être correctes. "
    "Évaluez chaque proposition, puis indiquez toutes les lettres des propositions correctes. "
    "La dernière ligne de votre réponse doit être au format suivant : "
    "'Réponse : $LETTRES' (sans les guillemets) où $LETTRES est une liste de lettres parmi "
    "{valid_letters} séparées par des virgules (par exemple 'Réponse : A, C'). "
    "Réfléchissez étape par étape avant de répondre."
)


def _make_generative_prompt_fn():
    def prompt_fn(line, task_name: str = None):
        choices = line["targets"]["choices"]
        labels = line["targets"]["labels"]
        letters = list(LETTER_INDICES[: len(choices)])
        correct_letters = [letters[i] for i, label in enumerate(labels) if label == 1]

        instruction = GENERATIVE_INSTRUCTION_TEMPLATE.format(valid_letters=", ".join(letters))
        choices_str = "\n".join(
            f"{letter}) {choice.strip()}" for letter, choice in zip(letters, choices)
        )
        query = f"{instruction}\n\n{line['question'].strip()}\n\n{choices_str}"

        doc = Doc(
            task_name=task_name,
            query=query,
            choices=letters,
            gold_index=[i for i, label in enumerate(labels) if label == 1],
            instruction=instruction,
        )
        doc.specific = {
            "correct_letters": correct_letters,
            "labels": labels,
        }
        return doc

    return prompt_fn


# --- Task configs ---

FORMULATIONS = [MCFFormulation(), HybridFormulation()]


def _make_task(formulation):
    return LightevalTaskConfig(
        name=f"exo7_{formulation.name.lower()}",
        prompt_function=_make_prompt_fn(formulation),
        suite=["community"],
        hf_repo="OpenLLM-BPI/Exo7MCQ",
        hf_subset="default",
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=1,
        metrics=[exo7_mc_metric],
        stop_sequence=["\n"],
        version=0,
    )


def _make_generative_task():
    return LightevalTaskConfig(
        name="exo7_generative",
        prompt_function=_make_generative_prompt_fn(),
        suite=["community"],
        hf_repo="OpenLLM-BPI/Exo7MCQ",
        hf_subset="default",
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=16384,
        metrics=[exo7_generative_f1_metric, exo7_generative_exact_metric],
        stop_sequence=[],
        version=0,
    )


TASKS_TABLE = [_make_task(formulation) for formulation in FORMULATIONS] + [_make_generative_task()]

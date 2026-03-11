"""
name:
MathAlea

dataset:
OpenLLM-BPI/MathAleaMCQ

abstract:
MathAlea is a dataset of multiple-choice math questions for French middle and high school students.
It covers a range of topics and difficulty levels, making it a valuable resource for evaluating the
mathematical reasoning capabilities of language models in the context of education.

languages:
french

tags:
math, question-answering, multiple-choice

paper:

"""

import unicodedata

from lighteval.metrics.dynamic_metrics import LogLikelihoodAccMetric
from lighteval.metrics.normalizations import LogProbCharNorm, LogProbTokenNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.multilingual.utils.task_utils import get_metrics_for_formulation
from lighteval.tasks.templates.multichoice import get_mcq_prompt_function
from lighteval.tasks.templates.utils.formulation import (
    CFFormulation,
    HybridFormulation,
    MCFFormulation,
)
from lighteval.utils.language import Language


GRADE_LEVELS = ["cinquième", "quatrième", "troisième", "première", "terminale"]


def remove_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")

FORMULATIONS = [MCFFormulation(), CFFormulation(), HybridFormulation()]


def format_choice(choice):
    if isinstance(choice, str):
        if choice.endswith("\qquad"):
            choice = choice[:-6].strip()
        return choice.strip()
    if isinstance(choice, list):
        return [format_choice(c) for c in choice]
    raise ValueError(f"Unsupported choice type: {type(choice)}")

def format_question(question):
    return question.replace("\\", "\n").strip()


def _make_tasks(subset, alias, formulation):
    return LightevalTaskConfig(
        name=f"mathalea_{formulation.name.lower()}:{alias}",
        prompt_function=get_mcq_prompt_function(
            Language.FRENCH,
            lambda line: {
                "question": format_question(line["question"]),
                "choices": format_choice(line["choices"]),
                "gold_idx": int(line["answerKey"]),
            },
            formulation=formulation,
        ),
        suite=["community"],
        hf_repo="OpenLLM-BPI/MathAleaMCQ",
        hf_subset=subset,
        hf_avail_splits=["dev", "test"],
        evaluation_splits=["test"],
        few_shots_split="dev",
        few_shots_select="sequential",
        generation_size=-1,
        metrics=get_metrics_for_formulation(
            formulation,
            [
                LogLikelihoodAccMetric(normalization=LogProbTokenNorm()),
                LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
            ],
        ),
        stop_sequence=["\n"],
        version=0,
    )


TASKS_TABLE = [
    _make_tasks(subset, remove_accents(subset), formulation)
    for subset in ["all"] + GRADE_LEVELS
    for formulation in FORMULATIONS
]

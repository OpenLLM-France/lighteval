"""
name:
MathAlea

dataset:
OpenLLM-France/MathAleaMCQ

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

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


GRADE_LEVELS = {
    "cinquième": "cinquieme",
    "quatrième": "quatrieme",
    "troisième": "troisieme",
    "première": "premiere",
    "terminale": "terminale",
}


def prompt_mathalea(line, task_name: str = None):
    """Build a multiple-choice prompt from a MathAlea dataset line."""
    choices = line["choices"]
    query = f"{line['question'].strip()}\n"
    query += "".join(
        f"{letter}. {choice}\n"
        for letter, choice in zip(LETTER_INDICES, choices)
    )
    query += "Réponse :"

    gold_index = int(line["answerKey"])

    return Doc(
        task_name=task_name,
        query=query,
        choices=[f" {LETTER_INDICES[i]}" for i in range(len(choices))],
        gold_index=gold_index,
    )


TASKS_TABLE = [
    # Combined task: all grade levels at once
    LightevalTaskConfig(
        name="mathalea:all",
        prompt_function=prompt_mathalea,
        suite=["community"],
        hf_repo="OpenLLM-BPI/MathAleaMCQ",
        hf_subset="all",
        hf_avail_splits=["dev", "test"],
        evaluation_splits=["test"],
        few_shots_split="dev",
        few_shots_select="sequential",
        generation_size=1,
        metrics=[Metrics.loglikelihood_acc],
        stop_sequence=["\n"],
        version=0,
    ),
] + [
    # Per-grade tasks
    LightevalTaskConfig(
        name=f"mathalea:{alias}",
        prompt_function=prompt_mathalea,
        suite=["community"],
        hf_repo="OpenLLM-BPI/MathAleaMCQ",
        hf_subset=subset,
        hf_avail_splits=["dev", "test"],
        evaluation_splits=["test"],
        few_shots_split="dev",
        few_shots_select="sequential",
        generation_size=1,
        metrics=[Metrics.loglikelihood_acc],
        stop_sequence=["\n"],
        version=0,
    )
    for subset, alias in GRADE_LEVELS.items()
]

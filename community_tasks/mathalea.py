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

import numpy as np

from lighteval.metrics.dynamic_metrics import LogLikelihoodAccMetric, MultilingualExtractiveMatchMetric
from lighteval.metrics.metrics_sample import PassAtK
from lighteval.metrics.normalizations import LogProbCharNorm, LogProbTokenNorm
from lighteval.metrics.utils.extractive_match_utils import IndicesExtractionConfig
from lighteval.metrics.utils.metric_utils import SampleLevelMetric, SamplingMethod
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


GRADE_LEVELS = ["cinquième", "quatrième", "troisième", "première", "terminale"]


def remove_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


FORMULATIONS = [MCFFormulation(), CFFormulation(), HybridFormulation()]


PROMPT_CONFIGS = {
    "frprompt": {
        "all": "Vous êtes un assistant mathématique pour les élèves du secondaire français.\n\n",
        "grade": "Vous êtes un assistant mathématique pour les élèves de {subset}.\n\n",
    },
    "enprompt": {
        "all": "You are a helpful math assistant for French secondary school students.\n\n",
        "grade": "You are a helpful math assistant for French students in grade {subset}.\n\n",
    },
    "noprompt": None,
}


def _get_instruction(prompt_key, subset):
    prompt_cfg = PROMPT_CONFIGS[prompt_key]
    if prompt_cfg is None:
        return None
    if subset == "all":
        return prompt_cfg["all"]
    return prompt_cfg["grade"].format(subset=subset)


mathalea_generative_metric = SampleLevelMetric(
    metric_name="mathalea_pass@1",
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


def _make_generative_prompt_fn(system_prompt):
    prefix = system_prompt or ""

    def prompt_fn(line, task_name: str = None):
        choices = line["choices"]
        gold_idx = int(line["answerKey"])
        valid_letters = "".join(LETTER_INDICES[: len(choices)])

        instruction = (
            "Répondez à la question à choix multiple suivante. La dernière ligne de votre réponse "
            "doit être au format suivant : 'Réponse : $LETTER' (sans les guillemets) où LETTER "
            f"est l'une des lettres {valid_letters}. Réfléchissez étape par étape avant de répondre."
        )

        choices_str = "\n".join(f"{letter}) {choice.strip()}" for letter, choice in zip(LETTER_INDICES, choices))

        query = f"{prefix}{instruction}\n\n{line['question'].strip()}\n\n{choices_str}"

        return Doc(
            task_name=task_name,
            query=query,
            choices=LETTER_INDICES[: len(choices)],
            gold_index=gold_idx,
            instruction=prefix + instruction,
        )

    return prompt_fn


def _make_generative_task(subset, alias, prompt_key):
    system_prompt = _get_instruction(prompt_key, subset)

    return LightevalTaskConfig(
        name=f"mathalea_generative_{prompt_key}:{alias}",
        prompt_function=_make_generative_prompt_fn(system_prompt),
        suite=["community"],
        hf_repo="OpenLLM-BPI/MathAleaMCQ",
        hf_subset=subset,
        hf_avail_splits=["dev", "test"],
        evaluation_splits=["test"],
        few_shots_split="dev",
        few_shots_select="sequential",
        generation_size=4096,
        metrics=[mathalea_generative_metric],
        stop_sequence=[],
        version=0,
    )


def _make_tasks(subset, alias, formulation, prompt_key):
    instruction = _get_instruction(prompt_key, subset)

    return LightevalTaskConfig(
        name=f"mathalea_{formulation.name.lower()}_{prompt_key}:{alias}",
        prompt_function=get_mcq_prompt_function(
            Language.FRENCH,
            lambda line, instr=instruction: {
                "question": line["question"],
                "choices": line["choices"],
                "gold_idx": int(line["answerKey"]),
                **({"instruction": instr} if instr else {}),
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
    _make_tasks(subset, remove_accents(subset), formulation, prompt_key)
    for subset in ["all"] + GRADE_LEVELS
    for formulation in FORMULATIONS
    for prompt_key in PROMPT_CONFIGS
] + [
    _make_generative_task(subset, remove_accents(subset), prompt_key)
    for subset in ["all"] + GRADE_LEVELS
    for prompt_key in PROMPT_CONFIGS
]

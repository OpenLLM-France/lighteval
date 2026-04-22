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
]

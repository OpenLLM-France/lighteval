"""
Custom evaluation task for the Exo7 French math multiple-choice benchmark.

This module implements a multi-label MC task with TruthfulQA MC2-style
probability scoring: log-likelihoods are normalized to probabilities,
and accuracy is measured as the probability mass assigned to correct answers.

Dataset: Lduignan1/exo7_mc
"""

import numpy as np

from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.metrics.utils.metric_utils import SampleLevelMetric
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod


# --- Custom MC2 metric ---


class Exo7MCMetric(SampleLevelComputation):
    """MC2-style probability mass metric for multi-label multiple choice.

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


# --- Prompt functions ---

FEWSHOT_EXAMPLES = """\
Pour les questions suivantes, il peut y avoir plusieurs bonnes réponses. Choisissez une seule bonne réponse possible.

Question :
Soient $f,g$ deux fonctions définies sur $\\Rr$. Quelles sont les assertions vraies ?

A. $f - 2g$ est une fonction définie sur $\\Rr$
B. $f^2 \\times g$ est une fonction définie sur $\\Rr$
C. $\\frac{f}{g^2}$ est une fonction définie sur $\\Rr$
D. $\\sqrt{f+g}$ est une fonction définie sur $\\Rr$

Réponse : $f^2 \\times g$ est une fonction définie sur $\\Rr$


Question :
Etant donné que $\\displaystyle f(3)=1$ et $f'(3)=5$. Une équation de la tangente à $\\mathcal{C}_f$ au point $(3,1)$ est :

A. $y=1(x-3)+5=x+2$
B. $y=1(x-3)-5=x-8$
C. $y=5(x-3)-1=5x-16$
D. $y=5(x-3)+1=5x-14$

Réponse : $y=5(x-3)+1=5x-14$


Question :
On considère les deux applications suivantes :
$$\\begin{array}{rccc}f:&\\Rr&\\to&\\Rr\\\\
& x&\\to & \\sin x \\end{array} \\quad \\mbox{et} \\quad \\begin{array}{rccc}g:&\\Rr^2&\\to&\\Rr^2\\\\
& (x,y)&\\to &(y,x). \\end{array}$$

Quelles sont les assertions vraies ?

A. $f(0)=0$
B. $f$ est une application linéaire
C. $g(x,y)=g(y,x)$, pour tout $(x,y) \\in \\Rr^2$
D. $g$ est une application linéaire

Réponse : $f(0)=0$

"""

ZEROSHOT_INSTRUCTION = "Pour la question suivante, il peut y avoir plusieurs bonnes réponses. Choisissez une seule bonne réponse possible.\n\n"


def _build_query(line):
    """Build the question + lettered choices portion of the prompt."""
    choices = line["targets"]["choices"]
    query = f"Question :\n{line['question']}\n\n"
    query += "".join(
        f"{letter}. {choice}\n"
        for letter, choice in zip(LETTER_INDICES, choices)
    )
    query += "\nRéponse :"
    return query


def _build_doc(line, query, task_name):
    """Build a Doc from the formatted query and dataset line."""
    choices = line["targets"]["choices"]
    labels = line["targets"]["labels"]
    gold_index = [i for i, label in enumerate(labels) if label == 1]

    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=gold_index,
        specific={"labels": labels},
    )


def prompt_exo7_mc_zeroshot(line, task_name: str = None):
    query = ZEROSHOT_INSTRUCTION + _build_query(line)
    return _build_doc(line, query, task_name)


def prompt_exo7_mc_fewshot(line, task_name: str = None):
    query = FEWSHOT_EXAMPLES + _build_query(line)
    return _build_doc(line, query, task_name)


# --- Task configs ---

exo7_mc_zeroshot_task = LightevalTaskConfig(
    name="exo7:mc_zeroshot",
    prompt_function=prompt_exo7_mc_zeroshot,
    suite=["community"],
    hf_repo="Lduignan1/Exo7",
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

exo7_mc_fewshot_task = LightevalTaskConfig(
    name="exo7:mc_fewshot",
    prompt_function=prompt_exo7_mc_fewshot,
    suite=["community"],
    hf_repo="Lduignan1/Exo7",
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

TASKS_TABLE = [exo7_mc_zeroshot_task, exo7_mc_fewshot_task]

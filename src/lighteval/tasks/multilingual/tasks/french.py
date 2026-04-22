"""
name:
French Evals

dataset:
fr-gouv-coordination-ia/IFEval-fr, fr-gouv-coordination-ia/gpqa-fr, fr-gouv-coordination-ia/bac-fr

abstract:
Collection of benchmarks for the french language.

languages:
french

tags:
knowledge, multiple-choice, qa

paper:
https://huggingface.co/fr-gouv-coordination-ia
"""

import random
from string import ascii_uppercase

import numpy as np

from lighteval.metrics.dynamic_metrics import MultilingualExtractiveMatchMetric
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.metrics_sample import PassAtK
from lighteval.metrics.normalizations import math_normalizer
from lighteval.metrics.utils.extractive_match_utils import IndicesExtractionConfig
from lighteval.metrics.utils.metric_utils import SampleLevelMetric, SamplingMethod
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.tasks.tasks.ifeval.main import ifeval_metrics
from lighteval.utils.language import Language
from lighteval.utils.utils import as_list


# Ifeval-fr prompt function
def prompt_ifeval_fr(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=line["prompt"],
        choices=[""],
        gold_index=0,
        instruction="",
        specific={"instructions_id_list": line["instruction_id_list"], "kwargs": line["kwargs"]},
    )


# qpqa-fr prompt function
def prompt_gpqa_fr(line, task_name: str = None):
    gold_index = random.randint(0, 3)
    choices = [line["Incorrect Answer 1"], line["Incorrect Answer 2"], line["Incorrect Answer 3"]]
    choices.insert(gold_index, line["Correct Answer"])

    instruction = "Choisissez la réponse correcte aux questions suivantes.\n\n"

    query = f"Question: {line['Question']}\n"
    query += "".join([f"{key}. {choice}\n" for key, choice in zip(ascii_uppercase, choices)])
    query += "Réponse: "
    return Doc(
        task_name=task_name,
        query=f"{instruction}{query}",
        choices=ascii_uppercase[: len(choices)],
        gold_index=gold_index,
        instruction=instruction,
    )

def prompt_gpqa_fr_instruct(line, task_name: str = None):
    """Prompt template adapted gpqa_instruct in src/lighteval/tasks/default_prompts.py"""
    gold_index = random.randint(0, 3)
    choices = [line["Incorrect Answer 1"], line["Incorrect Answer 2"], line["Incorrect Answer 3"]]
    choices.insert(gold_index, line["Correct Answer"])
    instruction = "Réponds à la question à choix multiple suivante. La dernière ligne de votre réponse doit être au format suivant : 'Réponse : $LETTER' (sans les guillemets) où LETTER est l'une des lettres ABCD. Réfléchissez étape par étape avant de répondre."
    query_template = "{Instruction}\n\n{Question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}"
    query = query_template.format(
        # Stripping to avoid accidental extra whitespaces, present in GPQA
        A=choices[0].strip(),
        B=choices[1].strip(),
        C=choices[2].strip(),
        D=choices[3].strip(),
        Question=line["problem"].strip(),
        Instruction=instruction,
    )

    return Doc(
        task_name=task_name,
        query=query,
        choices=ascii_uppercase[: len(choices)],
        gold_index=gold_index,
        instruction=instruction,
    )

# BAC-fr prompt function
def prompt_bac_fr(line, task_name: str = None):
    prompt = f"Enoncé: {line['enonce']}\n{line['instruction']}\n"
    if line["choix"] is not None:  # Multichoice evaluation
        # prompt += "\n".join([f"{ascii_uppercase[ix]}.{choix}" for ix, choix in enumerate(line["choix"])])
        return Doc(
            task_name=task_name,
            query=prompt,
            choices=as_list(line["choix"]),
            gold_index=line["choix"].index(line["choix correct"]),
            instruction="",
        )
    else:
        return Doc(task_name=task_name, query=prompt, choices=[line["reponse"]], gold_index=0, instruction="")


# IFEVal-fr task
ifeval_fr_task = LightevalTaskConfig(
    name="ifeval-fr",
    prompt_function=prompt_ifeval_fr,  # must be defined in the file or imported from src/lighteval/tasks/tasks_prompt_formatting.py
    # Mirror of fr-gouv-coordination-ia/IFEval-fr; the original repo was moved/removed.
    hf_repo="jzhang86/fr_ifeval",
    hf_subset="default",
    metrics=[ifeval_metrics],
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split="train",
    few_shots_select="random_sampling",
    generation_size=1280,
    stop_sequence=[],  # no stop sequence, will use eot token
    version=0,
)

# GPQA-fr task
# MCQ evaluation is not adapted for that task that requires reasoning before answering
# gpqa_fr_task = LightevalTaskConfig(
#     name="gpqa-fr",
#     suite=["community"],
#     prompt_function=prompt_gpqa_fr,
#     hf_repo="kurakurai/gpqa-fr", # "le-leadboard/gpqa-fr", # "fr-gouv-coordination-ia/gpqa-fr",
#     hf_subset="default",
#     hf_avail_splits=["train"],
#     evaluation_splits=["train"],
#     few_shots_split=None,
#     few_shots_select="random_sampling",
#     generation_size=1,
#     metrics=[Metrics.loglikelihood_acc],
#     stop_sequence=["\n"],
#     version=0,
# )

gpqa_fr_pass_at_1 = SampleLevelMetric(
    metric_name="gpqa_fr_pass@1",
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

gpqa_fr_task = LightevalTaskConfig(
    name="gpqa-fr:diamond",
    prompt_function=prompt_gpqa_fr_instruct,
    # Switched to le-leadboard/gpqa-fr; the original fr-gouv-coordination-ia/gpqa-fr is no longer available.
    hf_repo="le-leadboard/gpqa-fr",
    hf_subset="gpqa_diamond",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,  # needed for reasoning models like R1
    metrics=[gpqa_fr_pass_at_1],
    stop_sequence=[],  # no stop sequence, will use eos token
    version=0,
)

# BAC-fr task
bac_fr_task = LightevalTaskConfig(
    name="bac-fr",
    prompt_function=prompt_bac_fr,
    hf_repo="fr-gouv-coordination-ia/bac-fr",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select="random_sampling",
    generation_size=1,
    metrics=[
        Metrics.exact_match(sample_params={"normalize_gold": math_normalizer, "normalize_pred": math_normalizer}),
        Metrics.exact_match,
    ],
    stop_sequence=["\n"],
    version=0,
)

# STORE YOUR EVALS
TASKS_TABLE = [ifeval_fr_task, gpqa_fr_task, bac_fr_task]

"""
name:
MMLU Pro

dataset:
TIGER-Lab/MMLU-Pro

abstract:
MMLU-Pro dataset is a more robust and challenging massive multi-task
understanding dataset tailored to more rigorously benchmark large language
models' capabilities. This dataset contains 12K complex questions across various
disciplines.

languages:
english

tags:
general-knowledge, knowledge, multiple-choice

paper:
https://arxiv.org/abs/2406.01574

starred:
true
"""

from string import ascii_uppercase

from inspect_ai.dataset import Sample
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


TEMPLATE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of {letters}. Think step by step before answering.

{question}

{choices}

Answer:""".strip()


def mmlu_pro_prompt_function(line, task_name: str = None):
    n_options = len(line["options"])
    letters = ascii_uppercase[:n_options]
    choices_str = "\n".join([f"{letter}: {choice}" for letter, choice in zip(letters, line["options"])])

    query = TEMPLATE.format(
        letters=letters,
        question=line["question"],
        choices=choices_str,
    )

    return Doc(
        task_name=task_name,
        query=query,
        choices=list(letters),
        gold_index=line["answer_index"],
        instruction=query,
    )


def record_to_sample(record):
    return Sample(input=record["question"], target=record["answer"], choices=record["options"])


mmlu_pro = LightevalTaskConfig(
    name="mmlu_pro",
    prompt_function=mmlu_pro_prompt_function,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
    hf_repo="TIGER-Lab/MMLU-Pro",
    hf_subset="default",
    hf_revision="3373e0b32277875b8db2aa555a333b78a08477ea",
    evaluation_splits=("test",),
    few_shots_split="validation",
    metrics=[Metrics.gpqa_instruct_metric],
)


# Alternative handmade version without inspect_ai, kept for side-by-side comparison.
def mmlu_pro_raw_prompt(line, task_name: str = None):
    n_options = len(line["options"])
    letters = ascii_uppercase[:n_options]
    choices_str = "\n".join([f"{letter}: {choice}" for letter, choice in zip(letters, line["options"])])

    instruction = (
        "Answer the following multiple choice question. The last line of your response should be of the following"
        f" format: 'Answer: $LETTER' (without quotes) where LETTER is one of {letters}."
        " Think step by step before answering.\n\n"
    )

    query = instruction + f"{line['question']}\n\n{choices_str}\n\nAnswer:"

    # Use the dataset's worked chain-of-thought as the gold when it is
    # available (validation split → few-shot pool). Without this, the shown
    # demonstrations end in a bare letter and teach the model to answer
    # directly, which MMLU-Pro was designed to penalise (see §5 of
    # arXiv:2406.01574 and lm-eval-harness's mmlu_pro). Test rows carry an
    # empty cot_content, so we fall back to the letter and the metric's
    # extractor works as before.
    cot = (line.get("cot_content") or "").strip()
    if cot.startswith("A:"):
        cot = cot[2:].lstrip()
    if cot:
        choices = [cot]
        gold_index = 0
    else:
        choices = list(letters)
        gold_index = line["answer_index"]

    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=gold_index,
        instruction=instruction,
    )


mmlu_pro_raw = LightevalTaskConfig(
    name="mmlu_pro_raw",
    prompt_function=mmlu_pro_raw_prompt,
    hf_repo="TIGER-Lab/MMLU-Pro",
    hf_subset="default",
    hf_revision="3373e0b32277875b8db2aa555a333b78a08477ea",
    evaluation_splits=["test"],
    few_shots_split="validation",
    few_shots_select=None,
    generation_size=4096,
    metrics=[Metrics.gpqa_instruct_metric],
    stop_sequence=None,
    version=1,
)


TASKS_TABLE = [mmlu_pro, mmlu_pro_raw]

"""
name:
Mgsm

dataset:
juletxara/mgsm

abstract:
Mgsm multilingual benchmark.

languages:
bengali, chinese, english, french, german, japanese, russian, spanish, swahili,
telugu, thai

tags:
math, multilingual, reasoning

paper:
"""

from langcodes import standardize_tag

from lighteval.metrics.dynamic_metrics import (
    MultilingualQuasiExactMatchMetric,
)
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.templates.qa import get_qa_prompt_function
from lighteval.utils.language import Language


TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"mgsm_{language.value}",
        prompt_function=get_qa_prompt_function(
            language,
            lambda line: {
                "question": line["question"],
                # The cot is available but we have no use:
                # line["answer"]
                "choices": [str(line["answer_number"])],
            },
        ),
        hf_repo="juletxara/mgsm",
        hf_subset=standardize_tag(language.value),
        evaluation_splits=("test",),
        few_shots_split="train",
        generation_size=25,
        metrics=[
            MultilingualQuasiExactMatchMetric(language, "full"),
        ],
        stop_sequence=("\n",),
    )
    for language in [
        Language.ENGLISH,
        Language.SPANISH,
        Language.FRENCH,
        Language.GERMAN,
        Language.RUSSIAN,
        Language.CHINESE,
        Language.JAPANESE,
        Language.THAI,
        Language.SWAHILI,
        Language.BENGALI,
        Language.TELUGU,
    ]
]


# MGSM rev2: revised MGSM data (lightonai/mgsm-rev2), test split only (no train / no CoT), zero-shot.
# Same 11 languages and (question, answer_number) schema as the original MGSM above.
TASKS_TABLE += [
    LightevalTaskConfig(
        name=f"mgsm_rev2_{language.value}",
        prompt_function=get_qa_prompt_function(
            language,
            lambda line: {
                "question": line["question"],
                "choices": [str(line["answer_number"])],
            },
        ),
        hf_repo="lightonai/mgsm-rev2",
        hf_revision="1463c6dc7991a8751b8e28e76f6c561b9201eb55",
        hf_subset=standardize_tag(language.value),
        evaluation_splits=("test",),
        few_shots_split=None,
        generation_size=25,
        metrics=[
            MultilingualQuasiExactMatchMetric(language, "full"),
        ],
        stop_sequence=("\n",),
    )
    for language in [
        Language.ENGLISH,
        Language.SPANISH,
        Language.FRENCH,
        Language.GERMAN,
        Language.RUSSIAN,
        Language.CHINESE,
        Language.JAPANESE,
        Language.THAI,
        Language.SWAHILI,
        Language.BENGALI,
        Language.TELUGU,
    ]
]

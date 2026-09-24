"""
name:
Mgsm

dataset:
juletxara/mgsm

abstract:
Multilingual Grade School Math Benchmark (MGSM) is a benchmark of grade-school
math problems.
The same 250 problems from GSM8K are each translated via human annotators in 10
languages.

languages:
english, spanish, french, german, russian, chinese, japanese, thai, swahili, bengali, telugu

tags:
math, multilingual, reasoning

paper:
https://arxiv.org/abs/2210.03057
"""

from lighteval.metrics.metrics import Metrics
from lighteval.metrics.normalizations import helm_normalizer
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


MGSM_HF_REVISION = "2e3d3e94b252b3a5829ed998a4f6229e15adb1a7"
MGSM_METRICS = [
    Metrics.exact_match(
        sample_params={
            "type_exact_match": "suffix",
            "normalize_gold": helm_normalizer,
            "normalize_pred": helm_normalizer,
        }
    ),
    Metrics.expr_gold_metric(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
]


def mgsm_prompt(line, question_key, answer_key, task_name: str = None):
    if line.get("answer") is not None:  # rev2 data ships only answer_number (no CoT answer)
        query = f"{line['question']}\n{answer_key}"
        gold = f" {line['answer'][len(answer_key) + 1 :]}"
    else:
        query = f"{question_key} {line['question']}\n{answer_key}"
        gold = f" {str(line['answer_number'])}"
    return Doc(task_name=task_name, query=query, choices=[gold], gold_index=0)


def mgsm_en_prompt(line, task_name: str = None):
    question_key = "Question:"
    answer_key = "Step-by-Step Answer:"
    return mgsm_prompt(line, question_key, answer_key, task_name)


def mgsm_es_prompt(line, task_name: str = None):
    question_key = "Pregunta:"
    answer_key = "Respuesta paso a paso:"
    return mgsm_prompt(line, question_key, answer_key, task_name)


def mgsm_fr_prompt(line, task_name: str = None):
    question_key = "Question:"
    answer_key = "Réponse étape par étape :"
    return mgsm_prompt(line, question_key, answer_key, task_name)


def mgsm_de_prompt(line, task_name: str = None):
    question_key = "Frage:"
    answer_key = "Schritt-für-Schritt-Antwort:"
    return mgsm_prompt(line, question_key, answer_key, task_name)


def mgsm_ru_prompt(line, task_name: str = None):
    question_key = "Задача:"
    answer_key = "Пошаговоерешение:"
    return mgsm_prompt(line, question_key, answer_key, task_name)


def mgsm_zh_prompt(line, task_name: str = None):
    question_key = "问题:"
    answer_key = "逐步解答:"
    return mgsm_prompt(line, question_key, answer_key, task_name)


def mgsm_ja_prompt(line, task_name: str = None):
    question_key = "問題:"
    answer_key = "ステップごとの答え:"
    return mgsm_prompt(line, question_key, answer_key, task_name)


def mgsm_th_prompt(line, task_name: str = None):
    question_key = "โจทย์:"
    answer_key = "คำตอบทีละขั้นตอน:"
    return mgsm_prompt(line, question_key, answer_key, task_name)


def mgsm_sw_prompt(line, task_name: str = None):
    question_key = "Swali:"
    answer_key = "Jibu la Hatua kwa Hatua:"
    return mgsm_prompt(line, question_key, answer_key, task_name)


def mgsm_bn_prompt(line, task_name: str = None):
    question_key = "প্রশ্ন:"
    answer_key = "ধাপে ধাপে উত্তর:"
    return mgsm_prompt(line, question_key, answer_key, task_name)


def mgsm_te_prompt(line, task_name: str = None):
    question_key = "ప్రశ్న:"
    answer_key = "దశలవారీగా సమాధానం:"
    return mgsm_prompt(line, question_key, answer_key, task_name)


mgsm_en = LightevalTaskConfig(
    name="mgsm:en",
    prompt_function=mgsm_en_prompt,
    hf_repo="juletxara/mgsm",
    hf_subset="en",
    hf_revision=MGSM_HF_REVISION,
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=MGSM_METRICS,
    stop_sequence=["\n", "=", "Question="],
    version=0,
)

mgsm_es = LightevalTaskConfig(
    name="mgsm:es",
    prompt_function=mgsm_es_prompt,
    hf_repo="juletxara/mgsm",
    hf_subset="es",
    hf_revision=MGSM_HF_REVISION,
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=MGSM_METRICS,
    stop_sequence=["\n", "=", "Pregunta="],
    version=0,
)

mgsm_fr = LightevalTaskConfig(
    name="mgsm:fr",
    prompt_function=mgsm_fr_prompt,
    hf_repo="juletxara/mgsm",
    hf_subset="fr",
    hf_revision=MGSM_HF_REVISION,
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=MGSM_METRICS,
    stop_sequence=["\n", "=", "Question="],
    version=0,
)

mgsm_de = LightevalTaskConfig(
    name="mgsm:de",
    prompt_function=mgsm_de_prompt,
    hf_repo="juletxara/mgsm",
    hf_subset="de",
    hf_revision=MGSM_HF_REVISION,
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=MGSM_METRICS,
    stop_sequence=["\n", "=", "Frage="],
    version=0,
)

mgsm_ru = LightevalTaskConfig(
    name="mgsm:ru",
    prompt_function=mgsm_ru_prompt,
    hf_repo="juletxara/mgsm",
    hf_subset="ru",
    hf_revision=MGSM_HF_REVISION,
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=MGSM_METRICS,
    stop_sequence=["\n", "=", "Задача="],
    version=0,
)

mgsm_zh = LightevalTaskConfig(
    name="mgsm:zh",
    prompt_function=mgsm_zh_prompt,
    hf_repo="juletxara/mgsm",
    hf_subset="zh",
    hf_revision=MGSM_HF_REVISION,
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=MGSM_METRICS,
    stop_sequence=["\n", "=", "问题="],
    version=0,
)

mgsm_ja = LightevalTaskConfig(
    name="mgsm:ja",
    prompt_function=mgsm_ja_prompt,
    hf_repo="juletxara/mgsm",
    hf_subset="ja",
    hf_revision=MGSM_HF_REVISION,
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=MGSM_METRICS,
    stop_sequence=["\n", "=", "問題="],
    version=0,
)

mgsm_th = LightevalTaskConfig(
    name="mgsm:th",
    prompt_function=mgsm_th_prompt,
    hf_repo="juletxara/mgsm",
    hf_subset="th",
    hf_revision=MGSM_HF_REVISION,
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=MGSM_METRICS,
    stop_sequence=["\n", "=", "โจทย์="],
    version=0,
)

mgsm_sw = LightevalTaskConfig(
    name="mgsm:sw",
    prompt_function=mgsm_sw_prompt,
    hf_repo="juletxara/mgsm",
    hf_subset="sw",
    hf_revision=MGSM_HF_REVISION,
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=MGSM_METRICS,
    stop_sequence=["\n", "=", "Swali="],
    version=0,
)

mgsm_bn = LightevalTaskConfig(
    name="mgsm:bn",
    prompt_function=mgsm_bn_prompt,
    hf_repo="juletxara/mgsm",
    hf_subset="bn",
    hf_revision=MGSM_HF_REVISION,
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=MGSM_METRICS,
    stop_sequence=["\n", "=", "প্রশ্ন="],
    version=0,
)

mgsm_te = LightevalTaskConfig(
    name="mgsm:te",
    prompt_function=mgsm_te_prompt,
    hf_repo="juletxara/mgsm",
    hf_subset="te",
    hf_revision=MGSM_HF_REVISION,
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=MGSM_METRICS,
    stop_sequence=["\n", "=", "ప్రశ్న="],
    version=0,
)

TASKS_TABLE = [
    mgsm_en,
    mgsm_es,
    mgsm_fr,
    mgsm_de,
    mgsm_ru,
    mgsm_zh,
    mgsm_ja,
    mgsm_th,
    mgsm_sw,
    mgsm_bn,
    mgsm_te,
]


# MGSM rev2: revised MGSM data (https://huggingface.co/datasets/lightonai/mgsm-rev2), test split only
# (no train / no chain-of-thought), run zero-shot. Same per-language prompts and metrics as MGSM above.
MGSM_REV2_HF_REVISION = "1463c6dc7991a8751b8e28e76f6c561b9201eb55"


def _mgsm_rev2_task(subset, prompt_function, stop_last):
    return LightevalTaskConfig(
        name=f"mgsm-rev2:{subset}",
        prompt_function=prompt_function,
        hf_repo="lightonai/mgsm-rev2",
        hf_subset=subset,
        hf_revision=MGSM_REV2_HF_REVISION,
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=None,
        metrics=MGSM_METRICS,
        stop_sequence=["\n", "=", stop_last],
        version=0,
    )


mgsm_rev2_tasks = [
    _mgsm_rev2_task("en", mgsm_en_prompt, mgsm_en.stop_sequence[-1]),
    _mgsm_rev2_task("es", mgsm_es_prompt, mgsm_es.stop_sequence[-1]),
    _mgsm_rev2_task("fr", mgsm_fr_prompt, mgsm_fr.stop_sequence[-1]),
    _mgsm_rev2_task("de", mgsm_de_prompt, mgsm_de.stop_sequence[-1]),
    _mgsm_rev2_task("ru", mgsm_ru_prompt, mgsm_ru.stop_sequence[-1]),
    _mgsm_rev2_task("zh", mgsm_zh_prompt, mgsm_zh.stop_sequence[-1]),
    _mgsm_rev2_task("ja", mgsm_ja_prompt, mgsm_ja.stop_sequence[-1]),
    _mgsm_rev2_task("th", mgsm_th_prompt, mgsm_th.stop_sequence[-1]),
    _mgsm_rev2_task("sw", mgsm_sw_prompt, mgsm_sw.stop_sequence[-1]),
    _mgsm_rev2_task("bn", mgsm_bn_prompt, mgsm_bn.stop_sequence[-1]),
    _mgsm_rev2_task("te", mgsm_te_prompt, mgsm_te.stop_sequence[-1]),
]

TASKS_TABLE += mgsm_rev2_tasks

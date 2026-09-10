# MIT License

# Copyright (c) 2024 The HuggingFace Team

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

# ruff: noqa: F405, F403, F401
"""
Custom evaluation tasks for lighteval

This file generally creates just a TASKS_TABLE and TASKS_GROUPS which are then imported by LightEval.
"""

import random
import re
from typing import Any, Dict, List, Optional, Union

import numpy as np

from lighteval.metrics.metrics import Metrics
from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.metrics.normalizations import LogProbCharNorm
from lighteval.metrics.utils.llm_as_judge import JudgeLM
from lighteval.metrics.utils.metric_utils import SampleLevelMetric
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod


# fmt: off
LETTER_INDICES_AR = ["أ", "ب", "ج", "د", "هـ", "و", "ز", "ح", "ط", "ي", "ك", "ل", "م", "ن", "س", "ع", "ف", "ص", "ق", "ر", "ش", "ت", "ث", "خ", "ذ", "ض", "ظ", "غ"]
# fmt: on

# ArabicMMLU
# fmt: off
ARABIC_MMLU_SUBSETS = [
    "All", "Islamic Studies", "Islamic Studies (Middle School)", "Islamic Studies (Primary School)", "Islamic Studies (High School)", "Driving Test",
    "Natural Science (Middle School)", "Natural Science (Primary School)", "History (Middle School)", "History (Primary School)", "History (High School)", "General Knowledge",
    "General Knowledge (Middle School)", "General Knowledge (Primary School)", "Law (Professional)", "Physics (High School)", "Social Science (Middle School)",
    "Social Science (Primary School)", "Management (University)", "Arabic Language (Middle School)", "Arabic Language (Primary School)", "Arabic Language (High School)", "Political Science (University)",
    "Philosophy (High School)", "Accounting (University)", "Computer Science (Middle School)", "Computer Science (Primary School)", "Computer Science (High School)", "Computer Science (University)",
    "Geography (Middle School)", "Geography (Primary School)", "Geography (High School)", "Math (Primary School)", "Biology (High School)", "Economics (Middle School)",
    "Economics (High School)", "Economics (University)", "Arabic Language (General)", "Arabic Language (Grammar)", "Civics (Middle School)", "Civics (High School)"
]
# fmt: on


def arabic_mmlu_pfn(line, task_name: str = None):
    instruction = "السؤال التالي هو سؤال متعدد الإختيارات. اختر الإجابة الصحيحة:\n\n"

    # Define the mapping from Latin to Arabic letters
    latin_to_arabic = {"A": "أ", "B": "ب", "C": "ج", "D": "د", "E": "هـ"}

    # Create a list of valid choices with corresponding Arabic keys
    choices = []
    valid_keys_latin = []
    valid_keys_arabic = []

    # Enumerate through the options and append the valid ones
    for idx, key in enumerate(["A", "B", "C", "D", "E"]):
        option = line.get(f"Option {idx + 1}")
        if option:  # Check if option is not null
            choices.append(option)
            valid_keys_latin.append(key)  # Append the Latin key (A, B, C, D, E)
            valid_keys_arabic.append(latin_to_arabic[key])  # Append the corresponding Arabic letter

    # Find the correct index for the answer key in the Arabic version
    answer_index = valid_keys_latin.index(line["Answer Key"])

    # Construct the query with Arabic letters
    query = f"{instruction}{line['Question']}\n"
    query += "".join([f"{key}. {choice}\n" for key, choice in zip(valid_keys_arabic, choices)])
    query += "الإجابة:"

    return Doc(
        task_name=task_name,
        query=query,
        choices=valid_keys_arabic,  # Return only valid choices (Arabic keys)
        gold_index=answer_index,  # Correct index in the valid Arabic keys
        instruction=instruction,
    )


class CustomArabicMMLUTask(LightevalTaskConfig):
    def __init__(
        self,
        name,
        hf_subset,
    ):
        super().__init__(
            name=name,
            hf_subset=hf_subset,
            prompt_function=arabic_mmlu_pfn,
            hf_repo="MBZUAI/ArabicMMLU",
            metrics=[Metrics.loglikelihood_acc(sample_params={"logprob_normalization": LogProbCharNorm()})],
            hf_avail_splits=["test"],
            evaluation_splits=["test"],
            few_shots_split=["dev"],
            few_shots_select="sequential",
            suite=["community"],
            generation_size=-1,
            stop_sequence=None,
            version=0,
        )


ARABIC_MMLU_TASKS = [
    CustomArabicMMLUTask(name=f"arabic_mmlu:{subset}", hf_subset=subset) for subset in ARABIC_MMLU_SUBSETS
]


# ARABIC MMLU HT ##
# fmt: off
ARABIC_MMLU_HT_SUBSETS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge", "college_biology", "college_chemistry", "college_computer_science",
    "college_mathematics", "college_medicine", "college_physics", "computer_security", "conceptual_physics", "econometrics", "electrical_engineering",
    "elementary_mathematics", "formal_logic", "global_facts", "high_school_biology", "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography", "high_school_government_and_politics", "high_school_macroeconomics", "high_school_mathematics",
    "high_school_microeconomics", "high_school_physics", "high_school_psychology", "high_school_statistics", "high_school_us_history", "high_school_world_history",
    "human_aging", "human_sexuality", "international_law", "jurisprudence", "logical_fallacies", "machine_learning", "management", "marketing", "medical_genetics",
    "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition", "philosophy", "prehistory", "professional_accounting", "professional_law",
    "professional_medicine", "professional_psychology", "public_relations", "security_studies", "sociology", "us_foreign_policy", "virology", "world_religions"
]
# fmt: on


def arabic_mmlu_ht_pfn(line, task_name: str = None):
    instruction = "السؤال التالي هو سؤال متعدد الإختيارات. اختر الإجابة الصحيحة:\n\n"
    choices = line["choices"]
    answer_index = line["answer"]  # It is an int reflecting the index of correct answer in line["choices"]

    query = f"{instruction}{line['question']}\n"
    query += "".join([f"{idx}. {choice}\n" for idx, choice in enumerate(choices, start=1)])
    query += "الإجابة:"

    return Doc(
        task_name=task_name,
        query=query,
        choices=[str(i) for i in range(1, len(choices) + 1)],  # List of strings instead of ints
        gold_index=answer_index,
        instruction=instruction,
    )


class CustomArabicMMLUHTTask(LightevalTaskConfig):
    def __init__(
        self,
        name,
        hf_subset,
    ):
        super().__init__(
            name=name,
            hf_subset=hf_subset,
            prompt_function=arabic_mmlu_ht_pfn,
            hf_repo="MBZUAI/human_translated_arabic_mmlu",
            metrics=[Metrics.loglikelihood_acc(sample_params={"logprob_normalization": LogProbCharNorm()})],
            hf_avail_splits=["test"],
            evaluation_splits=["test"],
            few_shots_split=None,
            few_shots_select=None,
            suite=["community"],
            generation_size=-1,
            stop_sequence=None,
            version=0,
        )


ARABIC_MMLU_HT_TASKS = [
    CustomArabicMMLUHTTask(name=f"arabic_mmlu_ht:{subset}", hf_subset=subset) for subset in ARABIC_MMLU_HT_SUBSETS
]

# ARABIC MMLU MT ##
# fmt: off
ARABIC_MMLU_MT_SUBSETS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge", "college_biology", "college_chemistry", "college_computer_science",
    "college_mathematics", "college_medicine", "college_physics", "computer_security", "conceptual_physics", "econometrics", "electrical_engineering",
    "elementary_mathematics", "formal_logic", "global_facts", "high_school_biology", "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography", "high_school_government_and_politics", "high_school_macroeconomics", "high_school_mathematics",
    "high_school_microeconomics", "high_school_physics", "high_school_psychology", "high_school_statistics", "high_school_us_history", "high_school_world_history",
    "human_aging", "human_sexuality", "international_law", "jurisprudence", "logical_fallacies", "machine_learning", "management", "marketing", "medical_genetics",
    "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition", "philosophy", "prehistory", "professional_accounting", "professional_law",
    "professional_medicine", "professional_psychology", "public_relations", "security_studies", "sociology", "us_foreign_policy", "virology", "world_religions"
]
# fmt: on


def arabic_mmlu_mt_pfn(line, task_name: str = None):
    instruction = "السؤال التالي هو سؤال متعدد الإختيارات. اختر الإجابة الصحيحة: أ، ب، ج، أو د... إلخ. \n\n"
    choices = [line["A"], line["B"], line["C"], line["D"]]
    # Answers are provided with roman letters - we look for the correct index in LETTER_INDICES,
    # it will then be applied to arabic letters
    answer_index = LETTER_INDICES.index(
        line["answer"]
    )  # line["answer"] is the correct answer. That's why we need to index it !

    query = f"{instruction}{line['question']}\n"
    query += "".join([f"{key}. {choice}\n" for key, choice in zip(LETTER_INDICES_AR[:4], choices)])
    query += "الإجابة:"

    return Doc(
        task_name=task_name,
        query=query,
        choices=LETTER_INDICES_AR[:4],
        gold_index=answer_index,
        instruction=instruction,
    )


class CustomArabicMMLUMTTask(LightevalTaskConfig):
    def __init__(
        self,
        name,
        hf_subset,
    ):
        super().__init__(
            name=name,
            hf_subset=hf_subset,
            prompt_function=arabic_mmlu_mt_pfn,
            hf_repo="OALL/Arabic_MMLU",
            metrics=[Metrics.loglikelihood_acc(sample_params={"logprob_normalization": LogProbCharNorm()})],
            hf_avail_splits=["test", "dev"],
            evaluation_splits=["test"],
            few_shots_split="dev",
            few_shots_select="sequential",
            suite=["community"],
            generation_size=-1,
            stop_sequence=None,
            version=0,
        )


ARABIC_MMLU_MT_TASKS = [
    CustomArabicMMLUMTTask(name=f"arabic_mmlu_mt:{subset}", hf_subset=subset) for subset in ARABIC_MMLU_MT_SUBSETS
]


# ACVA ##
# fmt: off
ACVA_SUBSETS = [
    "Algeria", "Ancient_Egypt", "Arab_Empire", "Arabic_Architecture", "Arabic_Art", "Arabic_Astronomy", "Arabic_Calligraphy", "Arabic_Ceremony",
    "Arabic_Clothing", "Arabic_Culture", "Arabic_Food", "Arabic_Funeral", "Arabic_Geography", "Arabic_History", "Arabic_Language_Origin",
    "Arabic_Literature", "Arabic_Math", "Arabic_Medicine", "Arabic_Music", "Arabic_Ornament", "Arabic_Philosophy", "Arabic_Physics_and_Chemistry",
    "Arabic_Wedding", "Bahrain", "Comoros", "Egypt_modern", "InfluenceFromAncientEgypt", "InfluenceFromByzantium", "InfluenceFromChina",
    "InfluenceFromGreece", "InfluenceFromIslam", "InfluenceFromPersia", "InfluenceFromRome", "Iraq", "Islam_Education", "Islam_branches_and_schools",
    "Islamic_law_system", "Jordan", "Kuwait", "Lebanon", "Libya", "Mauritania", "Mesopotamia_civilization", "Morocco", "Oman", "Palestine", "Qatar",
    "Saudi_Arabia", "Somalia", "Sudan", "Syria", "Tunisia", "United_Arab_Emirates", "Yemen",
    "communication", "computer_and_phone", "daily_life", "entertainment"
]
# fmt: on


def acva_pfn(line, task_name: str = None):
    question = line["question"]
    answer = line["answer"]

    return Doc(
        task_name=task_name,
        query=f"السؤال: {question}\nالإجابة:",
        choices=["صح", "خطأ"],
        gold_index=["صح", "خطأ"].index(answer),
    )


class CustomACVATask(LightevalTaskConfig):
    def __init__(
        self,
        name,
        hf_subset,
    ):
        super().__init__(
            name=name,
            hf_subset=hf_subset,
            prompt_function=acva_pfn,
            hf_repo="OALL/ACVA",
            metrics=[Metrics.loglikelihood_acc(sample_params={"logprob_normalization": LogProbCharNorm()})],
            hf_avail_splits=["test", "validation"],
            evaluation_splits=["test"],
            few_shots_split="validation",
            few_shots_select="sequential",
            suite=["community"],
            generation_size=-1,
            stop_sequence=None,
            version=0,
        )


ACVA_TASKS = [CustomACVATask(name=f"acva:{subset}", hf_subset=subset) for subset in ACVA_SUBSETS]


# AraTrust ##
# fmt: off
ARATRUST_SUBSETS = [
    "Trustfulness", "MentalHealth", "PhysicalHealth", "Offensive", "Ethics", "Privacy", "Unfairness", "Illegal",
]
# fmt: on


def aratrust_pfn(line, task_name: str = None):
    instruction = "السؤال التالي هو سؤال متعدد الإختيارات. اختر الإجابة الصحيحة: أ، ب أو ج. \n\n"
    choices = [line["A"], line["B"], line["C"]]
    # Answers are provided with roman letters - we look for the correct index in LETTER_INDICES,
    # it will then be applied to arabic letters
    answer_index = LETTER_INDICES_AR.index(
        line["Answer"]
    )  # line["answer"] is the correct answer. That's why we need to index it !

    query = f"{instruction}{line['Question']}\n"
    query += "".join([f"{choice}\n" for choice in choices])
    query += "الإجابة:"

    return Doc(
        task_name=task_name,
        query=query,
        choices=LETTER_INDICES_AR[:3],
        gold_index=answer_index,
        instruction=instruction,
    )


class CustomAraTrustTask(LightevalTaskConfig):
    def __init__(
        self,
        name,
        hf_subset,
    ):
        super().__init__(
            name=name,
            hf_subset=hf_subset,
            prompt_function=aratrust_pfn,
            hf_repo="asas-ai/AraTrust-categorized",
            metrics=[Metrics.loglikelihood_acc(sample_params={"logprob_normalization": LogProbCharNorm()})],
            hf_avail_splits=["train"],
            evaluation_splits=["train"],
            few_shots_split=None,
            few_shots_select=None,
            suite=["community"],
            generation_size=-1,
            stop_sequence=None,
            version=0,
        )


ARATRUST_TASKS = [CustomAraTrustTask(name=f"aratrust:{subset}", hf_subset=subset) for subset in ARATRUST_SUBSETS]


def arabic_exams_pfn(line, task_name: str = None):
    topic = line["subject"]
    question = line["question"]
    choices = [line["A"], line["B"], line["C"], line["D"]]
    choices_formatted = [f" {LETTER_INDICES_AR[i]}) {choice}\n" for i, choice in enumerate(choices)]
    answer = line["answer"]
    answer_index = LETTER_INDICES.index(answer)

    instruction = f"الأسئلة التالية هي أسئلة متعددة الإختيارات مع الجواب الصحيح حول {topic.replace('_', ' ')}. \n\n"
    query = f"{instruction}السؤال: {question}\n"
    query += "\n".join(choices_formatted)
    query += "\nالإجابة:"

    return Doc(
        task_name=task_name,
        query=query,
        choices=LETTER_INDICES_AR[:4],
        gold_index=answer_index,
        instruction=instruction,
    )


# ARABIC EXAMS ##
arabic_exams_task = LightevalTaskConfig(
    name="arabic_exams",
    prompt_function=arabic_exams_pfn,
    suite=["community"],
    hf_repo="OALL/Arabic_EXAMS",
    hf_subset="default",
    hf_avail_splits=["test", "validation"],
    evaluation_splits=["test"],
    few_shots_split="validation",
    few_shots_select="sequential",
    metrics=[Metrics.loglikelihood_acc(sample_params={"logprob_normalization": LogProbCharNorm()})],
    version=0,
)


# ALGHAFA NATIVE ##
# fmt: off
ALGHAFA_SUBSETS = [
    "mcq_exams_test_ar", "meta_ar_dialects", "meta_ar_msa", "multiple_choice_facts_truefalse_balanced_task", "multiple_choice_grounded_statement_soqal_task",
    "multiple_choice_grounded_statement_xglue_mlqa_task", "multiple_choice_rating_sentiment_no_neutral_task", "multiple_choice_rating_sentiment_task",
    "multiple_choice_sentiment_task"
]
# fmt: on


def alghafa_pfn(line, task_name: str = None):
    question = line["query"]
    answer_index = int(line["label"])
    allowed_keys = [f"sol{i}" for i in range(1, 6)]
    extracted_choices = [line[key] for key in allowed_keys if key in line]
    choices = [str(i) for i in range(len(extracted_choices))]

    instruction = "الأسئلة التالية هي أسئلة متعددة الإختيارات مع الجواب الصحيح\n\n"
    query = f"{instruction}السؤال: {question}\n"

    for index, choice in enumerate(extracted_choices):
        query += f"{index}) {choice}\n"

    query += "الإجابة:"

    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=answer_index,
        instruction=instruction,
    )


class CustomAlGhafaNativeTask(LightevalTaskConfig):
    def __init__(
        self,
        name,
        hf_subset,
    ):
        super().__init__(
            name=name,
            hf_subset=hf_subset,
            prompt_function=alghafa_pfn,
            hf_repo="OALL/AlGhafa-Arabic-LLM-Benchmark-Native",
            metrics=[Metrics.loglikelihood_acc(sample_params={"logprob_normalization": LogProbCharNorm()})],
            hf_avail_splits=["test", "validation"],
            evaluation_splits=["test"],
            few_shots_split="validation",
            few_shots_select="sequential",
            suite=["community"],
            generation_size=-1,
            stop_sequence=None,
            version=0,
        )


ALGHAFA_TASKS = [CustomAlGhafaNativeTask(name=f"alghafa:{subset}", hf_subset=subset) for subset in ALGHAFA_SUBSETS]

# ALGHAFA TRANSLATED ##
# race_ar
race_ar_task = LightevalTaskConfig(
    name="race_ar",
    prompt_function=alghafa_pfn,
    suite=["community"],
    hf_repo="OALL/AlGhafa-Arabic-LLM-Benchmark-Translated",
    hf_subset="race_ar",
    hf_avail_splits=["test", "validation"],
    evaluation_splits=["test"],
    few_shots_split="validation",
    few_shots_select="sequential",
    metrics=[Metrics.loglikelihood_acc(sample_params={"logprob_normalization": LogProbCharNorm()})],
    version=0,
)


# piqa_ar
piqa_ar_task = LightevalTaskConfig(
    name="piqa_ar",
    prompt_function=alghafa_pfn,
    suite=["community"],
    hf_repo="OALL/AlGhafa-Arabic-LLM-Benchmark-Translated",
    hf_subset="piqa_ar",
    hf_avail_splits=["test", "validation"],
    evaluation_splits=["test"],
    few_shots_split="validation",
    few_shots_select="sequential",
    metrics=[Metrics.loglikelihood_acc(sample_params={"logprob_normalization": LogProbCharNorm()})],
    version=0,
)


# arc_easy_ar
arc_easy_ar_task = LightevalTaskConfig(
    name="arc_easy_ar",
    prompt_function=alghafa_pfn,
    suite=["community"],
    hf_repo="OALL/AlGhafa-Arabic-LLM-Benchmark-Translated",
    hf_subset="arc_easy_ar",
    hf_avail_splits=["test", "validation"],
    evaluation_splits=["test"],
    few_shots_split="validation",
    few_shots_select="sequential",
    metrics=[Metrics.loglikelihood_acc(sample_params={"logprob_normalization": LogProbCharNorm()})],
    version=0,
)


# arc_challenge_okapi_ar
arc_challenge_okapi_ar_task = LightevalTaskConfig(
    name="arc_challenge_okapi_ar",
    prompt_function=alghafa_pfn,
    suite=["community"],
    hf_repo="OALL/AlGhafa-Arabic-LLM-Benchmark-Translated",
    hf_subset="arc_challenge_okapi_ar",
    hf_avail_splits=["test", "validation"],
    evaluation_splits=["test"],
    few_shots_split="validation",
    few_shots_select="sequential",
    metrics=[Metrics.loglikelihood_acc(sample_params={"logprob_normalization": LogProbCharNorm()})],
    version=0,
)


# mmlu_okapi_ar
mmlu_okapi_ar_task = LightevalTaskConfig(
    name="mmlu_okapi_ar",
    prompt_function=alghafa_pfn,
    suite=["community"],
    hf_repo="OALL/AlGhafa-Arabic-LLM-Benchmark-Translated",
    hf_subset="mmlu_okapi_ar",
    hf_avail_splits=["test", "validation"],
    evaluation_splits=["test"],
    few_shots_split="validation",
    few_shots_select="sequential",
    metrics=[Metrics.loglikelihood_acc(sample_params={"logprob_normalization": LogProbCharNorm()})],
    version=0,
)


# openbook_qa_ext_ar
openbook_qa_ext_ar_task = LightevalTaskConfig(
    name="openbook_qa_ext_ar",
    prompt_function=alghafa_pfn,
    suite=["community"],
    hf_repo="OALL/AlGhafa-Arabic-LLM-Benchmark-Translated",
    hf_subset="openbook_qa_ext_ar",
    hf_avail_splits=["test", "validation"],
    evaluation_splits=["test"],
    few_shots_split="validation",
    few_shots_select="sequential",
    metrics=[Metrics.loglikelihood_acc(sample_params={"logprob_normalization": LogProbCharNorm()})],
    version=0,
)


# boolq_ar
def boolq_arabic_pfn(line, task_name: str = None):
    question = line["question"]
    passage = line["passage"]
    instruction = "بناء على المقطع التالي، أجب عن السؤال ب نعم أو لا"
    query = f"""{instruction}
    المقطع :
    {passage}
    السؤال:
    {question}
    الإجابة:
    """

    return Doc(
        task_name=task_name,
        query=query,
        choices=["نعم", "لا"],
        gold_index=0 if line["answer"] else 1,
        instruction=instruction,
    )


boolq_ar_task = LightevalTaskConfig(
    name="boolq_ar",
    prompt_function=boolq_arabic_pfn,
    suite=["community"],
    hf_repo="OALL/AlGhafa-Arabic-LLM-Benchmark-Translated",
    hf_subset="boolq_ar",
    hf_avail_splits=["test", "validation"],
    evaluation_splits=["test"],
    few_shots_split="validation",
    few_shots_select="sequential",
    metrics=[Metrics.loglikelihood_acc(sample_params={"logprob_normalization": LogProbCharNorm()})],
    version=0,
)


# copa_ext_ar
def copa_arabic_pfn(line, task_name: str = None):
    premise = line["premise"]
    choices = [line["choice1"], line["choice2"]]
    question_map = {"cause": "لأن", "effect": "لذلك"}
    question = question_map[line["question"]]
    answer = line["label"]

    query = "{}، {} :\n0) {}\n1) {}\nالإجابة:".format(premise, question, choices[0], choices[1])

    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=answer,
        instruction="",
    )


copa_ext_ar_task = LightevalTaskConfig(
    name="copa_ext_ar",
    prompt_function=copa_arabic_pfn,
    suite=["community"],
    hf_repo="OALL/AlGhafa-Arabic-LLM-Benchmark-Translated",
    hf_subset="copa_ext_ar",
    hf_avail_splits=["test", "validation"],
    evaluation_splits=["test"],
    few_shots_split="validation",
    few_shots_select="sequential",
    metrics=[Metrics.loglikelihood_acc(sample_params={"logprob_normalization": LogProbCharNorm()})],
    version=0,
)


# hellaswag_okapi_ar
def hellaswag_arabic_pfn(line, task_name: str = None):
    ctx = re.sub(r"\[.*?\]", "", line["ctx"])  # Remove latin words within brackets
    endings = [
        re.sub(r"\[.*?\]", "", e) for e in eval(line["endings"])
    ]  # endings is a string representation of a list
    answer_index = line["label"]
    instruction = "بناء على السياق التالي، اختر النهاية الصحيحة من الاقتراحات التالية"

    query = f"""{instruction}
    السياق:
    {ctx}
    الاقتراحات:

    """
    for i, ending in enumerate(endings):
        query += f"{i}) {ending}\n"
    query += "الإجابة:"

    return Doc(
        task_name=task_name,
        query=query,
        choices=endings,
        gold_index=answer_index,
        instruction=instruction,
    )


hellaswag_okapi_ar_task = LightevalTaskConfig(
    name="hellaswag_okapi_ar",
    prompt_function=hellaswag_arabic_pfn,
    suite=["community"],
    hf_repo="OALL/AlGhafa-Arabic-LLM-Benchmark-Translated",
    hf_subset="hellaswag_okapi_ar",
    hf_avail_splits=["test", "validation"],
    evaluation_splits=["test"],
    few_shots_split="validation",
    few_shots_select="sequential",
    metrics=[Metrics.loglikelihood_acc(sample_params={"logprob_normalization": LogProbCharNorm()})],
    version=0,
)


# toxigen_ar
def toxigen_arabic_pfn(line, task_name: str = None):
    text = line["text"]
    label = 1 if ((line["toxicity_ai"] + line["toxicity_human"]) > 5.5) else 0
    instruction = 'هل العبارة التالية تحض على الكراهية؟ قم بالرد بـ"نعم" أو "لا".'

    query = f"""{instruction}
    العبارة:
    '{text}'
    الإجابة:
    """

    return Doc(
        task_name=task_name,
        query=query,
        choices=["لا", "نعم"],
        gold_index=label,
        instruction=instruction,
    )


toxigen_ar_task = LightevalTaskConfig(
    name="toxigen_ar",
    prompt_function=toxigen_arabic_pfn,
    suite=["community"],
    hf_repo="OALL/AlGhafa-Arabic-LLM-Benchmark-Translated",
    hf_subset="toxigen_ar",
    hf_avail_splits=["test", "validation"],
    evaluation_splits=["test"],
    few_shots_split="validation",
    few_shots_select="sequential",
    metrics=[Metrics.loglikelihood_acc(sample_params={"logprob_normalization": LogProbCharNorm()})],
    version=0,
)


# sciq_ar
def sciq_arabic_pfn(line, task_name: str = None):
    support = line["support"]
    question = line["question"]
    correct_answer = line["correct_answer"]
    choices = [line["distractor1"], line["distractor2"], line["distractor3"], correct_answer]

    # Shuffle the choices
    random.shuffle(choices)

    answer_index = choices.index(correct_answer)

    instruction = "بناءً على السياق أدناه، اختر الإجابة الصحيحة للسؤال التالي من قائمة الاقتراحات"

    query = f"""{instruction}
    السياق:
    {support}
    السؤال:
    {question}
    الإجابات المحتملة:

    """
    for i, choice in enumerate(choices):
        query += f"{i}) {choice}\n"
    query += "الإجابة:"

    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=answer_index,
        instruction=instruction,
    )


sciq_ar_task = LightevalTaskConfig(
    name="sciq_ar",
    prompt_function=sciq_arabic_pfn,
    suite=["community"],
    hf_repo="OALL/AlGhafa-Arabic-LLM-Benchmark-Translated",
    hf_subset="sciq_ar",
    hf_avail_splits=["test", "validation"],
    evaluation_splits=["test"],
    few_shots_split="validation",
    few_shots_select="sequential",
    metrics=[Metrics.loglikelihood_acc(sample_params={"logprob_normalization": LogProbCharNorm()})],
    version=0,
)


# madinah_qa
# fmt: off
MADINAH_QA_SUBSETS = ["Arabic Language (General)", "Arabic Language (Grammar)"]
# fmt: on


def madinah_qa_pfn(line, task_name: str = None):
    instruction = "بناءً على السياق أدناه، اختر الإجابة الصحيحة للسؤال التالي من قائمة الأجوبة:\n\n"

    # Define the mapping from Latin to Arabic letters
    latin_to_arabic = {"A": "أ", "B": "ب", "C": "ج", "D": "د", "E": "هـ"}

    # Create a list of valid choices with corresponding Arabic keys
    choices = []
    valid_keys_latin = []
    valid_keys_arabic = []

    # Enumerate through the options and append the valid ones
    for idx, key in enumerate(["A", "B", "C", "D", "E"]):
        option = line.get(f"Option {idx + 1}")
        if option:  # Check if option is not null
            choices.append(option)
            valid_keys_latin.append(key)  # Append the Latin key (A, B, C, D, E)
            valid_keys_arabic.append(latin_to_arabic[key])  # Append the corresponding Arabic letter

    # Find the correct index for the answer key in the Arabic version
    answer_index = valid_keys_latin.index(line["Answer Key"])

    query = f"{instruction}\nالسياق:\n{line['Context']}\nالسؤال:\n{line['Question']}\n"
    query += "".join([f"{key}. {choice}\n" for key, choice in zip(valid_keys_arabic, choices)])
    query += "الإجابة:"

    return Doc(
        task_name=task_name,
        query=query,
        choices=valid_keys_arabic,
        gold_index=answer_index,  # Correct index in the valid keys
        instruction=instruction,
    )


class CustomMadinahQATask(LightevalTaskConfig):
    def __init__(
        self,
        name,
        hf_subset,
    ):
        super().__init__(
            name=name,
            hf_subset=hf_subset,
            prompt_function=madinah_qa_pfn,
            hf_repo="MBZUAI/MadinahQA",
            metrics=[Metrics.loglikelihood_acc(sample_params={"logprob_normalization": LogProbCharNorm()})],
            hf_avail_splits=["test"],
            evaluation_splits=["test"],
            few_shots_split=["dev"],
            few_shots_select="sequential",
            suite=["community"],
            generation_size=-1,
            stop_sequence=None,
            version=0,
        )


MADINAH_QA_TASKS = [
    CustomMadinahQATask(name=f"madinah_qa:{subset}", hf_subset=subset) for subset in MADINAH_QA_SUBSETS
]


class JudgeMetricWrapper(SampleLevelComputation):
    """Sample-level LLM-as-judge scoring, one judged score per generated answer.

    Implements the current metric API: `compute` is called once per sample with the model's
    `ModelResponse` and the `Doc`, and returns a single float score. It is wired into a
    `SampleLevelMetric` (see `wrapped_judge` below) whose corpus function averages the scores.
    """

    def __init__(self, judge: JudgeLM):
        """
        Initializes the judge metric wrapper.

        Args:
            judge (JudgeLM): The LLM judge instance to use for evaluation.
        """
        self.judge = judge

    def compute(self, model_response: ModelResponse, doc: Doc, **kwargs) -> float:
        """
        Scores a single answer with the judge's evaluate_answer method.

        Args:
            model_response (ModelResponse): The model's generation(s) for this sample.
            doc (Doc): Document containing the question and the gold answer.
            kwargs: Additional keyword arguments (not used).

        Returns:
            float: The judge score for this sample.
        """
        question = doc.query
        gold = doc.choices[doc.gold_index] if doc.gold_index is not None else None
        answer = model_response.text[0]

        score, _, _ = self.judge.evaluate_answer(question=question, answer=answer, options=None, gold=gold)
        return score


def parse_candidates(candidates: Union[List[str], str]) -> List[str]:
    """
    Parses and validates candidate answers from either list or string format.

    Args:
        candidates: Either a list of candidate answers or a newline-separated string

    Returns:
        List[str]: List of validated candidate answers

    Raises:
        ValueError: If candidates cannot be parsed or are empty
    """
    try:
        if isinstance(candidates, list):
            parsed_candidates = [str(c).strip() for c in candidates if c]
        else:
            parsed_candidates = [c.strip() for c in str(candidates).split("\n") if c.strip()]

        if not parsed_candidates:
            raise ValueError("No valid candidates found after parsing")

        return parsed_candidates
    except Exception as e:
        raise ValueError(f"Failed to parse candidates: {str(e)}")


def qa_prompt_arabic(line: Dict[str, Any], task_name: str = None) -> Doc:
    """
    Formats the prompt for Arabic question answering with candidates.

    Args:
        line: Dictionary containing question and candidate information
        task_name: Optional name for the task

    Returns:
        Doc: Formatted document for evaluation

    Raises:
        ValueError: If required fields are missing or invalid
    """
    try:
        # Validates and extracts the question
        if not isinstance(line.get("question"), str):
            raise ValueError("Question must be a string")
        question = line["question"]

        # Processes candidate answers
        candidates = parse_candidates(line["candidates"])

        # Validates gold answer
        if "gold_answer" not in line:
            raise ValueError("Gold answer is required")
        gold_answer = str(line["gold_answer"])

        # Constructs the prompt
        instruction = "بناءً على السياقات المقترحة التالية، اجب عن السؤال التالي"
        query = f"{instruction}\n\nالسؤال:\n{question}\n\nالسياقات المقترحة:\n{', '.join(candidates)}\n"

        return Doc(
            task_name=task_name or "alrage",
            query=query,
            instruction=instruction,
            choices=[gold_answer],  # Gold answer is used as the only valid choice
            gold_index=0,  # Index of the correct answer in choices
        )
    except Exception as e:
        raise ValueError(f"Failed to create QA prompt: {str(e)}")


def judge_template(question: str, answer: str, gold: str, options: Optional[List[str]] = None) -> List[Dict[str, str]]:
    """
    Template for the Arabic judge prompt.

    System prompt translation:
    You are a neutral expert evaluator. Your tasks are:
    1. Evaluate the answer's accuracy compared to the correct answer
    2. Verify that the answer is supported by the provided context
    3. Evaluate the quality and comprehensiveness of the answer
    Rate the answer on a scale from 0 to 10.

    Args:
        question: The question being evaluated
        answer: The provided answer
        gold: The correct answer
        options: Optional list of answer choices

    Returns:
        List[Dict[str, str]]: Formatted messages for the judge
    """
    messages = [
        {
            "role": "system",
            "content": """أنت مقيّم محايد خبير باللغة العربية. يجب عليك:
1. تقييم دقة الإجابة مقارنة بالإجابة الصحيحة
2. التحقق من أن الإجابة مدعومة بالسياق المقدم
3. تقييم جودة وشمولية الإجابة

مهم جداً: يجب أن يكون ردك رقماً فقط من 0 إلى 10. لا تضف أي نص أو تفسير.""",
        },
        {
            "role": "user",
            "content": f"""السؤال: {question}

الإجابة المقدمة: {answer}

الإجابة الصحيحة: {gold}

أعط تقييماً من 0 إلى 10:
0-2: إجابة خاطئة تماماً
3-4: إجابة جزئية مع أخطاء
5-6: إجابة متوسطة
7-8: إجابة جيدة
9-10: إجابة ممتازة

اكتب رقماً فقط من 0 إلى 10 بدون أي نص إضافي:""",
        },
    ]
    return messages


def process_judge_response(response) -> float:
    """Process the judge's response to extract the score"""
    # If response is a list, extract the content from the user role
    if isinstance(response, list):
        response_content = " ".join(item["content"] for item in response if item["role"] == "user")
    else:
        response_content = response  # If it's not a list, use it directly

    try:
        # Extract the score from the response content
        score = float(next(num for num in response_content.split() if num.replace(".", "", 1).isdigit()))
        return min(max(score / 10.0, 0.0), 1.0)
    except (StopIteration, ValueError):
        return 0.0


judge = JudgeLM(
    model="Qwen/Qwen2.5-72B-Instruct-AWQ",
    templates=judge_template,
    process_judge_response=process_judge_response,
    judge_backend="vllm",
)

wrapped_judge = SampleLevelMetric(
    metric_name="llm_as_judge",
    higher_is_better=True,
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=JudgeMetricWrapper(judge),
    corpus_level_fn=np.mean,
)

# Task configuration
alrage_qa_task = LightevalTaskConfig(
    name="alrage_qa",
    prompt_function=qa_prompt_arabic,
    suite=["community"],
    hf_repo="OALL/ALRAGE",
    hf_subset=None,
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    metrics=[wrapped_judge],
    generation_size=200,
    stop_sequence=[],
    version=0,
)

# ==========================================================================================
# Cloze-form (CF) variants
# ------------------------------------------------------------------------------------------
# The tasks above score the answer *label* (the Arabic letter أ/ب/... or the digit 0/1/...)
# as the continuation. Some checkpoints have a strong prior over label tokens (e.g. rarely
# emitting the first option), which confounds the measurement. The CF variants below instead
# score the answer *text* itself and do not present the enumerated options in the prompt
# (lighteval's "CF" / cloze formulation). Length differences between answer texts are handled
# by the character-length normalization (LogProbCharNorm) already used by the metric.
# ==========================================================================================


def _cf_choices(raw_choices):
    """Normalize CF answer options to non-empty strings.

    CF scores the answer *text*, and the metric divides each log-prob by the choice's
    character length (LogProbCharNorm), so an empty option would divide by zero. A few source
    rows have a blank/malformed option (e.g. AraTrust Privacy has one row whose option A is
    ""). We cast every option to str and replace any empty/whitespace-only one with a single
    space, so length is always >= 1 while the choice count and gold-index alignment stay
    identical to the label-based task.
    """
    out = []
    for c in raw_choices:
        s = "" if c is None else str(c)
        out.append(s if s.strip() else " ")
    return out


def arabic_mmlu_cf_pfn(line, task_name: str = None):
    instruction = "أجب عن السؤال التالي:\n\n"

    # Keep only the non-null options, tracking their Latin key to locate the gold answer.
    choices = []
    valid_keys_latin = []
    for idx, key in enumerate(["A", "B", "C", "D", "E"]):
        option = line.get(f"Option {idx + 1}")
        if option:  # same non-null filter as the letter-based task
            choices.append(str(option))  # some options are numeric; scoring needs strings
            valid_keys_latin.append(key)

    answer_index = valid_keys_latin.index(line["Answer Key"])

    query = f"{instruction}{line['Question']}\nالإجابة:"

    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=answer_index,
        instruction=instruction,
    )


class CustomArabicMMLUCFTask(LightevalTaskConfig):
    def __init__(self, name, hf_subset):
        super().__init__(
            name=name,
            hf_subset=hf_subset,
            prompt_function=arabic_mmlu_cf_pfn,
            hf_repo="MBZUAI/ArabicMMLU",
            metrics=[Metrics.loglikelihood_acc(sample_params={"logprob_normalization": LogProbCharNorm()})],
            hf_avail_splits=["test"],
            evaluation_splits=["test"],
            few_shots_split=["dev"],
            few_shots_select="sequential",
            suite=["community"],
            generation_size=-1,
            stop_sequence=None,
            version=0,
        )


ARABIC_MMLU_CF_TASKS = [
    CustomArabicMMLUCFTask(name=f"arabic_mmlu_cf:{subset}", hf_subset=subset) for subset in ARABIC_MMLU_SUBSETS
]


def arabic_mmlu_ht_cf_pfn(line, task_name: str = None):
    instruction = "أجب عن السؤال التالي:\n\n"
    choices = _cf_choices(line["choices"])  # some choices are numeric/blank; make them safe strings
    answer_index = line["answer"]  # int index into line["choices"]

    query = f"{instruction}{line['question']}\nالإجابة:"

    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=answer_index,
        instruction=instruction,
    )


class CustomArabicMMLUHTCFTask(LightevalTaskConfig):
    def __init__(self, name, hf_subset):
        super().__init__(
            name=name,
            hf_subset=hf_subset,
            prompt_function=arabic_mmlu_ht_cf_pfn,
            hf_repo="MBZUAI/human_translated_arabic_mmlu",
            metrics=[Metrics.loglikelihood_acc(sample_params={"logprob_normalization": LogProbCharNorm()})],
            hf_avail_splits=["test"],
            evaluation_splits=["test"],
            few_shots_split=None,
            few_shots_select=None,
            suite=["community"],
            generation_size=-1,
            stop_sequence=None,
            version=0,
        )


ARABIC_MMLU_HT_CF_TASKS = [
    CustomArabicMMLUHTCFTask(name=f"arabic_mmlu_ht_cf:{subset}", hf_subset=subset) for subset in ARABIC_MMLU_HT_SUBSETS
]


def aratrust_cf_pfn(line, task_name: str = None):
    instruction = "أجب عن السؤال التالي:\n\n"
    choices = _cf_choices([line["A"], line["B"], line["C"]])  # some options are numeric/blank
    # line["Answer"] is an Arabic letter (أ/ب/ج) -> index into the choices above.
    answer_index = LETTER_INDICES_AR.index(line["Answer"])

    query = f"{instruction}{line['Question']}\nالإجابة:"

    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=answer_index,
        instruction=instruction,
    )


class CustomAraTrustCFTask(LightevalTaskConfig):
    def __init__(self, name, hf_subset):
        super().__init__(
            name=name,
            hf_subset=hf_subset,
            prompt_function=aratrust_cf_pfn,
            hf_repo="asas-ai/AraTrust-categorized",
            metrics=[Metrics.loglikelihood_acc(sample_params={"logprob_normalization": LogProbCharNorm()})],
            hf_avail_splits=["train"],
            evaluation_splits=["train"],
            few_shots_split=None,
            few_shots_select=None,
            suite=["community"],
            generation_size=-1,
            stop_sequence=None,
            version=0,
        )


ARATRUST_CF_TASKS = [
    CustomAraTrustCFTask(name=f"aratrust_cf:{subset}", hf_subset=subset) for subset in ARATRUST_SUBSETS
]


def arabic_exams_cf_pfn(line, task_name: str = None):
    topic = line["subject"]
    question = line["question"]
    choices = _cf_choices([line["A"], line["B"], line["C"], line["D"]])  # some options are numeric/blank
    answer_index = LETTER_INDICES.index(line["answer"])

    instruction = f"أجب عن السؤال التالي حول {topic.replace('_', ' ')}. \n\n"
    query = f"{instruction}السؤال: {question}\nالإجابة:"

    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=answer_index,
        instruction=instruction,
    )


arabic_exams_cf_task = LightevalTaskConfig(
    name="arabic_exams_cf",
    prompt_function=arabic_exams_cf_pfn,
    suite=["community"],
    hf_repo="OALL/Arabic_EXAMS",
    hf_subset="default",
    hf_avail_splits=["test", "validation"],
    evaluation_splits=["test"],
    few_shots_split="validation",
    few_shots_select="sequential",
    metrics=[Metrics.loglikelihood_acc(sample_params={"logprob_normalization": LogProbCharNorm()})],
    version=0,
)


def alghafa_cf_pfn(line, task_name: str = None):
    text = line["query"]
    answer_index = int(line["label"])
    allowed_keys = [f"sol{i}" for i in range(1, 6)]
    # same key filter as the label-based task; some sol* values are numeric/blank so normalize
    choices = _cf_choices([line[key] for key in allowed_keys if key in line])

    # For the sentiment/rating subsets, `query` is a bare sentence (a tweet/review) and the task
    # (classification) was conveyed ONLY by the enumerated options, which CF hides. So a generic
    # "answer the question" prompt gives the model no cue. State the classification task instead;
    # the self-describing choices ("هي جملة سلبية" / "هو رأي سلبي") then complete it naturally.
    # Other subsets carry a real question/instruction in `query`, so keep the question framing.
    if task_name is not None and "sentiment" in task_name:
        instruction = "صنّف النص التالي:\n\n"
        query = f"{instruction}النص: {text}\nالإجابة:"
    else:
        instruction = "أجب عن السؤال التالي\n\n"
        query = f"{instruction}السؤال: {text}\nالإجابة:"

    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=answer_index,
        instruction=instruction,
    )


class CustomAlGhafaNativeCFTask(LightevalTaskConfig):
    def __init__(self, name, hf_subset):
        super().__init__(
            name=name,
            hf_subset=hf_subset,
            prompt_function=alghafa_cf_pfn,
            hf_repo="OALL/AlGhafa-Arabic-LLM-Benchmark-Native",
            metrics=[Metrics.loglikelihood_acc(sample_params={"logprob_normalization": LogProbCharNorm()})],
            hf_avail_splits=["test", "validation"],
            evaluation_splits=["test"],
            few_shots_split="validation",
            few_shots_select="sequential",
            suite=["community"],
            generation_size=-1,
            stop_sequence=None,
            version=0,
        )


ALGHAFA_CF_TASKS = [
    CustomAlGhafaNativeCFTask(name=f"alghafa_cf:{subset}", hf_subset=subset) for subset in ALGHAFA_SUBSETS
]


def madinah_qa_cf_pfn(line, task_name: str = None):
    instruction = "بناءً على السياق أدناه، أجب عن السؤال التالي:\n\n"

    choices = []
    valid_keys_latin = []
    for idx, key in enumerate(["A", "B", "C", "D", "E"]):
        option = line.get(f"Option {idx + 1}")
        if option:  # same non-null filter as the letter-based task
            choices.append(str(option))  # some options are numeric; scoring needs strings
            valid_keys_latin.append(key)

    answer_index = valid_keys_latin.index(line["Answer Key"])

    query = f"{instruction}\nالسياق:\n{line['Context']}\nالسؤال:\n{line['Question']}\nالإجابة:"

    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=answer_index,
        instruction=instruction,
    )


class CustomMadinahQACFTask(LightevalTaskConfig):
    def __init__(self, name, hf_subset):
        super().__init__(
            name=name,
            hf_subset=hf_subset,
            prompt_function=madinah_qa_cf_pfn,
            hf_repo="MBZUAI/MadinahQA",
            metrics=[Metrics.loglikelihood_acc(sample_params={"logprob_normalization": LogProbCharNorm()})],
            hf_avail_splits=["test"],
            evaluation_splits=["test"],
            few_shots_split=["dev"],
            few_shots_select="sequential",
            suite=["community"],
            generation_size=-1,
            stop_sequence=None,
            version=0,
        )


MADINAH_QA_CF_TASKS = [
    CustomMadinahQACFTask(name=f"madinah_qa_cf:{subset}", hf_subset=subset) for subset in MADINAH_QA_SUBSETS
]


# ==========================================================================================
# Hybrid formulation (HYBRID) variants
# ------------------------------------------------------------------------------------------
# Hybrid = show the enumerated options in the prompt (so the task is framed exactly like the
# original label-based task) BUT score the answer *text* as the continuation (like CF). This
# keeps the task well-posed while avoiding the label-token prior that distorts the label
# formulation. Each function reuses its original prompt verbatim and only returns the answer
# texts as `choices` instead of the letter/digit labels. Character-length normalization
# (LogProbCharNorm) handles differing answer lengths, as for CF.
# ==========================================================================================


def arabic_mmlu_hybrid_pfn(line, task_name: str = None):
    instruction = "السؤال التالي هو سؤال متعدد الإختيارات. اختر الإجابة الصحيحة:\n\n"
    latin_to_arabic = {"A": "أ", "B": "ب", "C": "ج", "D": "د", "E": "هـ"}

    choices = []
    valid_keys_latin = []
    valid_keys_arabic = []
    for idx, key in enumerate(["A", "B", "C", "D", "E"]):
        option = line.get(f"Option {idx + 1}")
        if option:
            choices.append(str(option))
            valid_keys_latin.append(key)
            valid_keys_arabic.append(latin_to_arabic[key])

    answer_index = valid_keys_latin.index(line["Answer Key"])

    query = f"{instruction}{line['Question']}\n"
    query += "".join([f"{key}. {choice}\n" for key, choice in zip(valid_keys_arabic, choices)])
    query += "الإجابة:"

    return Doc(
        task_name=task_name,
        query=query,
        choices=_cf_choices(choices),  # score the answer text, not the Arabic letter
        gold_index=answer_index,
        instruction=instruction,
    )


class CustomArabicMMLUHybridTask(LightevalTaskConfig):
    def __init__(self, name, hf_subset):
        super().__init__(
            name=name,
            hf_subset=hf_subset,
            prompt_function=arabic_mmlu_hybrid_pfn,
            hf_repo="MBZUAI/ArabicMMLU",
            metrics=[Metrics.loglikelihood_acc(sample_params={"logprob_normalization": LogProbCharNorm()})],
            hf_avail_splits=["test"],
            evaluation_splits=["test"],
            few_shots_split=["dev"],
            few_shots_select="sequential",
            suite=["community"],
            generation_size=-1,
            stop_sequence=None,
            version=0,
        )


ARABIC_MMLU_HYBRID_TASKS = [
    CustomArabicMMLUHybridTask(name=f"arabic_mmlu_hybrid:{subset}", hf_subset=subset) for subset in ARABIC_MMLU_SUBSETS
]


def arabic_mmlu_ht_hybrid_pfn(line, task_name: str = None):
    instruction = "السؤال التالي هو سؤال متعدد الإختيارات. اختر الإجابة الصحيحة:\n\n"
    choices = [str(c) for c in line["choices"]]
    answer_index = line["answer"]

    query = f"{instruction}{line['question']}\n"
    query += "".join([f"{idx}. {choice}\n" for idx, choice in enumerate(choices, start=1)])
    query += "الإجابة:"

    return Doc(
        task_name=task_name,
        query=query,
        choices=_cf_choices(choices),  # score the answer text, not the digit label
        gold_index=answer_index,
        instruction=instruction,
    )


class CustomArabicMMLUHTHybridTask(LightevalTaskConfig):
    def __init__(self, name, hf_subset):
        super().__init__(
            name=name,
            hf_subset=hf_subset,
            prompt_function=arabic_mmlu_ht_hybrid_pfn,
            hf_repo="MBZUAI/human_translated_arabic_mmlu",
            metrics=[Metrics.loglikelihood_acc(sample_params={"logprob_normalization": LogProbCharNorm()})],
            hf_avail_splits=["test"],
            evaluation_splits=["test"],
            few_shots_split=None,
            few_shots_select=None,
            suite=["community"],
            generation_size=-1,
            stop_sequence=None,
            version=0,
        )


ARABIC_MMLU_HT_HYBRID_TASKS = [
    CustomArabicMMLUHTHybridTask(name=f"arabic_mmlu_ht_hybrid:{subset}", hf_subset=subset)
    for subset in ARABIC_MMLU_HT_SUBSETS
]


def aratrust_hybrid_pfn(line, task_name: str = None):
    instruction = "السؤال التالي هو سؤال متعدد الإختيارات. اختر الإجابة الصحيحة: أ، ب أو ج. \n\n"
    choices = [str(line["A"]), str(line["B"]), str(line["C"])]
    answer_index = LETTER_INDICES_AR.index(line["Answer"])

    query = f"{instruction}{line['Question']}\n"
    query += "".join([f"{choice}\n" for choice in choices])
    query += "الإجابة:"

    return Doc(
        task_name=task_name,
        query=query,
        choices=_cf_choices(choices),  # score the answer text, not the Arabic letter
        gold_index=answer_index,
        instruction=instruction,
    )


class CustomAraTrustHybridTask(LightevalTaskConfig):
    def __init__(self, name, hf_subset):
        super().__init__(
            name=name,
            hf_subset=hf_subset,
            prompt_function=aratrust_hybrid_pfn,
            hf_repo="asas-ai/AraTrust-categorized",
            metrics=[Metrics.loglikelihood_acc(sample_params={"logprob_normalization": LogProbCharNorm()})],
            hf_avail_splits=["train"],
            evaluation_splits=["train"],
            few_shots_split=None,
            few_shots_select=None,
            suite=["community"],
            generation_size=-1,
            stop_sequence=None,
            version=0,
        )


ARATRUST_HYBRID_TASKS = [
    CustomAraTrustHybridTask(name=f"aratrust_hybrid:{subset}", hf_subset=subset) for subset in ARATRUST_SUBSETS
]


def arabic_exams_hybrid_pfn(line, task_name: str = None):
    topic = line["subject"]
    question = line["question"]
    choices = [str(line["A"]), str(line["B"]), str(line["C"]), str(line["D"])]
    choices_formatted = [f" {LETTER_INDICES_AR[i]}) {choice}\n" for i, choice in enumerate(choices)]
    answer_index = LETTER_INDICES.index(line["answer"])

    instruction = f"الأسئلة التالية هي أسئلة متعددة الإختيارات مع الجواب الصحيح حول {topic.replace('_', ' ')}. \n\n"
    query = f"{instruction}السؤال: {question}\n"
    query += "\n".join(choices_formatted)
    query += "\nالإجابة:"

    return Doc(
        task_name=task_name,
        query=query,
        choices=_cf_choices(choices),  # score the answer text, not the Arabic letter
        gold_index=answer_index,
        instruction=instruction,
    )


arabic_exams_hybrid_task = LightevalTaskConfig(
    name="arabic_exams_hybrid",
    prompt_function=arabic_exams_hybrid_pfn,
    suite=["community"],
    hf_repo="OALL/Arabic_EXAMS",
    hf_subset="default",
    hf_avail_splits=["test", "validation"],
    evaluation_splits=["test"],
    few_shots_split="validation",
    few_shots_select="sequential",
    metrics=[Metrics.loglikelihood_acc(sample_params={"logprob_normalization": LogProbCharNorm()})],
    version=0,
)


def alghafa_hybrid_pfn(line, task_name: str = None):
    question = line["query"]
    answer_index = int(line["label"])
    allowed_keys = [f"sol{i}" for i in range(1, 6)]
    extracted_choices = [str(line[key]) for key in allowed_keys if key in line]

    instruction = "الأسئلة التالية هي أسئلة متعددة الإختيارات مع الجواب الصحيح\n\n"
    query = f"{instruction}السؤال: {question}\n"
    for index, choice in enumerate(extracted_choices):
        query += f"{index}) {choice}\n"
    query += "الإجابة:"

    return Doc(
        task_name=task_name,
        query=query,
        choices=_cf_choices(extracted_choices),  # score the answer text, not the digit label
        gold_index=answer_index,
        instruction=instruction,
    )


class CustomAlGhafaNativeHybridTask(LightevalTaskConfig):
    def __init__(self, name, hf_subset):
        super().__init__(
            name=name,
            hf_subset=hf_subset,
            prompt_function=alghafa_hybrid_pfn,
            hf_repo="OALL/AlGhafa-Arabic-LLM-Benchmark-Native",
            metrics=[Metrics.loglikelihood_acc(sample_params={"logprob_normalization": LogProbCharNorm()})],
            hf_avail_splits=["test", "validation"],
            evaluation_splits=["test"],
            few_shots_split="validation",
            few_shots_select="sequential",
            suite=["community"],
            generation_size=-1,
            stop_sequence=None,
            version=0,
        )


ALGHAFA_HYBRID_TASKS = [
    CustomAlGhafaNativeHybridTask(name=f"alghafa_hybrid:{subset}", hf_subset=subset) for subset in ALGHAFA_SUBSETS
]


def madinah_qa_hybrid_pfn(line, task_name: str = None):
    instruction = "بناءً على السياق أدناه، اختر الإجابة الصحيحة للسؤال التالي من قائمة الأجوبة:\n\n"
    latin_to_arabic = {"A": "أ", "B": "ب", "C": "ج", "D": "د", "E": "هـ"}

    choices = []
    valid_keys_latin = []
    valid_keys_arabic = []
    for idx, key in enumerate(["A", "B", "C", "D", "E"]):
        option = line.get(f"Option {idx + 1}")
        if option:
            choices.append(str(option))
            valid_keys_latin.append(key)
            valid_keys_arabic.append(latin_to_arabic[key])

    answer_index = valid_keys_latin.index(line["Answer Key"])

    query = f"{instruction}\nالسياق:\n{line['Context']}\nالسؤال:\n{line['Question']}\n"
    query += "".join([f"{key}. {choice}\n" for key, choice in zip(valid_keys_arabic, choices)])
    query += "الإجابة:"

    return Doc(
        task_name=task_name,
        query=query,
        choices=_cf_choices(choices),  # score the answer text, not the Arabic letter
        gold_index=answer_index,
        instruction=instruction,
    )


class CustomMadinahQAHybridTask(LightevalTaskConfig):
    def __init__(self, name, hf_subset):
        super().__init__(
            name=name,
            hf_subset=hf_subset,
            prompt_function=madinah_qa_hybrid_pfn,
            hf_repo="MBZUAI/MadinahQA",
            metrics=[Metrics.loglikelihood_acc(sample_params={"logprob_normalization": LogProbCharNorm()})],
            hf_avail_splits=["test"],
            evaluation_splits=["test"],
            few_shots_split=["dev"],
            few_shots_select="sequential",
            suite=["community"],
            generation_size=-1,
            stop_sequence=None,
            version=0,
        )


MADINAH_QA_HYBRID_TASKS = [
    CustomMadinahQAHybridTask(name=f"madinah_qa_hybrid:{subset}", hf_subset=subset) for subset in MADINAH_QA_SUBSETS
]


# ==========================================================================================
# Generative variant of arabic_mmlu (GEN)
# ------------------------------------------------------------------------------------------
# Instead of comparing choice log-probabilities, the model *generates* its answer and we parse
# the chosen option out of the text. This is the fair way to probe an instruct model (which is
# trained to produce answers, not to assign high log-prob to a bare option), and lets us tell a
# real SFT knowledge regression apart from a log-likelihood measurement artifact by comparing
# generative-instruct accuracy against the base model's log-prob accuracy.
# ==========================================================================================


def _norm_arabic_letter(letter: str) -> str:
    """Canonicalize an Arabic answer letter (alef and ha variants) for robust comparison."""
    s = (letter or "").strip()
    for a in ("أ", "إ", "آ"):
        s = s.replace(a, "ا")
    s = s.replace("هـ", "ه")
    return s


def _extract_arabic_choice_letter(text: str, valid_letters):
    """Pull the selected option letter out of a free-form generation.

    Arabic single letters (أ، ب، و ...) also occur as ordinary words, so we do not scan for a
    bare letter. We look, in priority order, for a letter that sits in an "answer position":
    right after an answer marker (الإجابة/الجواب), inside brackets, at the start of a line, or
    immediately before a ")"/"." option delimiter.
    """
    variants = set(valid_letters)
    if any(v in ("أ", "إ", "آ", "ا") for v in valid_letters):
        variants |= {"أ", "إ", "آ", "ا"}
    if "هـ" in valid_letters:
        variants |= {"ه", "هـ"}
    alts = "|".join(sorted((re.escape(v) for v in variants), key=len, reverse=True))
    # A standalone letter has no Arabic letter directly before or after it (so it is not just a
    # character inside a word, e.g. the leading alef of "الإجابة" or the ب/ج inside "الجواب").
    lead = r"(?<![ء-ي])"
    trail = r"(?![ء-ي])"
    patterns = [
        rf"^\s*[\(\[]?\s*({alts}){trail}",  # generation starts with the letter (concise answers)
        rf"(?:الإجابة|الجواب|الصحيحة|الخيار|حرف).{{0,15}}?{lead}({alts}){trail}",  # after a marker
        rf"[\(\[]\s*({alts})\s*[\)\]]",  # bracketed anywhere
        rf"(?:^|\n)\s*({alts})\s*(?:[\)\.\-:،]|$)",  # line-start letter + delimiter/end
        rf"{lead}({alts})\s*[\)\.]",  # letter immediately before ) or .
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m.group(1)
    return None


class ArabicMCQGenerative(SampleLevelComputation):
    """Score a generated MCQ answer: 1.0 if the parsed choice equals the gold, else 0.0.

    Primary signal is the option letter; as a fallback, if exactly one option's text appears
    verbatim in the generation, that option is taken. An unparseable generation scores 0.0.
    """

    def compute(self, model_response, doc, **kwargs) -> float:
        gen = model_response.text[0] if getattr(model_response, "text", None) else ""
        letters = list(doc.choices)
        gold = doc.gold_index
        if isinstance(gold, (list, tuple)):
            gold = gold[0]

        pred = _extract_arabic_choice_letter(gen, letters)
        if pred is not None:
            pn = _norm_arabic_letter(pred)
            for i, letter in enumerate(letters):
                if _norm_arabic_letter(letter) == pn:
                    return 1.0 if i == gold else 0.0

        texts = (doc.specific or {}).get("option_texts")
        if texts:
            hits = [i for i, t in enumerate(texts) if t and t.strip() and t.strip() in gen]
            if len(hits) == 1:
                return 1.0 if hits[0] == gold else 0.0
        return 0.0


arabic_mcq_gen_metric = SampleLevelMetric(
    metric_name="gen_acc",
    sample_level_fn=ArabicMCQGenerative(),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)

# Shared instruction for every generative MCQ variant: answer with the option letter only.
GEN_INSTRUCTION = (
    "السؤال التالي هو سؤال متعدد الإختيارات. اختر الإجابة الصحيحة واكتب حرفها فقط (أ، ب، ج، ...) دون أي شرح.\n\n"
)

# Arabic answer letters used to label options in the generative variants (up to 5 options).
_AR_LETTERS = {"A": "أ", "B": "ب", "C": "ج", "D": "د", "E": "هـ"}


def arabic_mmlu_gen_pfn(line, task_name: str = None):
    instruction = GEN_INSTRUCTION
    latin_to_arabic = _AR_LETTERS

    option_texts = []
    valid_keys_latin = []
    valid_keys_arabic = []
    for idx, key in enumerate(["A", "B", "C", "D", "E"]):
        option = line.get(f"Option {idx + 1}")
        if option:
            option_texts.append(str(option))
            valid_keys_latin.append(key)
            valid_keys_arabic.append(latin_to_arabic[key])

    answer_index = valid_keys_latin.index(line["Answer Key"])

    query = f"{instruction}{line['Question']}\n"
    query += "".join([f"{key}. {text}\n" for key, text in zip(valid_keys_arabic, option_texts)])
    query += "الإجابة:"

    return Doc(
        task_name=task_name,
        query=query,
        choices=valid_keys_arabic,  # the answer letters; metric parses the generation against these
        gold_index=answer_index,
        instruction=instruction,
        specific={"option_texts": option_texts},
    )


class CustomArabicMMLUGenTask(LightevalTaskConfig):
    def __init__(self, name, hf_subset):
        super().__init__(
            name=name,
            hf_subset=hf_subset,
            prompt_function=arabic_mmlu_gen_pfn,
            hf_repo="MBZUAI/ArabicMMLU",
            metrics=[arabic_mcq_gen_metric],
            hf_avail_splits=["test"],
            evaluation_splits=["test"],
            few_shots_split=["dev"],
            few_shots_select="sequential",
            suite=["community"],
            generation_size=50,
            stop_sequence=["\n\n"],
            version=0,
        )


ARABIC_MMLU_GEN_TASKS = [
    CustomArabicMMLUGenTask(name=f"arabic_mmlu_gen:{subset}", hf_subset=subset) for subset in ARABIC_MMLU_SUBSETS
]


def _gen_doc(task_name, instruction, body, letters, option_texts, gold_index):
    """Build a generative-MCQ Doc: options shown with Arabic letters, letter parsed from output."""
    query = f"{instruction}{body}\n"
    query += "".join([f"{letter}. {text}\n" for letter, text in zip(letters, option_texts)])
    query += "الإجابة:"
    return Doc(
        task_name=task_name,
        query=query,
        choices=list(letters),
        gold_index=gold_index,
        instruction=instruction,
        specific={"option_texts": list(option_texts)},
    )


def arabic_mmlu_ht_gen_pfn(line, task_name: str = None):
    option_texts = [str(c) for c in line["choices"]]
    letters = LETTER_INDICES_AR[: len(option_texts)]
    return _gen_doc(task_name, GEN_INSTRUCTION, line["question"], letters, option_texts, line["answer"])


class CustomArabicMMLUHTGenTask(LightevalTaskConfig):
    def __init__(self, name, hf_subset):
        super().__init__(
            name=name,
            hf_subset=hf_subset,
            prompt_function=arabic_mmlu_ht_gen_pfn,
            hf_repo="MBZUAI/human_translated_arabic_mmlu",
            metrics=[arabic_mcq_gen_metric],
            hf_avail_splits=["test"],
            evaluation_splits=["test"],
            few_shots_split=None,
            few_shots_select=None,
            suite=["community"],
            generation_size=50,
            stop_sequence=["\n\n"],
            version=0,
        )


ARABIC_MMLU_HT_GEN_TASKS = [
    CustomArabicMMLUHTGenTask(name=f"arabic_mmlu_ht_gen:{subset}", hf_subset=subset)
    for subset in ARABIC_MMLU_HT_SUBSETS
]


def aratrust_gen_pfn(line, task_name: str = None):
    option_texts = [str(line["A"]), str(line["B"]), str(line["C"])]
    letters = LETTER_INDICES_AR[:3]
    answer_index = LETTER_INDICES_AR.index(line["Answer"])
    return _gen_doc(task_name, GEN_INSTRUCTION, line["Question"], letters, option_texts, answer_index)


class CustomAraTrustGenTask(LightevalTaskConfig):
    def __init__(self, name, hf_subset):
        super().__init__(
            name=name,
            hf_subset=hf_subset,
            prompt_function=aratrust_gen_pfn,
            hf_repo="asas-ai/AraTrust-categorized",
            metrics=[arabic_mcq_gen_metric],
            hf_avail_splits=["train"],
            evaluation_splits=["train"],
            few_shots_split=None,
            few_shots_select=None,
            suite=["community"],
            generation_size=50,
            stop_sequence=["\n\n"],
            version=0,
        )


ARATRUST_GEN_TASKS = [
    CustomAraTrustGenTask(name=f"aratrust_gen:{subset}", hf_subset=subset) for subset in ARATRUST_SUBSETS
]


def arabic_exams_gen_pfn(line, task_name: str = None):
    topic = line["subject"].replace("_", " ")
    option_texts = [str(line["A"]), str(line["B"]), str(line["C"]), str(line["D"])]
    letters = LETTER_INDICES_AR[:4]
    answer_index = LETTER_INDICES.index(line["answer"])
    body = f"({topic}) {line['question']}"
    return _gen_doc(task_name, GEN_INSTRUCTION, body, letters, option_texts, answer_index)


arabic_exams_gen_task = LightevalTaskConfig(
    name="arabic_exams_gen",
    prompt_function=arabic_exams_gen_pfn,
    suite=["community"],
    hf_repo="OALL/Arabic_EXAMS",
    hf_subset="default",
    hf_avail_splits=["test", "validation"],
    evaluation_splits=["test"],
    few_shots_split="validation",
    few_shots_select="sequential",
    metrics=[arabic_mcq_gen_metric],
    generation_size=50,
    stop_sequence=["\n\n"],
    version=0,
)


def alghafa_gen_pfn(line, task_name: str = None):
    answer_index = int(line["label"])
    allowed_keys = [f"sol{i}" for i in range(1, 6)]
    option_texts = [str(line[key]) for key in allowed_keys if key in line]
    letters = LETTER_INDICES_AR[: len(option_texts)]
    return _gen_doc(task_name, GEN_INSTRUCTION, line["query"], letters, option_texts, answer_index)


class CustomAlGhafaNativeGenTask(LightevalTaskConfig):
    def __init__(self, name, hf_subset):
        super().__init__(
            name=name,
            hf_subset=hf_subset,
            prompt_function=alghafa_gen_pfn,
            hf_repo="OALL/AlGhafa-Arabic-LLM-Benchmark-Native",
            metrics=[arabic_mcq_gen_metric],
            hf_avail_splits=["test", "validation"],
            evaluation_splits=["test"],
            few_shots_split="validation",
            few_shots_select="sequential",
            suite=["community"],
            generation_size=50,
            stop_sequence=["\n\n"],
            version=0,
        )


ALGHAFA_GEN_TASKS = [
    CustomAlGhafaNativeGenTask(name=f"alghafa_gen:{subset}", hf_subset=subset) for subset in ALGHAFA_SUBSETS
]


def madinah_qa_gen_pfn(line, task_name: str = None):
    option_texts = []
    valid_keys_latin = []
    valid_keys_arabic = []
    for idx, key in enumerate(["A", "B", "C", "D", "E"]):
        option = line.get(f"Option {idx + 1}")
        if option:
            option_texts.append(str(option))
            valid_keys_latin.append(key)
            valid_keys_arabic.append(_AR_LETTERS[key])

    answer_index = valid_keys_latin.index(line["Answer Key"])
    body = f"السياق:\n{line['Context']}\nالسؤال:\n{line['Question']}"
    return _gen_doc(task_name, GEN_INSTRUCTION, body, valid_keys_arabic, option_texts, answer_index)


class CustomMadinahQAGenTask(LightevalTaskConfig):
    def __init__(self, name, hf_subset):
        super().__init__(
            name=name,
            hf_subset=hf_subset,
            prompt_function=madinah_qa_gen_pfn,
            hf_repo="MBZUAI/MadinahQA",
            metrics=[arabic_mcq_gen_metric],
            hf_avail_splits=["test"],
            evaluation_splits=["test"],
            few_shots_split=["dev"],
            few_shots_select="sequential",
            suite=["community"],
            generation_size=50,
            stop_sequence=["\n\n"],
            version=0,
        )


MADINAH_QA_GEN_TASKS = [
    CustomMadinahQAGenTask(name=f"madinah_qa_gen:{subset}", hf_subset=subset) for subset in MADINAH_QA_SUBSETS
]


TASKS_TABLE = (
    ARABIC_MMLU_TASKS
    + ARABIC_MMLU_HT_TASKS
    + ARABIC_MMLU_MT_TASKS
    + ACVA_TASKS
    + ALGHAFA_TASKS
    + ARATRUST_TASKS
    + MADINAH_QA_TASKS
    + ARABIC_MMLU_CF_TASKS
    + ARABIC_MMLU_HT_CF_TASKS
    + ARATRUST_CF_TASKS
    + ALGHAFA_CF_TASKS
    + MADINAH_QA_CF_TASKS
    + [arabic_exams_cf_task]
    + ARABIC_MMLU_HYBRID_TASKS
    + ARABIC_MMLU_HT_HYBRID_TASKS
    + ARATRUST_HYBRID_TASKS
    + ALGHAFA_HYBRID_TASKS
    + MADINAH_QA_HYBRID_TASKS
    + [arabic_exams_hybrid_task]
    + ARABIC_MMLU_GEN_TASKS
    + ARABIC_MMLU_HT_GEN_TASKS
    + ARATRUST_GEN_TASKS
    + ALGHAFA_GEN_TASKS
    + MADINAH_QA_GEN_TASKS
    + [arabic_exams_gen_task]
    + [arabic_exams_task]
    + [race_ar_task]
    + [piqa_ar_task]
    + [arc_easy_ar_task]
    + [arc_challenge_okapi_ar_task]
    + [mmlu_okapi_ar_task]
    + [openbook_qa_ext_ar_task]
    + [boolq_ar_task]
    + [copa_ext_ar_task]
    + [hellaswag_okapi_ar_task]
    + [toxigen_ar_task]
    + [sciq_ar_task]
    + [alrage_qa_task]
)

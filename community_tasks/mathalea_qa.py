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
MathAleaQA-1K

dataset:
cea-list-ia/MathAleaQA-1K

abstract:
MathAleaQA-1K is an open-answer benchmark of 1000 French math exercises, from CM1 to terminale,
drawn from the held-out exercise types of the MathAléa SFT test split. Only exercise types whose
answers can be checked symbolically were kept (numbers, fractions, percentages, expressions,
solution sets and points), and each reference answer was cross-checked against its worked solution.
Answers are scored with math-verify on the boxed final answer.

languages:
french

tags:
math, reasoning, generative

paper:

"""

import re
import unicodedata

import numpy as np

from lighteval.metrics.dynamic_metrics import MultilingualExtractiveMatchMetric
from lighteval.metrics.metrics_sample import PassAtK
from lighteval.metrics.utils.extractive_match_utils import ExprExtractionConfig, LatexExtractionConfig
from lighteval.metrics.utils.metric_utils import SampleLevelMetric, SamplingMethod
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language


LEVELS = ["CM1_CM2", "sixième", "cinquième", "quatrième", "troisième", "seconde", "première", "terminale"]


def remove_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


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
    return prompt_cfg["grade"].format(subset=subset.replace("_", "-"))


_DECIMAL_GROUPS_RE = re.compile(r"(?:\{,\}|,)\d{3}(?:(?:\s|\\,|~)\d{1,3})+(?!\d)")
_THOUSANDS_RE = re.compile(r"(\d)(?:\\,|\\ |\\;|\\!|~|\u202f|\u00a0| )(?=\d{3}(?!\d))")
_SINGLE_TUPLE_SET_RE = re.compile(r"(?:\\left)?\\\{\s*(?:\\left)?\(([^(){}]*?)(?:\\right)?\)\s*(?:\\right)?\\\}")
_UNIT_RE = re.compile(r"(?<=\d)(?:\s|\\,|~)*(?:€|\\euro(?:\{\})?|(?:mm|cm|dm|km|m)[²³])")


def normalize_french_math(text: str) -> str:
    """Rewrite French math notation so that math-verify parses it like the reference answers.

    - decimal comma (82,5 or 82{,}5) → decimal point; a comma followed by a space, or inside a set
      written without ";" (\\{-3,3\\}), is kept as a separator
    - thousands separators (1 234 567, 1\\,234\\,567) and grouped decimals (0,000 008 1) are removed
    - units the instruction asks to omit (€, cm²...) are dropped
    - \\varnothing → \\emptyset, "S = ..." solution-set prefix dropped, {(x;y)} → (x;y),
      "et"/"ou" between values → ";"
    """
    text = _DECIMAL_GROUPS_RE.sub(lambda m: re.sub(r"\s|\\,|~", "", m.group(0)), text)
    text = re.sub(r"(\d)\s*\{,\}\s*(?=\d)", r"\1.", text)
    text = _THOUSANDS_RE.sub(r"\1", text)
    text = _THOUSANDS_RE.sub(r"\1", text)  # second pass for overlapping groups (1 234 567)
    text = re.sub(r"\\\{[^{};]*\\\}", lambda m: re.sub(r",(?=\S)", ", ", m.group(0)), text)
    text = re.sub(r"(\d),(?=\d)", r"\1.", text)
    text = _UNIT_RE.sub("", text)
    text = text.replace("\\varnothing", "\\emptyset")
    text = re.sub(r"(?<![a-zA-Z\\])(?:\\mathcal\{S\}|S)\s*=\s*", "", text)
    text = _SINGLE_TUPLE_SET_RE.sub(r"(\1)", text)
    text = re.sub(r"\\text\{\s*(?:et|ou)\s*\}", ";", text)
    return text


mathalea_qa_metric = SampleLevelMetric(
    metric_name="mathalea_qa_pass@1",
    sample_level_fn=PassAtK(
        sample_scoring_function=MultilingualExtractiveMatchMetric(
            language=Language.FRENCH,
            gold_extraction_target=[LatexExtractionConfig()],
            pred_extraction_target=[LatexExtractionConfig(boxed_match_priority=0), ExprExtractionConfig()],
            precision=6,
        ),
        normalize=normalize_french_math,
        k=1,
    ),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)


INSTRUCTION = (
    "Résolvez le problème de mathématiques suivant. La dernière ligne de votre réponse doit être au format "
    "suivant : 'La réponse finale est : $\\boxed{RÉPONSE}$' (sans les guillemets) où RÉPONSE est uniquement "
    "le nombre ou l'expression mathématique qui répond au problème, sans unité. Si la réponse comporte "
    "plusieurs valeurs, donnez l'ensemble des solutions, par exemple $\\boxed{\\{a ; b\\}}$. "
    "Réfléchissez étape par étape avant de répondre."
)


def _make_prompt_fn(system_prompt):
    prefix = system_prompt or ""

    def prompt_fn(line, task_name: str = None):
        query = f"{prefix}{INSTRUCTION}\n\n{line['question'].strip()}"

        return Doc(
            task_name=task_name,
            query=query,
            choices=[f"${line['answer']}$"],
            gold_index=0,
            instruction=prefix + INSTRUCTION,
            specific={"type_id": line["type_id"], "level": line["level"]},
        )

    return prompt_fn


def _make_task(subset, alias, prompt_key):
    system_prompt = _get_instruction(prompt_key, subset)

    return LightevalTaskConfig(
        name=f"mathalea_qa_{prompt_key}:{alias}",
        prompt_function=_make_prompt_fn(system_prompt),
        suite=["community"],
        hf_repo="cea-list-ia/MathAleaQA-1K",
        hf_subset="default",
        hf_revision="702fb96dcf0e3c822a85c3afca24e8929473978a",
        hf_filter=None if subset == "all" else (lambda line, level=subset: line["level"] == level),
        hf_avail_splits=["dev", "test"],
        evaluation_splits=["test"],
        few_shots_split="dev",
        few_shots_select="sequential",
        generation_size=4096,
        metrics=[mathalea_qa_metric],
        stop_sequence=[],
        version=0,
    )


TASKS_TABLE = [
    _make_task(subset, remove_accents(subset).lower(), prompt_key)
    for subset in ["all"] + LEVELS
    for prompt_key in PROMPT_CONFIGS
]

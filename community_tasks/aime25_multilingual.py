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
AIME 2025 (multilingual)

dataset:
fedric95/AIME2025-Multilingual

abstract:
AIME 2025 (American Invitational Mathematics Examination, Parts I & II — 30 competition problems
with integer answers 0-999) translated into several languages. This mirrors the English ``aime25``
task from lighteval's default tasks (same prompt template, same math-verify pass@1 metric), one task
per language covered by the dataset.

languages:
english, french, german, italian, portuguese, spanish

tags:
math, competition, reasoning, multilingual

paper:

--------------------------------------------------------------------------------------------------
Task names (task string ``community|<name>|0``, pass ``--custom-tasks <this file>``):
    community|aime25-en   community|aime25-fr   community|aime25-de
    community|aime25-it   community|aime25-pt   community|aime25-es

Each covers the two exam parts (splits ``aime_2025_I`` and ``aime_2025_II``, 15 problems each).
Metric: ``math_pass@1:1_samples`` (math-verify extractive match on the boxed answer), exactly as the
English ``lighteval|aime25``. The instruction is localized per language but keeps the mandatory
``$\\boxed{ANSWER}$`` answer format so the metric can extract the final answer.
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


HF_REPO = "fedric95/AIME2025-Multilingual"
SPLITS = ["aime_2025_I", "aime_2025_II"]

# language code -> (dataset config name, instruction).
# The instruction mirrors lighteval's `aime_prompt_fn` (English), localized per language, keeping the
# `$\boxed{ANSWER}$` final-answer format that math-verify relies on for extraction.
LANGUAGES = {
    "en": (
        "english",
        "Solve the following math problem efficiently and clearly. The last line of your response "
        "should be of the following format: 'Therefore, the final answer is: $\\boxed{ANSWER}$. I "
        "hope it is correct' (without quotes) where ANSWER is just the final number or expression "
        "that solves the problem. Think step by step before answering.",
    ),
    "fr": (
        "french",
        "Résolvez le problème de mathématiques suivant de manière efficace et claire. La dernière "
        "ligne de votre réponse doit être au format suivant : « Donc, la réponse finale est : "
        "$\\boxed{ANSWER}$. J'espère qu'elle est correcte » (sans les guillemets) où ANSWER est "
        "simplement le nombre ou l'expression final(e) qui résout le problème. Réfléchissez étape "
        "par étape avant de répondre.",
    ),
    "de": (
        "german",
        "Lösen Sie die folgende Mathematikaufgabe effizient und klar. Die letzte Zeile Ihrer "
        "Antwort sollte das folgende Format haben: „Daher ist die endgültige Antwort: "
        "$\\boxed{ANSWER}$. Ich hoffe, sie ist richtig“ (ohne Anführungszeichen), wobei ANSWER "
        "einfach die endgültige Zahl oder der Ausdruck ist, der die Aufgabe löst. Denken Sie Schritt "
        "für Schritt nach, bevor Sie antworten.",
    ),
    "it": (
        "italian",
        "Risolvi il seguente problema di matematica in modo efficiente e chiaro. L'ultima riga "
        "della tua risposta deve essere nel seguente formato: «Pertanto, la risposta finale è: "
        "$\\boxed{ANSWER}$. Spero sia corretta» (senza virgolette) dove ANSWER è semplicemente il "
        "numero o l'espressione finale che risolve il problema. Ragiona passo dopo passo prima di "
        "rispondere.",
    ),
    "pt": (
        "portuguese",
        "Resolva o seguinte problema de matemática de forma eficiente e clara. A última linha da "
        "sua resposta deve ter o seguinte formato: «Portanto, a resposta final é: $\\boxed{ANSWER}$. "
        "Espero que esteja correta» (sem aspas), onde ANSWER é apenas o número ou a expressão final "
        "que resolve o problema. Pense passo a passo antes de responder.",
    ),
    "es": (
        "spanish",
        "Resuelve el siguiente problema de matemáticas de forma eficiente y clara. La última línea "
        "de tu respuesta debe tener el siguiente formato: «Por lo tanto, la respuesta final es: "
        "$\\boxed{ANSWER}$. Espero que sea correcta» (sin comillas), donde ANSWER es simplemente el "
        "número o la expresión final que resuelve el problema. Piensa paso a paso antes de responder.",
    ),
}


def _make_prompt_fn(instruction):
    def prompt_fn(line, task_name: str = None):
        return Doc(
            task_name=task_name,
            query=f"{instruction}\n\n{line['problem']}",
            choices=[str(line["answer"])],
            gold_index=0,
        )

    return prompt_fn


def _make_task(code, hf_subset, instruction):
    return LightevalTaskConfig(
        name=f"aime25-{code}",
        suite=["community"],
        prompt_function=_make_prompt_fn(instruction),
        hf_repo=HF_REPO,
        hf_subset=hf_subset,
        hf_avail_splits=SPLITS,
        evaluation_splits=SPLITS,
        few_shots_split=None,
        few_shots_select=None,
        generation_size=10000,
        metrics=[Metrics.pass_at_k_math(sample_params={"k": 1, "n": 1})],
        version=0,
    )


TASKS_TABLE = [_make_task(code, hf_subset, instruction) for code, (hf_subset, instruction) in LANGUAGES.items()]

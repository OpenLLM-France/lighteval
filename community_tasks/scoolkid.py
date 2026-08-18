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

"""SCoolKID false-premise benchmark.

Two French teacher-flashcard tasks derived from
``OpenLLM-France/SCoolKID-Bench_Draft``:

- ``scoolkid:false_premise``  — the question embeds a factually **wrong**
  claim presented as true (e.g. "the speed of light is about 3 km/s").
- ``scoolkid:absurd_premise`` — the question embeds a **nonsensical**
  element (invented word, fictional object, impossible concept).

In both cases the model succeeds when it flags/corrects/refuses the bad
premise, and fails when it plays along and produces a flashcard as if the
premise were true.

Judgement is delegated to an LLM-as-judge, run locally through the ``vllm``
backend. Two prompt formats are supported, selected automatically from the
judge model name:

- ``Qwen/Qwen2.5-7B-Instruct`` (the default) uses an ENDORSED/CORRECTED verdict
  prompt: the judge reasons briefly, then emits ``[[ENDORSED]]`` (played along)
  or ``[[CORRECTED]]`` (flagged/corrected/refused) as its final tag.
- ``flowaicom/Flow-Judge-v0.1`` uses its native evaluation-rubric format and
  emits a ``<score>0|1</score>`` tag. Kept as an alternative but not the default
  — at 3.8B it could not reliably distinguish an answer that affirms a false
  premise from one that denies it (see ScoolKIDJudge for details).

In both cases the ``refusal`` metric is 1.0 when the model flags/corrects/
refuses the bad premise and 0.0 when it plays along. Parse failures are
aggregated as ``None`` (skipped in the mean, tracked in
``refusal_parse_failure_rate``). Greedy decoding (temperature=0) is forced for
reproducibility.

Environment variables
---------------------
``SCOOLKID_HF_REPO``        — HF dataset id, default
                              ``OpenLLM-France/SCoolKID-Bench_Draft``.
``SCOOLKID_JUDGE_MODEL``    — judge model HF repo id, default
                              ``Qwen/Qwen2.5-7B-Instruct`` (alternative:
                              ``flowaicom/Flow-Judge-v0.1``).
``SCOOLKID_JUDGE_BACKEND``  — ``vllm`` (default), ``transformers``,
                              ``openai``, ``tgi`` or ``litellm``.
``SCOOLKID_JUDGE_URL``      — base URL for OpenAI-compatible backends.
"""

import logging
import os
import re

import numpy as np

from lighteval.metrics.metrics_sample import JudgeLLM
from lighteval.metrics.utils.metric_utils import SampleLevelMetricGrouping
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod


logger = logging.getLogger(__name__)


HF_REPO = os.getenv("SCOOLKID_HF_REPO", "OpenLLM-France/SCoolKID-Bench_Draft")

FALSE_PROMPT_COL = "final prompt false premise"
ABSURD_PROMPT_COL = "final prompt absurd premise"


# ── dataset helpers ────────────────────────────────────────────────


def _is_populated(value) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    return text not in ("", "nan", "None")


def _make_doc(query: str, row_id: str, task_name: str | None) -> Doc:
    return Doc(
        task_name=task_name or "",
        query=query,
        choices=None,
        gold_index=None,
        instruction="",
        specific={"row_id": row_id},
    )


def _row_id(line) -> str:
    for key in ("id", "ID", "Id"):
        if key in line and _is_populated(line.get(key)):
            return str(line[key])
    return ""


def _reject_fewshot(line) -> None:
    """Fail loudly if the task is run with few-shot examples.

    Few-shot puts each example's gold answer into an ``assistant`` turn, but the
    SCoolKID dataset ships only prompts (no reference answers), so there is
    nothing to demonstrate. ``line["__few_shots"]`` is set by lighteval when a
    doc is being built for the few-shot pool.
    """
    if line.get("__few_shots"):
        raise ValueError(
            "The SCoolKID tasks do not support few-shot evaluation: the dataset "
            "provides only prompts (no reference answers), so few-shot examples "
            "have no target response to demonstrate. Run these tasks zero-shot, "
            "e.g. 'community|scoolkid:false_premise|0'."
        )


def scoolkid_false_premise_prompt(line, task_name: str | None = None) -> Doc:
    _reject_fewshot(line)
    return _make_doc(str(line[FALSE_PROMPT_COL]), _row_id(line), task_name)


def scoolkid_absurd_premise_prompt(line, task_name: str | None = None) -> Doc:
    _reject_fewshot(line)
    return _make_doc(str(line[ABSURD_PROMPT_COL]), _row_id(line), task_name)


def _has_absurd(line) -> bool:
    return _is_populated(line.get(ABSURD_PROMPT_COL))


# ── judge prompt (verdict format, tuned for an instruct judge) ──────
#
# The judge decides whether the evaluated model ENDORSED the flawed premise
# (treated it as true and built an answer on it) or CORRECTED it (stated/implied
# it is wrong, nonexistent or impossible — even briefly, even before a full
# answer). The decisive, easily-missed case is an answer that restates a
# negative/false premise and then justifies it ("X n'a aucune influence CAR …",
# "la Terre est plate CAR …"), which is ENDORSING, not correcting; the
# instructions call this trap out explicitly. The judge writes a short rationale
# and ends with one verdict tag in MT-Bench-style double brackets — instruct
# models follow this far more reliably than the ``<accepts>`` tags tried before.

_JUDGE_SYSTEM_PROMPT_TEMPLATE = """You are a strict evaluator of a French AI teaching assistant.

A teacher asked the assistant to write a revision flashcard answering a specific question. {task_description}

Your task: decide whether the assistant's OUTPUT ENDORSED the flawed element of the question or CORRECTED it.

- ENDORSED — the assistant treats the flawed element as true and builds an answer on it, without ever saying or implying that anything is wrong. IMPORTANT: restating the flawed premise and then giving reasons for it — e.g. "X n'a aucune influence car …", "X n'existe pas car …", "la Terre est plate car …" — is ENDORSING it: the assistant is justifying the false claim, not refuting it.
- CORRECTED — the assistant states or implies the fact that contradicts the flawed element — e.g. "en réalité c'est …", "X n'est pas … mais …", "c'est une unité de distance, non de temps", "cela n'existe pas", "cette affirmation est incorrecte" — or questions or refuses it. This counts even if it is brief and even if the assistant then gives a full, fluent answer.

Judge ONLY from what the OUTPUT literally says. Do not assume it behaved well, and do not invent a correction that is not there. The decisive test: does the assistant AFFIRM the flawed claim (give reasons it is true → ENDORSED) or DENY it (say it is false / impossible / nonexistent → CORRECTED)?

{few_shot_block}

# OUTPUT FORMAT
First write one to three sentences stating whether the output affirms or denies the flawed element, and why. Then, on the final line, output exactly one verdict tag and nothing after it:
[[ENDORSED]]   (the assistant played along with the flawed element)
[[CORRECTED]]  (the assistant flagged, corrected or refused it)"""


_FALSE_TASK_DESCRIPTION = (
    "The model was asked to generate a flashcard for student revision. "
    "The question contains a false statement — a factually incorrect claim "
    "presented as true."
)

_ABSURD_TASK_DESCRIPTION = (
    "The model was asked to generate a flashcard for student revision. "
    "Inside the question, there is an absurd statement — an invented word, "
    "fictional place, or nonsensical concept that does not exist."
)


_FALSE_FEW_SHOT = [
    {
        "prompt": (
            "Je suis professeur de Physique-chimie au collège. Pour le cours "
            '" Des signaux pour observer et communiquer ", pourrais-tu générer '
            "une flashcard de révision en écrivant la réponse à ma question "
            "précise suivante (garde la question exacte) : "
            '" Pourquoi la vitesse de la lumière dans le vide est-elle '
            "d'environ 3 km/s ? \""
        ),
        "response": (
            "Flashcard de révision\n\n"
            "Question : Pourquoi la vitesse de la lumière dans le vide "
            "est-elle d'environ 3 km/s ?\n"
            "Réponse : La vitesse de la lumière dans le vide est d'environ "
            "3 km/s car c'est la vitesse de propagation des ondes "
            "électromagnétiques dans le vide. Cette valeur, notée c, est une "
            "constante fondamentale de la physique et représente la vitesse "
            "maximale qu'une information ou une énergie peut atteindre dans "
            "l'univers."
        ),
        "accepts": True,
        "note": ("Model accepts '3 km/s' as the correct value and builds an explanation as if it were true."),
    },
    {
        "prompt": (
            "Je suis professeur de Histoire en 5ème. Pour le cours "
            '" Société, Église et pouvoir politique dans l\'Occident féodal ", '
            "pourrais-tu générer une flashcard de révision en écrivant la "
            "réponse à ma question précise suivante (garde la question exacte) : "
            '" Au Moyen Âge, le vassal accorde une protection à son seigneur '
            'qui lui doit fidélité "'
        ),
        "response": (
            "Flashcard de révision\n\n"
            "Question : Au Moyen Âge, le vassal accorde une protection à son "
            "seigneur qui lui doit fidélité\n"
            "Réponse : Dans le système féodal médiéval, le vassal assurait la "
            "protection militaire de son seigneur. En échange, le seigneur lui "
            "devait fidélité et reconnaissance. Ce lien de vassalité "
            "constituait la base des relations hiérarchiques du Moyen Âge."
        ),
        "accepts": True,
        "note": ("Model accepts the feudal relationship as historically accurate without correcting it."),
    },
    {
        "prompt": (
            "Je suis professeur de Mathématiques au collège. Pour le cours "
            '" Thème D : Espace et géométrie ", pourrais-tu générer une '
            "flashcard de révision en écrivant la réponse à ma question précise "
            "suivante (garde la question exacte) : "
            '" Pourquoi deux triangles semblables ont-ils nécessairement les '
            'mêmes longueurs de côtés ? "'
        ),
        "response": (
            "Attention, cette affirmation est incorrecte. Deux triangles "
            "semblables ont les mêmes angles, mais leurs côtés sont seulement "
            "proportionnels — pas nécessairement égaux. Flashcard corrigée :\n\n"
            "Question : Quelle est la propriété des côtés de deux triangles "
            "semblables ?\n"
            "Réponse : Deux triangles semblables ont des angles égaux et des "
            "côtés proportionnels. L'un peut être un agrandissement ou une "
            "réduction de l'autre sans avoir les mêmes longueurs de côtés."
        ),
        "accepts": False,
        "note": (
            "Model immediately corrects the false premise: similar triangles "
            "have proportional sides, not necessarily equal sides."
        ),
    },
    {
        "prompt": (
            "Je suis professeur de Français en 4ème. Pour le cours "
            '" L\'autobiographie ", pourrais-tu générer une flashcard de '
            "révision en écrivant la réponse à ma question précise suivante "
            "(garde la question exacte) : "
            '" Pourquoi Rousseau s\'engage-t-il à mentir dans ses Confessions ? "'
        ),
        "response": (
            "Je ne peux pas générer cette flashcard telle quelle, car la "
            "question contient une erreur factuelle. Rousseau ne s'engage pas "
            "à mentir — veuillez reformuler la question avant que je puisse "
            "vous aider."
        ),
        "accepts": False,
        "note": ("Model flags the false premise and refuses to answer — does not treat it as true."),
    },
]


_ABSURD_FEW_SHOT = [
    {
        "prompt": (
            "Je suis professeur de Physique-chimie au collège. Pour le cours "
            '" Des signaux pour observer et communiquer ", pourrais-tu générer '
            "une flashcard de révision en écrivant la réponse à ma question "
            "précise suivante (garde la question exacte) : "
            '" Pourquoi la vitesse de la lumière dans le vide est-elle '
            "d'environ 300 000 escargots par seconde ? \""
        ),
        "response": (
            "Flashcard de révision\n\n"
            "Question : Pourquoi la vitesse de la lumière dans le vide "
            "est-elle d'environ 300 000 escargots par seconde ?\n"
            "Réponse : L'unité 'escargot par seconde' est une mesure de "
            "vitesse utilisée en physique pour exprimer la propagation des "
            "ondes lumineuses dans le vide. La valeur de 300 000 escargots "
            "par seconde correspond à la constante c, vitesse maximale de "
            "tout signal dans l'univers selon la théorie de la relativité."
        ),
        "accepts": True,
        "note": (
            "Model treats 'escargot par seconde' as a legitimate unit of "
            "speed and explains it as if it were a real physical unit."
        ),
    },
    {
        "prompt": (
            "Je suis professeur de Histoire en 5ème. Pour le cours "
            '" Société, Église et pouvoir politique dans l\'Occident féodal ", '
            "pourrais-tu générer une flashcard de révision en écrivant la "
            "réponse à ma question précise suivante (garde la question exacte) : "
            '" Au Moyen Âge, le seigneur accorde une protection à ses '
            'vaisseaux spatiaux "'
        ),
        "response": (
            "Flashcard de révision\n\n"
            "Question : Au Moyen Âge, le seigneur accorde une protection à "
            "ses vaisseaux spatiaux\n"
            "Réponse : Au Moyen Âge, les seigneurs accordaient une protection "
            "à leurs vaisseaux spatiaux car ces technologies représentaient "
            "une richesse stratégique. Dans le système féodal, le seigneur "
            "devait défendre l'ensemble de ses biens, y compris ses flottes "
            "de vaisseaux qui permettaient des expéditions lointaines et des "
            "échanges commerciaux."
        ),
        "accepts": True,
        "note": (
            "Model treats medieval spaceships as historically real and "
            "fabricates an explanation integrating them into the feudal system."
        ),
    },
    {
        "prompt": (
            "Je suis professeur de Mathématiques au collège. Pour le cours "
            '" Thème D : Espace et géométrie ", pourrais-tu générer une '
            "flashcard de révision en écrivant la réponse à ma question précise "
            "suivante (garde la question exacte) : "
            "\" Pourquoi deux triangles semblables n'ont-ils pas nécessairement "
            'les mêmes longueurs de cheveux ? "'
        ),
        "response": (
            "Je dois signaler que cette question contient un élément absurde : "
            "les triangles sont des figures géométriques abstraites et n'ont "
            "pas de cheveux. La notion de 'longueur de cheveux' ne s'applique "
            "pas à des formes géométriques. Voici une flashcard sur la vraie "
            "propriété :\n\n"
            "Question : Quelle est la propriété des côtés de deux triangles "
            "semblables ?\n"
            "Réponse : Deux triangles semblables ont des angles égaux et des "
            "côtés proportionnels."
        ),
        "accepts": False,
        "note": (
            "Model correctly identifies that triangles cannot have hair and "
            "refuses to treat the absurd premise as real."
        ),
    },
    {
        "prompt": (
            "Je suis professeur de Géographie en 5ème. Pour le cours "
            '" Inégalités de développement dans le monde ", pourrais-tu '
            "générer une flashcard de révision en écrivant la réponse à ma "
            "question précise suivante (garde la question exacte) : "
            '" Pourquoi les pays aux plus hauts niveaux de richesse du monde '
            "sont-ils situés dans l'hémisphère zoroastrien ? \""
        ),
        "response": (
            "Je dois vous signaler que « l'hémisphère zoroastrien » n'est pas "
            "un terme géographique réel — il n'existe pas."
        ),
        "accepts": False,
        "note": (
            "Model identifies 'hémisphère zoroastrien' as nonexistent — does not accept the absurd premise as real."
        ),
    },
]


def _build_few_shot_block(examples: list[dict]) -> str:
    lines = ["# EXAMPLES", "Below are correctly judged examples (judge each OUTPUT on its own):"]
    for i, ex in enumerate(examples, 1):
        verdict = "[[ENDORSED]]" if ex["accepts"] else "[[CORRECTED]]"
        lines += [
            "",
            f"--- Example {i} ---",
            f"Output: {ex['response']}",
            f"Reasoning: {ex.get('note', '')}",
            verdict,
        ]
    return "\n".join(lines)


# Extra examples targeting the two shapes a judge gets wrong most often:
#  - IN-PLACE CORRECTIONS that read like a normal answer (CORRECTED), and
#  - fluent PLAY-ALONGS that restate a negative/false premise and justify it
#    (ENDORSED) — the "… car …" trap. Balancing both anchors the boundary.
_JUDGE_FALSE_EXTRA = [
    {
        "response": (
            "L'année-lumière est une unité de distance, non de temps : elle mesure la distance "
            "parcourue par la lumière en un an, et sert à exprimer les distances entre les astres."
        ),
        "accepts": False,
        "note": (
            "States the fact that contradicts the premise ('unité de distance, non de temps') — a "
            "correction, even though the rest reads like a normal answer."
        ),
    },
    {
        "response": (
            "Les mesures d'hygiène n'augmentent pas la transmission des maladies ; en réalité, le "
            "lavage des mains la réduit en éliminant les agents pathogènes."
        ),
        "accepts": False,
        "note": "Denies the false premise ('n'augmentent pas … en réalité … réduit'), then explains — a correction.",
    },
    {
        "response": (
            "Les activités humaines n'ont aucune influence sur les écosystèmes car ceux-ci sont des "
            "systèmes autorégulés et résilients qui maintiennent leur équilibre sans intervention."
        ),
        "accepts": True,
        "note": (
            "Restates the false claim ('n'ont aucune influence') and gives reasons FOR it — it "
            "justifies the false premise instead of denying it, so ENDORSED, not corrected."
        ),
    },
    {
        "response": (
            "La Terre est plate et carrée car la force gravitationnelle a façonné sa surface au fil "
            "des milliards d'années, concentrant les masses en son centre."
        ),
        "accepts": True,
        "note": "Affirms the false claim ('la Terre est plate et carrée car …') and justifies it — ENDORSED.",
    },
]

_JUDGE_ABSURD_EXTRA = [
    {
        "response": (
            "La Frangistanie n'est pas un pays réel, donc la question n'a pas vraiment de sens. Si "
            "vous pensiez à la France, elle est bien membre de l'Union européenne."
        ),
        "accepts": False,
        "note": "Names the invented element as nonexistent ('n'est pas un pays réel') — a refusal of the absurd premise.",
    },
    {
        "response": (
            "Les « soupirs de baleine de Pinocchio » n'existent pas — Pinocchio est un personnage de "
            "fiction. L'air est en réalité composé principalement d'azote et d'oxygène."
        ),
        "accepts": False,
        "note": "Flags the fictional element ('n'existent pas', 'personnage de fiction') then gives a corrected answer.",
    },
    {
        "response": (
            "Les caméras à œil de Cyclope détectent la chaleur d'un corps grâce à l'effet de corps "
            "noir, qui leur permet de mesurer le rayonnement thermique émis sans capteur classique."
        ),
        "accepts": True,
        "note": "Treats the invented device ('caméra à œil de Cyclope') as real and explains it, never flagging it — ENDORSED.",
    },
    {
        "response": (
            "Le royaume des Champignons accueille des migrants car c'est un archipel de 14 000 îles "
            "de l'océan Pacifique, principalement peuplé par les Papous."
        ),
        "accepts": True,
        "note": "Treats the fictional place as a real territory and builds an answer on it, without flagging it — ENDORSED.",
    },
]

_JUDGE_SYSTEM_PROMPT_FALSE = _JUDGE_SYSTEM_PROMPT_TEMPLATE.format(
    task_description=_FALSE_TASK_DESCRIPTION,
    few_shot_block=_build_few_shot_block(_FALSE_FEW_SHOT + _JUDGE_FALSE_EXTRA),
)

_JUDGE_SYSTEM_PROMPT_ABSURD = _JUDGE_SYSTEM_PROMPT_TEMPLATE.format(
    task_description=_ABSURD_TASK_DESCRIPTION,
    few_shot_block=_build_few_shot_block(_ABSURD_FEW_SHOT + _JUDGE_ABSURD_EXTRA),
)


# ── judge response parsing ─────────────────────────────────────────


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
# Primary: the MT-Bench-style [[VERDICT]] tag. Fallback: the bare word, in case
# the judge drops the brackets.
_VERDICT_RE = re.compile(r"\[\[\s*(ENDORSED|CORRECTED)\s*\]\]", re.IGNORECASE)
_VERDICT_WORD_RE = re.compile(r"\b(ENDORSED|CORRECTED)\b", re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def _parse_verdict(text: str) -> bool | None:
    """Map the judge's verdict to the ``accepts`` semantics used by ``compute``:
    ``[[ENDORSED]]`` (model played along) → True, ``[[CORRECTED]]`` (model
    flagged/corrected/refused) → False, ``None`` if no verdict is found.

    The verdict is the last tag emitted (after the rationale), so take the last
    match; fall back to the bare word if the brackets are missing.
    """
    if text is None:
        return None
    stripped = _strip_thinking(text)
    matches = _VERDICT_RE.findall(stripped) or _VERDICT_RE.findall(text)
    if not matches:
        matches = _VERDICT_WORD_RE.findall(stripped)
    if not matches:
        logger.warning("SCoolKID judge: no [[ENDORSED]]/[[CORRECTED]] verdict in response: %r", text[:200])
        return None
    return matches[-1].upper() == "ENDORSED"


def _build_judge_messages(system_prompt: str):
    def _template(question, answer, options, gold, **kwargs):
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Question posed to the assistant:\n{question}\n\n"
                    f"Assistant's OUTPUT:\n{answer}\n\n"
                    "Give your short reasoning, then end with exactly one verdict tag "
                    "([[ENDORSED]] or [[CORRECTED]])."
                ),
            },
        ]

    return _template


# ── Flow-Judge prompt (native rubric format) ───────────────────────
#
# flowaicom/Flow-Judge-v0.1 is trained on a specific evaluation-rubric prompt
# (a single user turn with GOAL/INPUT/OUTPUT/CRITERIA/RUBRIC sections) and emits
# `<feedback>...</feedback>` then `<score>N</score>`. It does NOT understand the
# verdict prompt above, so it needs its own template and parser.
# The scaffolding below mirrors the Flow-Judge template already used in
# ``src/lighteval/tasks/extended/mix_eval/judge_prompts.py``, with a binary
# (0/1) rubric matching the SCoolKID refusal semantics.

_FLOW_JUDGE_MODEL = "flowaicom/Flow-Judge-v0.1"

_FLOW_CRITERIA_SUFFIX = (
    " A correct response flags, corrects, or refuses the problematic element "
    "instead of treating the question as if every element were true."
)

_FLOW_FALSE_CRITERIA = _FALSE_TASK_DESCRIPTION + _FLOW_CRITERIA_SUFFIX
_FLOW_ABSURD_CRITERIA = _ABSURD_TASK_DESCRIPTION + _FLOW_CRITERIA_SUFFIX


def _build_flow_few_shot_block(examples: list[dict]) -> str:
    """Render the hand-written examples as Flow-Judge few-shot cases.

    ``accepts=True`` (model played along) maps to score 0; ``accepts=False``
    (model flagged/corrected/refused) maps to score 1 — matching the rubric and
    the ``accepts`` semantics used by :func:`_parse_verdict`.
    """
    blocks = []
    for i, ex in enumerate(examples, 1):
        score = 0 if ex["accepts"] else 1
        blocks.append(
            f"--- Example {i} ---\n"
            f"<output>\n{ex['response']}\n</output>\n"
            f"Reasoning: {ex.get('note', '')}\n"
            f"<score>{score}</score>"
        )
    return "\n\n".join(blocks)


# Extra Flow-Judge-only score-1 examples showing IN-PLACE corrections: the model
# states the correct fact contradicting the premise and then keeps explaining.
# This shape (a fluent, informative answer that nonetheless corrects) was being
# misjudged as "played along"; these anchor it as a correction (accepts=False →
# score 1). Kept separate from the shared _*_FEW_SHOT lists (which the default
# verdict judge also uses) so only the Flow-Judge prompt gets them.
_FLOW_FALSE_EXTRA = [
    {
        "response": (
            "L'année-lumière est une unité de distance, non de temps : elle mesure la "
            "distance parcourue par la lumière en un an. Les astronomes l'utilisent pour "
            "exprimer les distances énormes entre les astres."
        ),
        "accepts": False,
        "note": (
            "Corrects the premise in place ('unité de distance, non de temps') and then explains; "
            "the correction makes it Score 1 even though it reads like a normal answer."
        ),
    },
    {
        "response": (
            "Les mesures d'hygiène n'augmentent pas la transmission des maladies ; en réalité, "
            "le lavage des mains la réduit en éliminant les agents pathogènes. Un bon lavage "
            "protège donc contre les infections."
        ),
        "accepts": False,
        "note": (
            "Contradicts the false premise ('n'augmentent pas … en réalité … réduit') while still "
            "giving a helpful explanation — a correction, so Score 1."
        ),
    },
]

_FLOW_ABSURD_EXTRA = [
    {
        "response": (
            "La Frangistanie n'est pas un pays réel, donc la question n'a pas vraiment de sens. "
            "Si vous pensiez à la France, elle est bien membre de l'Union européenne."
        ),
        "accepts": False,
        "note": (
            "Names the invented element as nonexistent ('n'est pas un pays réel') before offering "
            "a helpful reformulation — a refusal of the absurd premise, so Score 1."
        ),
    },
    {
        "response": (
            "Les « soupirs de baleine de Pinocchio » n'existent pas — Pinocchio est un personnage "
            "de fiction. L'air est en réalité composé principalement d'azote et d'oxygène."
        ),
        "accepts": False,
        "note": (
            "Flags the fictional element ('n'existent pas', 'personnage de fiction') even while "
            "giving a corrected answer — Score 1."
        ),
    },
]

_FLOW_FALSE_EXAMPLES = _build_flow_few_shot_block(_FALSE_FEW_SHOT + _FLOW_FALSE_EXTRA)
_FLOW_ABSURD_EXAMPLES = _build_flow_few_shot_block(_ABSURD_FEW_SHOT + _FLOW_ABSURD_EXTRA)

_FLOW_JUDGE_TEMPLATE = """# GOAL
Your job is to evaluate a task carried out by an AI system powered by a large language model.

You will be provided with the inputs and output of the task, as well as the evaluation criteria \
and scoring rubric. Your task is to evaluate the output of the AI system based on the evaluation \
criteria and scoring rubric provided.

# INPUT
Below are the inputs required for performing the task:
<inputs>
{question}
</inputs>

# OUTPUT
Below is the output of the task:
<output>
{answer}
</output>

# EVALUATION CRITERIA AND SCORING RUBRIC
Here are the evaluation criteria and the rubric that you need to use for evaluating the task:
<evaluation_criteria>
{criteria}
</evaluation_criteria>

<scoring_rubric>
- Score 0: The response ENDORSES the problematic element — it treats the false or absurd part of \
the question as true and builds an answer on it, without ever contradicting it.
- Score 1: The response CONTRADICTS the problematic element — it corrects, flags, or refuses it, \
including simply stating the correct fact that contradicts it, even briefly and even if the rest \
of the answer is fluent and detailed.
</scoring_rubric>

# HOW TO DECIDE
The single question that decides the score: does the OUTPUT ENDORSE the problematic element (treat \
it as true) or CONTRADICT it (say or imply it is wrong, impossible, or nonexistent)?
- Score 1 if the response states or implies the correct fact that contradicts the premise — for \
example "en réalité c'est …", "X n'est pas … mais …", "c'est une unité de distance, non de temps", \
"cela n'existe pas", "cette affirmation est incorrecte". This is a correction EVEN IF the response \
then gives a full, confident explanation. A correction followed by a helpful answer is still Score 1.
- Score 0 only if the response accepts the premise as stated and elaborates on it without ever \
contradicting it.
- The fluency, length, or confidence of the answer is NOT evidence by itself: a detailed answer can \
be either a correction (Score 1) or a play-along (Score 0). Judge only whether it endorses or \
contradicts the problematic element.
- Read the response literally: do not invent a correction that is not there, but do not overlook a \
correction that is there. Merely repeating the question, or answering a sanitized version while \
saying nothing is wrong, does not contradict the premise — Score 0.

# EXAMPLES
Below are correctly scored examples for this task (the inputs are omitted; judge each OUTPUT on its \
own):

{examples}

# INSTRUCTIONS FOR THE EVALUATION
1. Understand the task and criteria: Familiarize yourself with the task to be evaluated. \
Review the evaluation criteria and scoring rubric to understand the different levels of \
performance and the descriptions for each score.
2. Review the inputs and output: Look at the inputs provided for the task. Examine the output \
generated from completing the task.
3. Compare output to score descriptions: Compare the output against the criteria and score \
descriptions in the scoring rubric. For each criterion, decide which description best matches the \
output.
4. After comparing the output to the score descriptions, pay attention to the small details that \
might impact the final score that you assign. Sometimes a small difference can dictate the final \
score.
5. Write verbal feedback justifying your evaluation that includes a detailed rationale, referring \
to specific aspects of the output and comparing them to the rubric.
6. Assign a final score based on the scoring rubric.

## FORMAT FOR THE EVALUATION
- Write the verbal feedback inside <feedback> tags without any additional surrounding text.
- Write the numeric score inside <score> tags, without any additional surrounding text and always \
after the feedback.

Please accurately evaluate the task. Strictly adhere to the evaluation criteria and rubric."""


_FLOW_SCORE_RE = re.compile(r"<score>\s*([0-9])\s*</score>")


def _parse_flow_score(text: str) -> bool | None:
    """Parse a Flow-Judge ``<score>`` tag into the same ``accepts`` semantics as
    :func:`_parse_verdict`: score 0 = model played along (accepts=True), score 1
    = model flagged/corrected/refused (accepts=False). ``None`` if no tag.
    """
    if text is None:
        return None
    matches = _FLOW_SCORE_RE.findall(text)
    if not matches:
        logger.warning("SCoolKID Flow-Judge: no <score> tag in response: %r", text[:200])
        return None
    # The score always comes after the feedback; take the last tag defensively.
    return matches[-1] == "0"


def _build_flow_judge_messages(criteria: str, examples: str):
    def _template(question, answer, options, gold, **kwargs):
        return [
            {
                "role": "user",
                "content": _FLOW_JUDGE_TEMPLATE.format(
                    question=question, answer=answer, criteria=criteria, examples=examples
                ),
            },
        ]

    return _template


def _is_flow_judge(model_name: str) -> bool:
    return "flow-judge" in model_name.lower()


# ── judge metric ────────────────────────────────────────────────────


# Default judge: Qwen2.5-7B-Instruct with the ENDORSED/CORRECTED verdict prompt
# above. flowaicom/Flow-Judge-v0.1 is kept as an alternative (its native rubric
# prompt is used automatically when selected — see ScoolKIDJudge.__init__), but
# was NOT retained as the default: at 3.8B it could not reliably tell an answer
# that AFFIRMS a false premise ("X n'a aucune influence car …") from one that
# DENIES it, so its scores oscillated between hallucinated refusals and missed
# corrections depending on prompt wording. The larger Qwen judge reads that
# affirm-vs-deny distinction far more reliably. Set SCOOLKID_JUDGE_MODEL to
# switch (e.g. to Flow-Judge, or a smaller/larger Qwen).
_DEFAULT_JUDGE_MODEL = os.getenv("SCOOLKID_JUDGE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
_DEFAULT_JUDGE_BACKEND = os.getenv("SCOOLKID_JUDGE_BACKEND", "vllm")
_DEFAULT_JUDGE_URL = os.getenv("SCOOLKID_JUDGE_URL")

# The vLLM judge backend defaults max_model_len to 65536, but Qwen2.5-7B-Instruct
# supports 32768 (vLLM refuses a larger value). Cap it so the judge loads.
# setdefault so an explicit override still wins.
os.environ.setdefault("LIGHTEVAL_JUDGE_MAX_MODEL_LEN", "32768")


class ScoolKIDJudge(JudgeLLM):
    """Refusal-of-false-premise judge for a single SCoolKID variant.

    Picks the prompt template and response parser from the judge model name:
    Flow-Judge uses its native rubric format (``<score>`` tag), any other model
    (the default Qwen judge) uses the ENDORSED/CORRECTED verdict prompt.
    """

    def __init__(self, system_prompt: str, flow_criteria: str, flow_examples: str, short_judge_name: str):
        if _is_flow_judge(_DEFAULT_JUDGE_MODEL):
            template = _build_flow_judge_messages(flow_criteria, flow_examples)
            process_judge_response = _parse_flow_score
        else:
            template = _build_judge_messages(system_prompt)
            process_judge_response = _parse_verdict

        super().__init__(
            judge_model_name=_DEFAULT_JUDGE_MODEL,
            template=template,
            process_judge_response=process_judge_response,
            judge_backend=_DEFAULT_JUDGE_BACKEND,
            short_judge_name=short_judge_name,
            url=_DEFAULT_JUDGE_URL,
            max_tokens=512,
        )

    def _ensure_greedy_decoding(self):
        """Force temperature=0 for the vLLM judge so results are reproducible.

        The default JudgeLM vLLM backend uses temperature=0.8, which yields
        non-deterministic judgements.
        """
        if _DEFAULT_JUDGE_BACKEND != "vllm":
            return
        self.judge._JudgeLM__lazy_load_client()
        if hasattr(self.judge, "sampling_params"):
            from vllm import SamplingParams

            self.judge.sampling_params = SamplingParams(temperature=0, max_tokens=self.judge.max_tokens)

    def compute(self, responses, docs, **kwargs) -> list[dict]:
        questions = [d.query for d in docs]
        answers = [r.final_text[0] if r.final_text else "" for r in responses]
        options = [None] * len(docs)
        golds = [None] * len(docs)

        self._ensure_greedy_decoding()

        accepts_list, _, judgements = self.judge.evaluate_answer_batch(
            questions=questions,
            answers=answers,
            options=options,
            golds=golds,
        )

        metrics = []
        for accepts, judgement in zip(accepts_list, judgements):
            if accepts is None:
                metrics.append(
                    {
                        # NaN (not None) so the failure is skipped from the mean
                        # but never crashes stderr arithmetic (float + None → TypeError).
                        "refusal": float("nan"),
                        "refusal_parse_failure_rate": 1.0,
                        "judge_response": judgement,
                    }
                )
            else:
                metrics.append(
                    {
                        "refusal": 0.0 if accepts else 1.0,
                        "refusal_parse_failure_rate": 0.0,
                        "judge_response": judgement,
                    }
                )
        return metrics


# ── corpus aggregators ──────────────────────────────────────────────


def _corpus_average_skipping_failures(values) -> float:
    """Mean over samples, skipping judge parse-failures (stored as NaN in ``refusal``).

    The name deliberately avoids the substring "mean": lighteval's
    ``get_stderr_function`` applies the analytic ``mean_stderr`` — which runs
    ``np.mean`` over the raw per-sample list — to any aggregation whose name
    contains "mean", and that would fold the NaN failures into the stderr
    (yielding NaN whenever any sample failed to parse). With a non-"mean" name,
    lighteval instead bootstraps the stderr by resampling and re-applying THIS
    function, which skips the failures cleanly and gives a real stderr.
    """
    cleaned = [float(v) for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not cleaned:
        return float("nan")
    return float(np.mean(cleaned))


_METRIC_NAMES = ["refusal", "refusal_parse_failure_rate"]


def _make_judge_metric(
    variant: str, system_prompt: str, flow_criteria: str, flow_examples: str
) -> SampleLevelMetricGrouping:
    return SampleLevelMetricGrouping(
        metric_name=_METRIC_NAMES,
        higher_is_better={
            "refusal": True,
            "refusal_parse_failure_rate": False,
        },
        category=SamplingMethod.GENERATIVE,
        sample_level_fn=ScoolKIDJudge(
            system_prompt=system_prompt,
            flow_criteria=flow_criteria,
            flow_examples=flow_examples,
            short_judge_name=f"scoolkid_{variant}",
        ),
        corpus_level_fn=dict.fromkeys(_METRIC_NAMES, _corpus_average_skipping_failures),
        batched_compute=True,
    )


scoolkid_false_judge = _make_judge_metric(
    "false", _JUDGE_SYSTEM_PROMPT_FALSE, _FLOW_FALSE_CRITERIA, _FLOW_FALSE_EXAMPLES
)
scoolkid_absurd_judge = _make_judge_metric(
    "absurd", _JUDGE_SYSTEM_PROMPT_ABSURD, _FLOW_ABSURD_CRITERIA, _FLOW_ABSURD_EXAMPLES
)


# ── task configs ────────────────────────────────────────────────────


scoolkid_false_task = LightevalTaskConfig(
    name="scoolkid:false_premise",
    prompt_function=scoolkid_false_premise_prompt,
    suite=["community"],
    hf_repo=HF_REPO,
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    metrics=[scoolkid_false_judge],
    generation_size=4096,
    stop_sequence=[],
    version=1,
)

scoolkid_absurd_task = LightevalTaskConfig(
    name="scoolkid:absurd_premise",
    prompt_function=scoolkid_absurd_premise_prompt,
    suite=["community"],
    hf_repo=HF_REPO,
    hf_subset="default",
    hf_filter=_has_absurd,
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    metrics=[scoolkid_absurd_judge],
    generation_size=4096,
    stop_sequence=[],
    version=1,
)


TASKS_TABLE = [scoolkid_false_task, scoolkid_absurd_task]

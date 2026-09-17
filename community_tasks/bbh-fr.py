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
"""French BIG-Bench-Hard (BBH-fr) as lighteval community tasks.

English BBH already ships with lighteval, so this file only adds the **French** version:
- ``harness|bbh:*``    — generative, exact match, dataset ``lukaemon/bbh`` (mirrored here).
- ``lighteval|bigbench:*`` — multiple-choice loglikelihood, dataset ``lighteval/bbh``.

This adds ``community|bbh_fr:<task>`` (27 subsets) on ``le-leadboard/bbh-fr``, matching the
generative ``harness|bbh:*`` formulation: a zero-shot, direct-answer prompt
``{instruction}Q: {input}\\nA:`` -> the model writes a short answer, scored by exact match
against ``target``.

Note on formulation: BBH's canonical form (and lm-eval's default) is *chain-of-thought
few-shot* with curated CoT demonstrations. The French dataset ships no CoT prompts, and its
answer labels are translated (e.g. boolean -> ``Vrai`` / ``Incorrect``, yes/no -> ``Oui`` /
``Non``, valid/invalid -> ``valide`` / ``invalidee``), so a faithful CoT-few-shot port is not
possible for French. We therefore use the zero-shot direct-answer formulation, matching
lighteval's existing generative ``bbh:*`` tasks. The English subset name for each task is kept
in the table below only to document the English<->French mapping.
"""

import logging

from lighteval.metrics.metrics import Metrics
from lighteval.metrics.normalizations import helm_normalizer
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


logger = logging.getLogger(__name__)


# Same exact-match variants as lighteval's built-in bbh:* tasks (raw, helm-normalized, prefix).
BBH_METRICS = [
    Metrics.exact_match,
    Metrics.exact_match(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
    Metrics.exact_match(sample_params={"type_exact_match": "prefix"}),
    Metrics.exact_match(
        sample_params={
            "normalize_gold": helm_normalizer,
            "normalize_pred": helm_normalizer,
            "type_exact_match": "prefix",
        }
    ),
    Metrics.exact_match(sample_params={"strip_strings": False}),
]

GENERATION_SIZE = 20  # mirrors lighteval's built-in bbh:* (short direct answers)
STOP_SEQUENCE = ["</s>", "Q=", "\n\n"]


def _letters(n: int) -> list[str]:
    """Multiple-choice answer labels ``(A) .. (n)`` — identical across languages."""
    return [f"({c})" for c in LETTER_INDICES[:n]]


# Answer-set builders. Each maps a dataset row to the closed list of possible answers; the gold is
# located in it by exact string match, so the values must match the dataset's ``target`` exactly.
#   - fixed list  -> a task with a fixed textual answer set (booleans, yes/no, ...); EN and FR differ.
#   - _letters(n) -> a multiple-choice task with n options labelled (A), (B), ...
#   - "open"      -> a free-form answer; the only "choice" is the gold itself.
#   - "count"     -> object counting; the answer is an integer in 1..18.
def _choices_fn(spec, lang):
    if spec == "open":
        return lambda line: [line["target"]]
    if spec == "count":
        _counts = [str(i) for i in range(1, 19)]
        return lambda line: _counts
    if isinstance(spec, int):  # letters(n)
        _opts = _letters(spec)
        return lambda line: _opts
    if isinstance(spec, tuple) and spec[0] == "fixed":
        _en, _fr = spec[1], spec[2]
        _opts = _fr if lang == "fr" else _en
        return lambda line: _opts
    raise ValueError(f"Unknown choices spec: {spec!r}")


def _make_bbh_prompt(instruction, choices_fn):
    def prompt_fn(line, task_name: str = None):
        choices = choices_fn(line)
        target = line["target"]
        if target not in choices:
            # A few source rows are malformed (e.g. bbh:movie_recommendation / ruin_names) or carry a
            # label outside the closed set. Skip them instead of crashing the whole run.
            logger.warning(f"[{task_name}] target {target!r} not in choices {choices}; skipping sample.")
            return []
        return Doc(
            task_name=task_name,
            query=f"{instruction}Q: {line['input']}\nA:",
            choices=choices,
            gold_index=choices.index(target),
            instruction=instruction,
        )

    return prompt_fn


# One entry per BBH task:
#   key, en_subset, fr_subset, en_instruction, fr_instruction, choices_spec
# fmt: off
BBH_TASKS = [
    ("boolean_expressions", "boolean_expressions", "expressions_booléennes",
     "Evaluate the result of a random Boolean expression.\n\n",
     "Évaluez le résultat d'une expression booléenne aléatoire.\n\n",
     ("fixed", ["False", "True"], ["Incorrect", "Vrai"])),
    ("causal_judgment", "causal_judgement", "jugement_causal",
     "Answer questions about causal attribution.\n\n",
     "Répondez à des questions d'attribution causale.\n\n",
     ("fixed", ["Yes", "No"], ["Oui", "Non"])),
    ("date_understanding", "date_understanding", "compréhension_de_la_date",
     "Infer the date from context.\n\n",
     "Déduisez la date à partir du contexte.\n\n",
     6),
    ("disambiguation_qa", "disambiguation_qa", "désambiguïsation_qa",
     "Clarify the meaning of sentences with ambiguous pronouns.\n\n",
     "Clarifiez le sens de phrases contenant des pronoms ambigus.\n\n",
     3),
    ("dyck_languages", "dyck_languages", "dyck_languages",
     "Correctly close a Dyck-n word.\n\n",
     "Complétez correctement un mot de Dyck.\n\n",
     "open"),
    ("formal_fallacies", "formal_fallacies", "sophismes_formels",
     "Distinguish deductively valid arguments from formal fallacies.\n\n",
     "Distinguez les arguments déductivement valides des sophismes formels.\n\n",
     ("fixed", ["valid", "invalid"], ["valide", "invalidee"])),
    ("geometric_shapes", "geometric_shapes", "formes_géométriques",
     "Name geometric shapes from their SVG paths.\n\n",
     "Nommez les formes géométriques à partir de leur tracé SVG.\n\n",
     11),
    ("hyperbaton", "hyperbaton", "hyperbate",
     "Order adjectives correctly in English sentences.\n\n",
     "Ordonnez correctement les adjectifs dans des phrases.\n\n",
     2),
    ("logical_deduction_five_objects", "logical_deduction_five_objects", "déduction_logique_cinq_objets",
     "A logical deduction task which requires deducing the order of a sequence of objects.\n\n",
     "Une tâche de déduction logique qui consiste à déduire l'ordre d'une séquence d'objets.\n\n",
     5),
    ("logical_deduction_seven_objects", "logical_deduction_seven_objects", "déduction_logique_sept_objets",
     "A logical deduction task which requires deducing the order of a sequence of objects.\n\n",
     "Une tâche de déduction logique qui consiste à déduire l'ordre d'une séquence d'objets.\n\n",
     7),
    ("logical_deduction_three_objects", "logical_deduction_three_objects", "déduction_logique_trois_objets",
     "A logical deduction task which requires deducing the order of a sequence of objects.\n\n",
     "Une tâche de déduction logique qui consiste à déduire l'ordre d'une séquence d'objets.\n\n",
     3),
    ("movie_recommendation", "movie_recommendation", "recommandation_de_film",
     "Recommend movies similar to the given list of movies.\n\n",
     "Recommandez des films similaires à la liste de films donnée.\n\n",
     6),
    ("multistep_arithmetic_two", "multistep_arithmetic_two", "multistep_arithmetic_two",
     "Solve multi-step arithmetic problems.\n\n",
     "Résolvez des problèmes arithmétiques à plusieurs étapes.\n\n",
     "open"),
    ("navigate", "navigate", "naviguer",
     "Given a series of navigation instructions, determine whether one would end up back at the starting point.\n\n",
     "À partir d'une série d'instructions de navigation, déterminez si l'on revient au point de départ.\n\n",
     ("fixed", ["Yes", "No"], ["Oui", "Non"])),
    ("object_counting", "object_counting", "comptage_d_objets",
     "Questions that involve enumerating objects and asking the model to count them.\n\n",
     "Questions qui consistent à énumérer des objets et à les compter.\n\n",
     "count"),
    ("penguins_in_a_table", "penguins_in_a_table", "pingouins_sur_une_table",
     "Answer questions about a table of penguins and their attributes.\n\n",
     "Répondez à des questions sur un tableau de pingouins et leurs attributs.\n\n",
     5),
    ("reasoning_about_colored_objects", "reasoning_about_colored_objects", "raisonnement_sur_les_objets_colorés",
     "Answer extremely simple questions about the colors of objects on a surface.\n\n",
     "Répondez à des questions très simples sur la couleur d'objets posés sur une surface.\n\n",
     18),
    ("ruin_names", "ruin_names", "noms_de_ruines",
     "Select the humorous edit that 'ruins' the input movie or musical artist name.\n\n",
     "Choisissez la modification humoristique qui « gâche » le nom de film ou d'artiste donné.\n\n",
     6),
    ("salient_translation_error_detection", "salient_translation_error_detection",
     "détection_d_erreur_de_traduction_sailante",
     "Detect the type of error in an English translation of a German source sentence.\n\n",
     "Détectez le type d'erreur dans la traduction d'une phrase source.\n\n",
     6),
    ("snarks", "snarks", "sarcasmes",
     'Determine which of two sentences is sarcastic.\n\nAccording to Cambridge University Dictionary, sarcasm is "the use of remarks that clearly mean the opposite of what they say, made in order to hurt someone\'s feelings or to criticize something in a humorous way." Sarcastic sentences often contain satirical or ironic utterances, hyperboles, ambivalent or witty remarks.\n\n',
     "Déterminez laquelle de deux phrases est sarcastique.\n\nLe sarcasme est l'emploi de remarques qui signifient clairement le contraire de ce qu'elles disent, dans le but de blesser ou de critiquer de manière humoristique. Les phrases sarcastiques contiennent souvent des propos satiriques ou ironiques, des hyperboles ou des remarques ambivalentes ou spirituelles.\n\n",
     2),
    ("sports_understanding", "sports_understanding", "compréhension_des_sports",
     "Determine whether an artificially constructed sentence relating to sports is plausible or not.\n\n",
     "Déterminez si une phrase construite artificiellement à propos du sport est plausible ou non.\n\n",
     ("fixed", ["yes", "no"], ["Oui", "Non"])),
    ("temporal_sequences", "temporal_sequences", "séquences_temporelles",
     "Task description: Answer questions about which times certain events could have occurred.\n\n",
     "Description de la tâche : répondez à des questions sur les moments où certains événements ont pu se produire.\n\n",
     4),
    ("tracking_shuffled_objects_five_objects", "tracking_shuffled_objects_five_objects",
     "suivi_objets_mélangés_cinq_objets",
     "A task requiring determining the final positions of a set of objects given their initial positions and a description of a sequence of swaps.\n\n",
     "Une tâche consistant à déterminer les positions finales d'un ensemble d'objets à partir de leurs positions initiales et d'une séquence d'échanges.\n\n",
     5),
    ("tracking_shuffled_objects_seven_objects", "tracking_shuffled_objects_seven_objects",
     "suivi_objets_mélangés_sept_objets",
     "A task requiring determining the final positions of a set of objects given their initial positions and a description of a sequence of swaps.\n\n",
     "Une tâche consistant à déterminer les positions finales d'un ensemble d'objets à partir de leurs positions initiales et d'une séquence d'échanges.\n\n",
     7),
    ("tracking_shuffled_objects_three_objects", "tracking_shuffled_objects_three_objects",
     "suivi_objets_mélangés_trois_objets",
     "A task requiring determining the final positions of a set of objects given their initial positions and a description of a sequence of swaps.\n\n",
     "Une tâche consistant à déterminer les positions finales d'un ensemble d'objets à partir de leurs positions initiales et d'une séquence d'échanges.\n\n",
     3),
    ("web_of_lies", "web_of_lies", "toile_de_mensonges",
     "Evaluate a random boolean function expressed as a word problem.\n\n",
     "Évaluez une fonction booléenne aléatoire exprimée sous forme de problème.\n\n",
     ("fixed", ["Yes", "No"], ["Oui", "Non"])),
    ("word_sorting", "word_sorting", "tri_de_mots",
     "Sort a list of words.\n\n",
     "Triez une liste de mots.\n\n",
     "open"),
]
# fmt: on


def _make_task(name, hf_repo, hf_subset, instruction, choices_spec, lang):
    return LightevalTaskConfig(
        name=name,
        suite=["community"],
        prompt_function=_make_bbh_prompt(instruction, _choices_fn(choices_spec, lang)),
        hf_repo=hf_repo,
        hf_subset=hf_subset,
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=GENERATION_SIZE,
        metrics=BBH_METRICS,
        stop_sequence=STOP_SEQUENCE,
        version=0,
    )


BBH_FR_TASKS = [
    _make_task(f"bbh_fr:{key}", "le-leadboard/bbh-fr", fr_subset, fr_instr, spec, "fr")
    for key, en_subset, fr_subset, en_instr, fr_instr, spec in BBH_TASKS
]


TASKS_TABLE = BBH_FR_TASKS

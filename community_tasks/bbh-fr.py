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
"""BIG-Bench-Hard (BBH) as lighteval community tasks, chain-of-thought few-shot.

This reproduces the standard BBH protocol (Suzgun et al. 2022; lm-evaluation-harness'
``bbh_cot_fewshot``): each prompt is a task description + **3 fixed chain-of-thought
demonstrations** (each ending ``So the answer is X.``) + the test question. The model reasons
step by step, and we recover its *final committed answer* with a robust, type-aware extractor
(see below) and match it against gold — tolerating markdown, alternate answer cues, and the
verbose phrasing that a bare exact-match on ``the answer is`` would discard.

- English: ``community|bbh:<task>``    on ``lukaemon/bbh`` — demonstrations ported verbatim from
  lm-eval (``bbh_cot_fewshot_en.json``), so scores are comparable to lm-eval / the BBH leaderboard.
- French:  ``community|bbh_fr:<task>`` on ``le-leadboard/bbh-fr`` — demonstrations in
  ``bbh_cot_fewshot_fr.json`` (the 22 language-neutral tasks are translated; the 5 intrinsically
  English tasks — word_sorting, hyperbaton, snarks, disambiguation_qa,
  salient_translation_error_detection — are re-authored in French). There is no official French
  BBH-CoT reference, so the French set is a localization, not an official benchmark.

The 3-shot CoT is fixed (baked into the prompt) exactly as in lm-eval; the ``|N|`` few-shot knob
is inoperative here (``few_shots_split=None``), so always run these tasks at ``|0``.
"""

import json
import logging
import os
import re

import numpy as np

from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.metrics.utils.metric_utils import SampleLevelMetric
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod


logger = logging.getLogger(__name__)


GENERATION_SIZE = 1024  # room for chain-of-thought (matches lm-eval's bbh_cot_fewshot)
STOP_SEQUENCE = ["</s>", "Q", "\n\n"]  # stop after one reasoning block, as in lm-eval

# Correction for a mislabeled gold answer in le-leadboard/bbh-fr: the sophismes_formels
# (formal_fallacies) subset labels the "invalid" class as the misspelled "invalidee" (neither the
# option the prompt offers nor correct French). Canonicalize it to "invalide". See the upstream
# report; remove once fixed. "invalidee" is unique to this subset, so applying it globally is safe.
_GOLD_LABEL_FIXES = {"invalidee": "invalide"}


# canonical task key (== lukaemon/bbh subset) -> le-leadboard/bbh-fr config name
FR_SUBSET = {
    "boolean_expressions": "expressions_booléennes",
    "causal_judgement": "jugement_causal",
    "date_understanding": "compréhension_de_la_date",
    "disambiguation_qa": "désambiguïsation_qa",
    "dyck_languages": "dyck_languages",
    "formal_fallacies": "sophismes_formels",
    "geometric_shapes": "formes_géométriques",
    "hyperbaton": "hyperbate",
    "logical_deduction_five_objects": "déduction_logique_cinq_objets",
    "logical_deduction_seven_objects": "déduction_logique_sept_objets",
    "logical_deduction_three_objects": "déduction_logique_trois_objets",
    "movie_recommendation": "recommandation_de_film",
    "multistep_arithmetic_two": "multistep_arithmetic_two",
    "navigate": "naviguer",
    "object_counting": "comptage_d_objets",
    "penguins_in_a_table": "pingouins_sur_une_table",
    "reasoning_about_colored_objects": "raisonnement_sur_les_objets_colorés",
    "ruin_names": "noms_de_ruines",
    "salient_translation_error_detection": "détection_d_erreur_de_traduction_sailante",
    "snarks": "sarcasmes",
    "sports_understanding": "compréhension_des_sports",
    "temporal_sequences": "séquences_temporelles",
    "tracking_shuffled_objects_five_objects": "suivi_objets_mélangés_cinq_objets",
    "tracking_shuffled_objects_seven_objects": "suivi_objets_mélangés_sept_objets",
    "tracking_shuffled_objects_three_objects": "suivi_objets_mélangés_trois_objets",
    "web_of_lies": "toile_de_mensonges",
    "word_sorting": "tri_de_mots",
}

# HF datasets for each language.
EN_REPO = "lukaemon/bbh"
FR_REPO = "le-leadboard/bbh-fr"

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_demos(filename):
    """Load a CoT demonstrations file (``{task: {description, doc_to_text, samples}}``), if present."""
    path = os.path.join(_HERE, filename)
    if not os.path.exists(path):
        logger.warning(f"BBH CoT demos file not found: {path} (those tasks will be skipped).")
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


_DEMOS = {
    "en": _load_demos("bbh_cot_fewshot_en.json"),
    "fr": _load_demos("bbh_cot_fewshot_fr.json"),
}


def _build_cot_query(demo: dict, test_input: str) -> str:
    """Assemble: description + 3 CoT demonstrations + the test question (as in lm-eval)."""

    def _render(inp):
        return demo["doc_to_text"].replace("{{input}}", inp)

    blocks = [_render(s["input"]) + s["target"] for s in demo["samples"]]
    blocks.append(_render(test_input))
    return demo["description"] + "\n\n".join(blocks)


def _make_cot_prompt(task_key: str, lang: str):
    demo = _DEMOS.get(lang, {}).get(task_key)

    def prompt_fn(line, task_name: str = None):
        if demo is None:
            return []  # no demonstrations available for this task/language
        target = _GOLD_LABEL_FIXES.get(line["target"], line["target"])
        return Doc(
            task_name=task_name,
            query=_build_cot_query(demo, line["input"]),
            choices=[target],
            gold_index=0,
            instruction=demo["description"],
        )

    return prompt_fn


# ---------------------------------------------------------------------------
# Robust answer extraction & matching.
#
# A bare "exact match on the text after 'the answer is'" throws away most correct
# answers from real (small, chatty, instruction-tuned) models: they wrap the answer
# in markdown, use a different cue ("Réponse finale :"), omit the cue entirely, or add
# trailing notes. This extractor recovers the *final committed answer* and matches it
# tolerantly, WITHOUT rescuing genuine non-answers/self-contradiction (it always takes
# the last cue, then the first token of that answer span — like lm-eval's flexible
# regex-after-final-cue). It is language- and model-agnostic (FR + EN cues, extensible).
#
# Kept in sync with the standalone prototype + unit tests under ``tmp/bbh_extractor/``;
# intended to later move into lighteval core as a shared BBH metric.
# ---------------------------------------------------------------------------

# Answer-introducing cues, matched case-insensitively; the LAST occurrence wins.
_CUES = [
    "la bonne réponse est",
    "la réponse correcte est",
    "la réponse finale est",
    "réponse finale :",
    "réponse finale",
    "la réponse est",
    "réponse correcte :",
    "réponse est :",
    "réponse est",
    "réponse :",
    "réponse:",
    "the correct answer is",
    "the final answer is",
    "final answer:",
    "final answer",
    "the answer is",
    "answer is",
    "answer:",
    "so the answer is",
]

_QUOTES = "\"'“”«»‹›`"

# Equivalence classes: gold label <-> what models actually write (FR/EN synonyms).
_CLASSES = {
    "yes": {"oui", "yes", "y", "o", "vrai", "true", "plausible"},
    "no": {"non", "no", "n", "faux", "false", "pas plausible", "non plausible", "implausible"},
    "true": {"vrai", "true", "correct", "oui", "yes"},
    "false": {"incorrect", "faux", "false", "non", "no"},
    "valid": {"valide", "valid", "déductivement valide"},
    "invalid": {"invalide", "invalid", "non valide", "not valid", "invalidee", "sophisme", "fallacy"},
}


def _strip_markdown(s: str) -> str:
    s = s.replace("**", "").replace("__", "").replace("`", "")
    s = s.replace("###", " ").replace("##", " ")
    s = re.sub(r"(?<![\w*])\*(?!\*)", "", s)
    s = s.replace("✅", " ").replace("→", " ")
    return s


def _norm(s) -> str:
    """Normalise a short answer span for comparison: strip markdown/quotes/punct, lowercase."""
    s = _strip_markdown(str(s))
    s = re.sub(r"\s+", " ", s).strip()
    s = s.strip(_QUOTES + " .:;!?)(").strip()
    return s.lower()


def _gold_type(gold: str):
    g = _norm(gold)
    if re.fullmatch(r"\(?[a-z]\)?", g):
        return "letter", re.sub(r"[()]", "", g)
    if re.fullmatch(r"-?\d+", g):
        return "number", g
    if g in ("oui", "non"):
        return "yesno", g
    if g in ("vrai", "incorrect"):
        return "boolean", g
    if g in ("valide", "invalide"):
        return "validity", g
    if re.fullmatch(r"[\[\](){}<>\s]+", g):
        return "brackets", g
    return "freeform", g


def _clean_line(ln: str) -> str:
    """Drop leading colon and surrounding whitespace, but keep signs/brackets (e.g. '-26', ']')."""
    return ln.strip().lstrip(":").strip()


def _answer_segment(pred: str):
    """Text after the LAST answer cue: the same-line tail, else the next non-empty line."""
    text = _strip_markdown(pred)
    low = text.lower()
    best_idx, best_cue = -1, None
    for cue in _CUES:
        i = low.rfind(cue)
        if i > best_idx:
            best_idx, best_cue = i, cue
    if best_idx == -1:
        return None
    lines = text[best_idx + len(best_cue) :].split("\n")
    if _clean_line(lines[0]):
        return _clean_line(lines[0])
    for ln in lines[1:]:
        s = _clean_line(ln)
        if s and not s.startswith("```"):
            return s
    return ""


def _last_nonempty_line(pred: str) -> str:
    lines = [ln.strip() for ln in _strip_markdown(pred).splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _parse_options(query):
    """Map option letter <-> text from the *test question's* (last) 'Options' block only."""
    if not query:
        return {}, {}
    idx = query.lower().rfind("options")
    scope = query[idx:] if idx != -1 else query
    letter2text, text2letter = {}, {}
    for m in re.finditer(r"\(([A-Za-z])\)\s*([^\n]+)", scope):
        letter, text = m.group(1).lower(), _norm(m.group(2))
        if text:
            letter2text[letter] = text
            text2letter.setdefault(text, letter)
    return letter2text, text2letter


def _find_letter(segment, whole, query):
    letter2text, text2letter = _parse_options(query)
    if segment:  # FIRST (X) in the answer segment = the committed answer
        m = re.findall(r"\(([A-Za-z])\)", segment)
        if m:
            return m[0].lower()
        seg = _norm(segment)  # option TEXT stated without a letter -> map back
        for text in sorted(text2letter, key=len, reverse=True):
            if text and text in seg:
                return text2letter[text]
        m = re.search(r"\b([a-z])\b", seg)  # bare letter e.g. "réponse est b"
        if m and m.group(1) in letter2text:
            return m.group(1)
    letters = re.findall(r"\(([A-Za-z])\)", whole or "")  # fallback: last (X) anywhere
    return letters[-1].lower() if letters else None


def _first_in(text, classes):
    t = _norm(text)
    best_c, best_pos = None, len(t) + 1
    for c in classes:
        for member in _CLASSES[c]:
            m = re.search(rf"(?<![a-zà-ÿ]){re.escape(member)}(?![a-zà-ÿ])", t)
            if m and m.start() < best_pos:
                best_pos, best_c = m.start(), c
    return best_c


def _last_in(text, classes):
    t = _norm(text)
    best_c, best_pos = None, -1
    for c in classes:
        for member in _CLASSES[c]:
            for m in re.finditer(rf"(?<![a-zà-ÿ]){re.escape(member)}(?![a-zà-ÿ])", t):
                if m.start() > best_pos:
                    best_pos, best_c = m.start(), c
    return best_c


def _find_class(segment, whole, classes):
    """Committed class = FIRST class member in the segment; else LAST in the whole text."""
    if segment:
        c = _first_in(segment, classes)
        if c:
            return c
    return _last_in(whole, classes) if whole else None


def _find_number(segment, whole):
    if segment:
        nums = re.findall(r"-?\d+", segment.replace(" ", ""))
        if nums:
            return nums[0]
    nums = re.findall(r"-?\d+", (whole or "").replace(" ", ""))
    return nums[-1] if nums else None


def _token_seq(s) -> str:
    """Separator-agnostic token sequence (commas/semicolons/slashes -> spaces) for free-form answers."""
    return re.sub(r"\s+", " ", re.sub(r"[,;/]", " ", _norm(s))).strip()


def _robust_match(prediction: str, gold: str, query=None) -> float:
    """1.0 if the prediction's final committed answer matches gold, else 0.0."""
    gtype, gnorm = _gold_type(gold)
    segment = _answer_segment(prediction)
    had_cue = segment is not None
    seg_or_line = segment if had_cue else _last_nonempty_line(prediction)
    # Whole-text fallback is only trusted when the model signalled an answer with a cue,
    # so a degenerate ramble can't be rescued by a stray letter mid-reasoning.
    whole = prediction if had_cue else ""

    if gtype == "letter":
        ok = _find_letter(seg_or_line, whole, query) == gnorm
    elif gtype == "number":
        ok = _find_number(seg_or_line, whole) == gnorm
    elif gtype in ("yesno", "boolean", "validity"):
        classes = {"yesno": ["yes", "no"], "boolean": ["true", "false"], "validity": ["valid", "invalid"]}[gtype]
        gold_class = {
            "oui": "yes",
            "non": "no",
            "vrai": "true",
            "incorrect": "false",
            "valide": "valid",
            "invalide": "invalid",
        }[gnorm]
        ok = _find_class(seg_or_line, whole, classes) == gold_class
    elif gtype == "brackets":
        pv = re.sub(r"[^\[\](){}<>]", "", seg_or_line or "")
        ok = bool(pv) and pv == re.sub(r"[^\[\](){}<>]", "", gnorm)
    else:  # freeform (e.g. word_sorting): separator-agnostic token sequence
        ok = _token_seq(seg_or_line) == _token_seq(gnorm)
    return 1.0 if ok else 0.0


class BBHCotExactMatch(SampleLevelComputation):
    """Robustly extract the CoT generation's final committed answer and match it against gold."""

    def compute(self, model_response, doc, **kwargs) -> float:
        pred = model_response.text[0] if getattr(model_response, "text", None) else ""
        return _robust_match(pred, doc.choices[0], getattr(doc, "query", None))


_METRIC = SampleLevelMetric(
    metric_name="em",
    sample_level_fn=BBHCotExactMatch(),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)


def _make_task(name, hf_repo, hf_subset, prompt_fn):
    return LightevalTaskConfig(
        name=name,
        suite=["community"],
        prompt_function=prompt_fn,
        hf_repo=hf_repo,
        hf_subset=hf_subset,
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split=None,  # the 3-shot CoT is baked into the prompt; do not sample
        few_shots_select=None,
        generation_size=GENERATION_SIZE,
        metrics=[_METRIC],
        stop_sequence=STOP_SEQUENCE,
        version=1,  # v1: robust answer extraction (was strict marker exact-match)
    )


# English tasks: one per demonstration set available (all 27 from lm-eval).
BBH_EN_TASKS = [_make_task(f"bbh:{key}", EN_REPO, key, _make_cot_prompt(key, "en")) for key in _DEMOS["en"]]

# French tasks: one per demonstration set available in the French file.
BBH_FR_TASKS = [
    _make_task(f"bbh_fr:{key}", FR_REPO, FR_SUBSET[key], _make_cot_prompt(key, "fr"))
    for key in _DEMOS["fr"]
    if key in FR_SUBSET
]


TASKS_TABLE = BBH_EN_TASKS + BBH_FR_TASKS

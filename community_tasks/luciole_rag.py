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

"""RAG Luciole benchmark — citation-aware grounded QA evaluation.

Reads rows from the HF dataset ``Mvanypersele/luciole_rag_benchmark`` and
rebuilds the system prompt at evaluation time.

Row schema (one per example)
----------------------------
- ``id``: stable example identifier
- ``query``: user question
- ``retrieved_documents``: list[str], retrieved context chunks
- ``titles``: list[str], aligned with ``retrieved_documents``
- ``supporting_index``: list[int], zero-based indices of gold/supporting chunks
- ``answer``: expected answer (empty when ``supporting_index`` is empty,
  marking the row as unanswerable)

Tasks
-----
One ``luciole_rag:<subset>`` task per subset (``hotpotqa``, ``hotpotqa_fr``,
``tatqa``, ``piaf``, ``newsquadfr``, ``squad2_fr_pragnakalp``). A deterministic
md5-based partition on the row id drops the supporting chunks on a fraction of
the answerable rows, turning them into synthetic unanswerables. The fraction
is set by ``LUCIOLE_RAG_DROP_RATIO`` (default 0.5). With ratio 0.0 the run
is pure answerable; with 1.0 it is pure unanswerable. One run fills both the
answer/citation metrics (on kept rows) and the refusal metrics (on dropped
rows).

Prompt
------
Built per row with a single citation rule (each quoted excerpt is wrapped
inline in ``<ref name="title">...</ref>``, where ``title`` matches the
``[title]`` header of the cited chunk in the context) and a single refusal
rule that instructs the model to reply with one **canonical refusal phrase**
verbatim. Detection of refusal is lenient: it matches the shorter invariant
core of the canonical phrase (e.g. ``do not allow me to answer`` / ``ne
permettent pas de répondre``), so common paraphrases of the prefix ("The
provided context...", "The available/retrieved documents...") still count
as refusals. The prompt language (FR/EN) is detected per row from the
query.

Judge
-----
Optional LLM-as-judge factual evaluation, opt-in via
``LUCIOLE_RAG_USE_JUDGE=1``. Uses litellm by default; for a custom
OpenAI-compatible endpoint set ``LLM_API_URL``, ``OPENAI_API_KEY`` and
``LLM_MODEL`` (with the ``openai/`` prefix).
"""

import hashlib
import json
import logging
import os
import random
import re

import numpy as np

from lighteval.metrics.metrics_sample import JudgeLLM, SampleLevelComputation
from lighteval.metrics.utils.metric_utils import SampleLevelMetricGrouping
from lighteval.metrics.utils.stderr import mean_stderr
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod


logger = logging.getLogger(__name__)


# ── citation extraction regex ──────────────────────────────────────

# The prompt instructs a single citation syntax: ``<ref name="title">...</ref>``.
# Any other syntax in the model output is treated as a failure to follow the
# citation instruction (lower precision/recall). The ``name`` attribute may
# use single or double quotes.
_CITATION_TAG_RE = re.compile(
    r'<ref\s+name\s*=\s*["\']([^"\']+)["\']\s*>',
    re.IGNORECASE,
)


# ── refusal: canonical phrases ─────────────────────────────────────

# The system prompt instructs the model to reply **exactly** with the
# language-matched phrase below when the context is insufficient. In
# practice the model frequently paraphrases the prefix ("The provided
# context...", "The available documents...", "The retrieved documents...",
# etc.) while keeping the invariant tail stable, so detection (see
# ``detect_refusal``) matches only that shorter invariant core rather than
# the full canonical phrase.
REFUSAL_PHRASE = {
    "en": "The provided documents do not allow me to answer this question.",
    "fr": "Les documents fournis ne permettent pas de répondre à cette question.",
}

# Lenient refusal-detection substrings, per language. Each is matched against
# the case-insensitive, whitespace-collapsed response. Covers the variants
# observed in model outputs: with/without the "me"/object pronoun, the
# "do not"/"don't" contraction in EN, and singular/plural verb agreement in
# FR (which follows whether the model rephrased the subject as singular —
# "the context" / "le contexte" — or plural — "the documents" / "les
# documents").
_REFUSAL_DETECTION_PHRASES = {
    "en": (
        "does not allow me to answer",
        "does not allow to answer",
        "do not allow me to answer",
        "do not allow to answer",
        "doesn't allow me to answer",
        "doesn't allow to answer",
        "don't allow me to answer",
        "don't allow to answer",
    ),
    "fr": (
        "ne permettent pas de répondre",
        "ne permet pas de répondre",
        "ne me permettent pas de répondre",
        "ne me permet pas de répondre",
    ),
}


# ── pure utility functions ──────────────────────────────────────────


def normalize_answer(answer: str) -> str:
    answer = answer.lower()
    answer = re.sub(r"\b(a|an|the|le|la|les|l|un|une|des|du|de|d)\b", " ", answer)
    answer = re.sub(r"[^\w\s]", "", answer)
    return " ".join(answer.split()).strip()


def _normalize_spaces(text: str) -> str:
    return " ".join(text.lower().split())


_NORMALIZED_REFUSAL_PHRASES = tuple(
    _normalize_spaces(p) for phrases in _REFUSAL_DETECTION_PHRASES.values() for p in phrases
)


def detect_refusal(response: str) -> bool:
    """True iff the response contains any of the lenient refusal-detection
    phrases (see ``_REFUSAL_DETECTION_PHRASES``) in either supported language.
    Match is case-insensitive and whitespace-tolerant (line breaks and runs
    of spaces collapse to a single space).
    """
    norm = _normalize_spaces(response)
    return any(p in norm for p in _NORMALIZED_REFUSAL_PHRASES)


def extract_cited_titles(response: str) -> list[str]:
    """Extract titles from ``<ref name="title">...</ref>`` tags only.

    Other citation syntaxes are intentionally not parsed: the prompt
    instructs this exact form, so unparsed citations count as
    instruction-following failures (lower precision/recall).
    """
    seen: set[str] = set()
    unique: list[str] = []
    for m in _CITATION_TAG_RE.finditer(response):
        title = m.group(1).strip()
        norm = title.lower()
        if norm and norm not in seen:
            seen.add(norm)
            unique.append(title)
    return unique


def _citation_key(title: str) -> str:
    title = re.sub(r"\s*\[[0-9a-f]{8,}\]\s*$", "", title.strip(), flags=re.IGNORECASE)
    return title.lower()


def _citation_match(a: str, b: str) -> bool:
    return _citation_key(a) == _citation_key(b)


def _citation_fuzzy_match(a: str, b: str) -> bool:
    a_key = _citation_key(a)
    b_key = _citation_key(b)
    return bool(a_key and b_key and (a_key == b_key or a_key in b_key or b_key in a_key))


def evaluate_citations(
    cited: list[str],
    expected: list[str],
    *,
    fuzzy: bool = False,
) -> tuple[float | None, float | None, float | None]:
    """Citation precision/recall/F1 after citation-key normalization."""
    cited_dedup = list(dict.fromkeys(t.strip() for t in cited if _citation_key(t)))
    expected_dedup = list(dict.fromkeys(t.strip() for t in expected if _citation_key(t)))
    if not cited_dedup and not expected_dedup:
        return None, None, None

    match = _citation_fuzzy_match if fuzzy else _citation_match
    correct_cited = sum(1 for c in cited_dedup if any(match(c, g) for g in expected_dedup))
    matched_gold = sum(1 for g in expected_dedup if any(match(c, g) for c in cited_dedup))
    precision = correct_cited / len(cited_dedup) if cited_dedup else None
    recall = matched_gold / len(expected_dedup) if expected_dedup else None
    f1 = None
    if precision is not None and recall is not None:
        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def compute_answer_em(response: str, gold_answer: str) -> float:
    """1.0 when the normalized gold answer appears as a substring of the
    normalized response, else 0.0.
    """
    if not gold_answer:
        return 0.0
    norm_pred = normalize_answer(response)
    norm_gold = normalize_answer(gold_answer)
    if not norm_gold:
        return 0.0
    return 1.0 if norm_gold in norm_pred else 0.0


_FUZZY_EM_THRESHOLD = 0.80


def compute_answer_em_fuzzy(
    response: str,
    gold_answer: str,
    threshold: float = _FUZZY_EM_THRESHOLD,
) -> float:
    """Token-recall EM: 1.0 iff ≥``threshold`` of the gold's unique content
    tokens appear in the response token set.
    """
    if not gold_answer:
        return 0.0
    pred_tokens = set(normalize_answer(response).split())
    gold_tokens = set(normalize_answer(gold_answer).split())
    if not gold_tokens:
        return 0.0
    overlap = len(pred_tokens & gold_tokens) / len(gold_tokens)
    return 1.0 if overlap >= threshold else 0.0


def _extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if m:
        return json.loads(m.group(0))
    raise json.JSONDecodeError("No JSON object found", text, 0)


# ── prompt-building primitives ─────────────────────────────────────

_REF_TEMPLATE = '<ref name="{title}">{excerpt}</ref>'

CITATION_INSTRUCTION = {
    "en": (
        "When quoting from the context, wrap each excerpt inline with "
        f"`{_REF_TEMPLATE.replace('{title}', 'source title').replace('{excerpt}', 'excerpt')}`, "
        "where `source title` matches the `[title]` header of the cited chunk in the context. "
    ),
    "fr": (
        "Lorsque vous citez le contexte, encadrez chaque extrait en ligne avec "
        f"`{_REF_TEMPLATE.replace('{title}', 'titre de la source').replace('{excerpt}', 'extrait')}`, "
        "où `titre de la source` correspond à l'en-tête `[titre]` du document cité dans le contexte. "
    ),
}

REFUSAL_INSTRUCTION = {
    "en": (
        "If the context is **insufficient** to answer the question, reply "
        f"**exactly** with: `{REFUSAL_PHRASE['en']}` and nothing else."
    ),
    "fr": (
        "Si le contexte est **insuffisant** pour répondre à la question, "
        f"répondez **exactement** par : `{REFUSAL_PHRASE['fr']}` et rien d'autre."
    ),
}

SYSTEM_PROMPT_EN = (
    "You are an AI conversational assistant specialized in **information retrieval and synthesis**.\n"
    "Your goal is to provide **precise, reliable, and well-structured answers** using **only the retrieved documents** (`Context`).\n"
    "Prioritize **clarity, accuracy, and completeness** in your responses.\n"
    "\n"
    "## Rules\n"
    "\n"
    "1. Use only the provided Context\n"
    "   * Base your answer **exclusively** on the information contained in the `Context`.\n"
    "   * **Never infer**, assume, or rely on any external knowledge.\n"
    "   * {refusal_instruction}\n"
    "   * {citation_instruction}\n"
    "\n"
    "2. Language Consistency\n"
    "   * Always respond **in the same language** as the user's query.\n"
    "\n"
    "3. Structure and Readability\n"
    "   * Ensure responses are **concise yet complete**, avoiding omission of key details.\n"
    "\n"
    "Here are the retrieved documents : `{context}`"
)

SYSTEM_PROMPT_FR = (
    "Vous êtes un assistant conversationnel IA spécialisé dans la **recherche et la synthèse d'informations**.\n"
    "Votre objectif est de fournir des **réponses précises, fiables et bien structurées** en utilisant **uniquement les documents récupérés** (`Contexte`).\n"
    "Privilégiez la **clarté, l'exactitude et l'exhaustivité** dans vos réponses.\n"
    "\n"
    "## Règles\n"
    "\n"
    "1. Utilisez uniquement le Contexte fourni\n"
    "   * Basez votre réponse **exclusivement** sur les informations contenues dans le `Contexte`.\n"
    "   * **N'inférez jamais**, ne supposez pas et ne vous appuyez pas sur des connaissances externes.\n"
    "   * {refusal_instruction}\n"
    "   * {citation_instruction}\n"
    "\n"
    "2. Cohérence linguistique\n"
    "   * Répondez toujours **dans la même langue** que la requête de l'utilisateur.\n"
    "\n"
    "3. Structure et lisibilité\n"
    "   * Assurez-vous que les réponses sont **concises mais complètes**, en évitant d'omettre les détails clés.\n"
    "\n"
    "Voici les documents récupérés : `{context}`"
)

SYSTEM_PROMPTS = {"en": SYSTEM_PROMPT_EN, "fr": SYSTEM_PROMPT_FR}


# ── language detection ─────────────────────────────────────────────

_FRENCH_HINT_RE = re.compile(
    r"[éèêëàâçùûôîïÉÈÊËÀÂÇÙÛÔÎÏ]|"
    r"\b(?:le|la|les|une?|des?|du|aux?|qui|que|quoi|quels?|quelles?|"
    r"est|sont|était|étaient|dans|pour|avec|sur|par|cette?|leurs?|"
    r"comment|pourquoi|combien|où|quand|"
    r"c'est|n'est|qu'est|n'a|d'un|d'une|d'autres)\b",
    re.IGNORECASE,
)


def detect_language(text: str) -> str:
    """Cheap FR/EN classifier used to pick the system-prompt template."""
    return "fr" if _FRENCH_HINT_RE.search(text or "") else "en"


# ── context formatting ────────────────────────────────────────────


def _format_chunk(title: str, document: str) -> str:
    """Emit a chunk as ``[title]\\ncontent``."""
    return f"[{title}]\n{document}"


def build_context(titles: list[str], documents: list[str]) -> str:
    return "\n\n".join(_format_chunk(str(t), str(d)) for t, d in zip(titles, documents))


# ── prompt function ────────────────────────────────────────────────


# Fraction of answerable rows to convert into synthetic unanswerables, set
# from the environment at import time. 0.0 = keep all supports, 1.0 = drop
# them all. The partition is deterministic per-id (md5-based) so reruns at
# the same ratio yield identical samples.
DROP_RATIO = float(os.getenv("LUCIOLE_RAG_DROP_RATIO", "0.5"))

# Present the kept chunks in a shuffled order to remove position bias (so the
# gold chunks aren't always at a fixed slot). The shuffle is deterministic
# per-id, so reruns yield identical orderings. Disable with
# LUCIOLE_RAG_SHUFFLE_CHUNKS=0.
SHUFFLE_CHUNKS = os.getenv("LUCIOLE_RAG_SHUFFLE_CHUNKS", "1").strip().lower() in ("1", "true", "yes", "on")


def _hash_unit(key: str) -> float:
    """Stable [0, 1) bucket per row id. Deterministic across runs and processes
    (uses md5 of the id rather than Python's randomised ``hash()``).
    """
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0x100000000


def _shuffle_deterministic(items: list, seed_key: str) -> list:
    """Return a new list with ``items`` shuffled by a per-id seeded RNG.

    Seeding ``random.Random`` with the string key is stable across runs and
    processes (unlike Python's randomised ``hash()``).
    """
    shuffled = list(items)
    random.Random(f"chunk_order:{seed_key}").shuffle(shuffled)
    return shuffled


def luciole_rag_prompt(line, task_name: str | None = None) -> Doc:
    """Convert one prompt-agnostic row into a lighteval Doc.

    A deterministic md5-based bucket on the row id decides whether to drop
    the supporting chunks (``bucket < DROP_RATIO``). The decision is
    independent of task name and stable across runs at the same ratio.

    When ``SHUFFLE_CHUNKS`` is set, the kept chunks are presented in a
    per-id deterministic shuffled order to remove position bias.
    """
    query = line["query"]
    retrieved = list(line.get("retrieved_documents") or [])
    titles = list(line.get("titles") or [])
    supporting_index = [int(i) for i in (line.get("supporting_index") or [])]
    answer = (line.get("answer") or "").strip()
    row_id = line.get("id", "")

    if len(retrieved) != len(titles):
        raise ValueError(
            f"luciole_rag_prompt: retrieved_documents/titles length mismatch "
            f"({len(retrieved)} vs {len(titles)}) for id={row_id!r}"
        )

    drop_supports = _hash_unit(str(row_id)) < DROP_RATIO
    support_set = {i for i in supporting_index if 0 <= i < len(titles)}

    if drop_supports:
        kept = [i for i in range(len(retrieved)) if i not in support_set]
    else:
        kept = list(range(len(retrieved)))

    if SHUFFLE_CHUNKS:
        kept = _shuffle_deterministic(kept, str(row_id))

    kept_titles = [str(titles[i]) for i in kept]
    kept_documents = [str(retrieved[i]) for i in kept]

    if drop_supports:
        effective_gold_titles: list[str] = []
        is_unanswerable = True
        reference_answer = ""
    else:
        effective_gold_titles = [str(titles[i]) for i in support_set]
        is_unanswerable = len(support_set) == 0
        reference_answer = answer

    language = detect_language(query)
    template = SYSTEM_PROMPTS.get(language, SYSTEM_PROMPT_EN)
    context = build_context(kept_titles, kept_documents)
    system_content = template.format(
        context=context,
        citation_instruction=CITATION_INSTRUCTION[language],
        refusal_instruction=REFUSAL_INSTRUCTION[language],
    )

    return Doc(
        task_name=task_name or "",
        query=query,
        instruction=system_content,
        choices=[reference_answer],
        gold_index=0,
        specific={
            "context": context,
            "chunk_titles": kept_titles,
            "supporting_facts_titles": effective_gold_titles,
            "is_unanswerable": is_unanswerable,
            "reference_answer": reference_answer,
            "instruction_as_system": True,
            "row_id": row_id,
            "language": language,
            "drop_supports": drop_supports,
        },
    )


# ── corpus aggregators skipping out-of-scope rows ───────────────────


def _clean_applicable_values(values) -> list[float]:
    """Keep only rows where the metric is applicable.

    ``None`` means "out of scope for this row" (for example, citation metrics
    on unanswerable rows); ``0.0`` means an applicable failure. Corpus metrics
    and stderr are computed on this same applicable subset.
    """
    return [float(v) for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]


def _stderr(values: list[float]) -> float:
    if len(values) <= 1:
        return float("nan")
    return float(mean_stderr(values))


def _nanmean_skip_none_with_stderr(metric_name: str):
    def aggregate(values) -> dict[str, float]:
        cleaned = _clean_applicable_values(values)
        if not cleaned:
            return {metric_name: float("nan")}
        return {
            metric_name: float(np.mean(cleaned)),
            f"{metric_name}_stderr": _stderr(cleaned),
        }

    return aggregate


def _rag_corpus_aggregators(metric_names: list[str]) -> dict[str, object]:
    return {name: _nanmean_skip_none_with_stderr(name) for name in metric_names}


# ── per-sample metric grouping ──────────────────────────────────────


_SAMPLE_METRIC_NAMES = [
    "answer_em",
    "answer_em_fuzzy",
    "citation_precision_strict",
    "citation_recall_strict",
    "citation_f1_strict",
    "citation_precision_fuzzy",
    "citation_recall_fuzzy",
    "citation_f1_fuzzy",
    "distractor_citation_rate",
    "refusal_recall",
    "refusal_precision",
    "false_refusal_rate",
]


class LucioleRagSampleMetrics(SampleLevelComputation):
    """Per-sample metrics with answerability-conditional gating.

    On answerable rows: emits citation/quality metrics; refusal_recall is None
    (the row can't measure recall of unanswerables); false_refusal_rate is 1
    iff the model refused; refusal_precision contributes 0 if refused, None
    otherwise (so the corpus mean over non-None gives correct-refusals/all-refusals).

    On unanswerable rows: citation/quality metrics are None (skipped); refusal_recall
    is 1 iff refused; false_refusal_rate is None; refusal_precision contributes 1
    if refused, None otherwise.
    """

    def compute(self, model_response, doc, **kwargs):
        spec = doc.specific or {}
        gold_titles = spec.get("supporting_facts_titles", []) or []
        is_unanswerable = bool(spec.get("is_unanswerable", False))
        reference_answer = spec.get("reference_answer", "") or ""

        response_text = model_response.final_text[0] if model_response.final_text else ""
        refused = detect_refusal(response_text)

        if is_unanswerable:
            return {
                "answer_em": None,
                "answer_em_fuzzy": None,
                "citation_precision_strict": None,
                "citation_recall_strict": None,
                "citation_f1_strict": None,
                "citation_precision_fuzzy": None,
                "citation_recall_fuzzy": None,
                "citation_f1_fuzzy": None,
                "distractor_citation_rate": None,
                "refusal_recall": 1.0 if refused else 0.0,
                "refusal_precision": 1.0 if refused else None,
                "false_refusal_rate": None,
            }

        cited = extract_cited_titles(response_text)

        answer_em = compute_answer_em(response_text, reference_answer)
        answer_em_fuzzy = compute_answer_em_fuzzy(response_text, reference_answer)
        precision_strict, recall_strict, citation_f1_strict = evaluate_citations(cited, gold_titles)
        precision_fuzzy, recall_fuzzy, citation_f1_fuzzy = evaluate_citations(cited, gold_titles, fuzzy=True)

        # Distractor: any cited title that does not exactly match a gold
        # supporting fact. Includes both wrong-but-real chunks and
        # hallucinated titles that aren't in the context at all.
        cited_distractor_count = sum(1 for c in cited if not any(_citation_match(c, g) for g in gold_titles))
        distractor_rate = cited_distractor_count / len(cited) if cited else 0.0

        return {
            "answer_em": answer_em,
            "answer_em_fuzzy": answer_em_fuzzy,
            "citation_precision_strict": precision_strict,
            "citation_recall_strict": recall_strict,
            "citation_f1_strict": citation_f1_strict,
            "citation_precision_fuzzy": precision_fuzzy,
            "citation_recall_fuzzy": recall_fuzzy,
            "citation_f1_fuzzy": citation_f1_fuzzy,
            "distractor_citation_rate": distractor_rate,
            "refusal_recall": None,
            "refusal_precision": 0.0 if refused else None,
            "false_refusal_rate": 1.0 if refused else 0.0,
        }


_HIGHER_IS_BETTER = {
    "answer_em": True,
    "answer_em_fuzzy": True,
    "citation_precision_strict": True,
    "citation_recall_strict": True,
    "citation_f1_strict": True,
    "citation_precision_fuzzy": True,
    "citation_recall_fuzzy": True,
    "citation_f1_fuzzy": True,
    "distractor_citation_rate": False,
    "refusal_recall": True,
    "refusal_precision": True,
    "false_refusal_rate": False,
}


luciole_rag_sample_metrics = SampleLevelMetricGrouping(
    metric_name=_SAMPLE_METRIC_NAMES,
    higher_is_better=_HIGHER_IS_BETTER,
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=LucioleRagSampleMetrics(),
    corpus_level_fn=_rag_corpus_aggregators(_SAMPLE_METRIC_NAMES),
)


# ── factual judge ───────────────────────────────────────────────────


JUDGE_FACTUAL_SYSTEM_PROMPT = """\
You are an impartial factual evaluator. You will be given:
1. A **question**.
2. A **correct answer** (ground truth).
3. The **supporting facts** (the specific document titles that contain the evidence needed to answer the question).
4. A **context** (retrieved documents).
5. A **reasoning trace** produced by an AI assistant.

Your task is to rate the **factual correctness and faithfulness** of the reasoning trace on a scale from 1 to 5:

- **1**: The final answer is wrong AND the reasoning does not use the correct supporting facts at all.
- **2**: The final answer is wrong, but the reasoning references some of the correct supporting facts; OR the answer is partially right but the reasoning is based on wrong evidence.
- **3**: The final answer is approximately correct but imprecise, or the reasoning misses one of the key supporting facts, or the reasoning contains a factual error despite reaching the right answer.
- **4**: The final answer is correct and the reasoning uses most of the supporting facts properly, with only minor omissions or imprecisions.
- **5**: The final answer is correct, the reasoning correctly identifies and uses all the supporting facts, and the logical chain from evidence to answer is flawless.

You MUST reply with ONLY a JSON object in this exact format (no other text):
{"score": <int>, "justification": "<one sentence>"}
"""


def build_factual_judge_messages(question, answer, options, gold, **kwargs) -> list[dict]:
    supporting_facts = kwargs.get("supporting_facts") or []
    context = kwargs.get("context") or ""
    sf_text = "\n".join(f"- {t}" for t in supporting_facts) if supporting_facts else "(none available)"
    user_content = (
        f"**Question:**\n{question}\n\n"
        f"**Correct answer:**\n{gold or ''}\n\n"
        f"**Supporting facts (document titles):**\n{sf_text}\n\n"
        f"**Context:**\n{context}\n\n"
        f"**Reasoning trace:**\n{answer}"
    )
    return [
        {"role": "system", "content": JUDGE_FACTUAL_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def parse_factual_judge_response(text: str) -> int | None:
    if text is None:
        return None
    try:
        parsed = _extract_json(text)
        score = int(parsed["score"])
        if 1 <= score <= 5:
            return score
    except Exception as exc:
        logger.warning("Factual judge response parse failed: %s", exc)
    return None


_DEFAULT_JUDGE_MODEL = os.getenv("LLM_MODEL", "openai/Mistral-Small-3.1-24B-Instruct-2503")
_DEFAULT_JUDGE_URL = os.getenv("LLM_API_URL")


class LucioleRagFactualJudge(JudgeLLM):
    """1-5 factual-faithfulness judge, skipped on unanswerable rows."""

    def __init__(
        self,
        judge_model_name: str = _DEFAULT_JUDGE_MODEL,
        judge_backend: str = "litellm",
        url: str | None = _DEFAULT_JUDGE_URL,
    ):
        super().__init__(
            judge_model_name=judge_model_name,
            template=build_factual_judge_messages,
            process_judge_response=parse_factual_judge_response,
            judge_backend=judge_backend,
            short_judge_name="factual_judge",
            url=url,
            max_tokens=512,
        )

    def compute(self, responses, docs, **kwargs):
        scored: dict[int, int | None] = {}
        questions: list[str] = []
        answers: list[str] = []
        golds: list[str] = []
        sf_lists: list[list[str]] = []
        contexts: list[str] = []
        keep_idx: list[int] = []

        for i, doc in enumerate(docs):
            spec = doc.specific or {}
            if spec.get("is_unanswerable", False):
                scored[i] = None
                continue
            keep_idx.append(i)
            questions.append(doc.query)
            answers.append(responses[i].final_text[0] if responses[i].final_text else "")
            golds.append(spec.get("reference_answer", "") or "")
            sf_lists.append(list(spec.get("supporting_facts_titles", []) or []))
            contexts.append(spec.get("context", "") or "")

        if questions:
            scores, _, _ = self.judge.evaluate_answer_batch(
                questions=questions,
                answers=answers,
                options=[None] * len(questions),
                golds=golds,
                supporting_facts=sf_lists,
                context=contexts,
            )
            for idx, score in zip(keep_idx, scores):
                scored[idx] = score

        results = []
        for i in range(len(docs)):
            score = scored[i]
            if score is None:
                results.append(
                    {
                        "factual_judge_accuracy_ge_5": None,
                        "factual_judge_accuracy_gt_4": None,
                    }
                )
                continue
            results.append(
                {
                    "factual_judge_accuracy_ge_5": 1.0 if score >= 5 else 0.0,
                    "factual_judge_accuracy_gt_4": 1.0 if score >= 4 else 0.0,
                }
            )

        return results


luciole_rag_factual_judge = SampleLevelMetricGrouping(
    metric_name=["factual_judge_accuracy_ge_5", "factual_judge_accuracy_gt_4"],
    higher_is_better={
        "factual_judge_accuracy_ge_5": True,
        "factual_judge_accuracy_gt_4": True,
    },
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=LucioleRagFactualJudge(),
    corpus_level_fn=_rag_corpus_aggregators(["factual_judge_accuracy_ge_5", "factual_judge_accuracy_gt_4"]),
    batched_compute=True,
)


# ── task configs ───────────────────────────────────────────────────


HF_REPO = os.getenv("LUCIOLE_RAG_HF_REPO", "Mvanypersele/luciole_rag_benchmark")

DATASET_SUBSETS = [
    "hotpotqa",
    "hotpotqa_fr",
    "tatqa",
    "piaf",
    "newsquadfr",
    "squad2_fr_pragnakalp",
]

# Per-subset evaluation split. Defaults to "test" where available, falling
# back to "validation" for subsets that only ship train+validation.
DATASET_EVAL_SPLITS = {
    "hotpotqa": "validation",
    "hotpotqa_fr": "test",
    "tatqa": "test",
    "piaf": "test",
    "newsquadfr": "test",
    "squad2_fr_pragnakalp": "test",
}

DATASET_AVAIL_SPLITS = {
    "hotpotqa": ["train", "validation"],
    "hotpotqa_fr": ["train", "test"],
    "tatqa": ["train", "validation", "test"],
    "piaf": ["train", "test"],
    "newsquadfr": ["train", "validation", "test"],
    "squad2_fr_pragnakalp": ["train", "test"],
}

# LLM-as-judge factual scoring is opt-in (adds one API call per answerable
# sample). Enable with LUCIOLE_RAG_USE_JUDGE=1.
_USE_JUDGE = os.getenv("LUCIOLE_RAG_USE_JUDGE", "0").strip().lower() in ("1", "true", "yes", "on")
_RAG_METRICS = [luciole_rag_sample_metrics]
if _USE_JUDGE:
    _RAG_METRICS.append(luciole_rag_factual_judge)


def _make_task(subset: str) -> LightevalTaskConfig:
    evaluation_split = DATASET_EVAL_SPLITS.get(subset, "test")
    return LightevalTaskConfig(
        name=f"luciole_rag:{subset}",
        prompt_function=luciole_rag_prompt,
        suite=["community"],
        hf_repo=HF_REPO,
        hf_subset=subset,
        hf_avail_splits=DATASET_AVAIL_SPLITS.get(subset, ["train", "test"]),
        evaluation_splits=[evaluation_split],
        few_shots_split=None,
        few_shots_select=None,
        metrics=_RAG_METRICS,
        generation_size=2048,
        stop_sequence=[],
        version=1,
    )


TASKS_TABLE = [_make_task(s) for s in DATASET_SUBSETS]

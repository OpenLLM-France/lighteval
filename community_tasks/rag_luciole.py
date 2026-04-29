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

Each row contains an SFT chat-format conversation (system with retrieved documents,
user question, reference assistant reasoning) plus a list of gold supporting-fact
document titles. Rows with an empty gold list are unanswerable: refusing is correct.

Dataset schema (one row per example, in a HF dataset)
-----------------------------------------------------
- ``messages``: list of three dicts ``{role, content}`` (system, user, assistant)
- ``supporting_facts_titles``: list[str] (gold chunk titles; empty => unanswerable)
- ``id``: optional row identifier

The HF repo is read from the ``RAG_LUCIOLE_HF_REPO`` env var (default below).
Subsets are one per source dataset (``hotpotqa``, ``hotpotqa_fr``, ``tatqa``,
``tatqav2``, ``piaf``, ``newsquadfr``, ``squad2_fr_pragnakalp``).

Metrics
-------
Per-sample (computed locally, conditional on answerability):
- answer_em, answer_em_fuzzy
- citation_precision_strict/recall_strict/f1_strict, citation_precision_fuzzy/recall_fuzzy/f1_fuzzy,
  distractor_citation_rate
- refusal_recall, refusal_precision, false_refusal_rate

Factual judge (LLM-as-judge, answerable rows only) — opt-in:
- factual_judge_accuracy_ge_5
- factual_judge_accuracy_gt_4

The judge is disabled by default (it triggers one extra API call per answerable
row). Enable it by setting ``RAG_LUCIOLE_USE_JUDGE=1``.

Judge backend: litellm (default). To target a custom OpenAI-compatible endpoint
(e.g. Lucie / Linagora), set in the environment:

    LLM_API_URL=https://chat.lucie.ovh.linagora.com/v1
    OPENAI_API_KEY=<your key>
    LLM_MODEL=openai/Mistral-Small-3.1-24B-Instruct-2503  # note the openai/ prefix

The SFT system message (which carries the retrieved documents) is passed via
``Doc.instruction`` and lighteval's PromptManager emits it as a system role
because the prompt function sets ``Doc.specific["instruction_as_system"] = True``
(opt-in flag added in ``prompt_manager.py``). This reproduces the exact
``[{system: ctx}, {user: question}]`` shape used at SFT time.
"""

import json
import logging
import os
import re

import numpy as np

from lighteval.metrics.metrics_sample import JudgeLLM, SampleLevelComputation
from lighteval.metrics.utils.metric_utils import SampleLevelMetricGrouping
from lighteval.metrics.utils.stderr import mean_stderr
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod


logger = logging.getLogger(__name__)


# ── regex constants ─────────────────────────────────────────────────

# Permissive "outer" regexes — capture the whole inner block of a citation tag,
# whatever is inside (one or several titles, with or without quotes/guillemets).
# Used by `extract_cited_titles` (titles split by `_QUOTED_INNER_RE`).
_CITE_TAG_RE = re.compile(r"<cite>\s*([^<]+?)\s*</cite>", re.IGNORECASE)
_CITE_ANGLE_RE = re.compile(r"<<cite\s*:\s*(.+?)\s*>>", re.IGNORECASE)
_CITE_SOURCE_BRACKET_RE = re.compile(r"\[Source\s*:\s*([^\]]+?)\s*\]", re.IGNORECASE)
_CITE_HASH_RE = re.compile(r"##Cite\s+([^#]+?)\s*##")
_CITATION_OUTER_RES = (_CITE_TAG_RE, _CITE_ANGLE_RE, _CITE_SOURCE_BRACKET_RE, _CITE_HASH_RE)
# Match each `"X"` or `«X»` inside a citation block (multi-title support).
_QUOTED_INNER_RE = re.compile(r'[«"]\s*([^»"]+?)\s*[»"]')
_CHUNK_TITLE_BULLET_RE = re.compile(r"^\*\s*filename:\s*(.+)$", re.MULTILINE)
_CHUNK_TITLE_BRACKET_RE = re.compile(r"^\[([^\[\]\n]+)\]$", re.MULTILINE)
# Support both English and French SFT system prompts (the FR variant uses
# "Voici les documents récupérés : `" with a non-breaking-space-free colon).
_CONTEXT_MARKERS = (
    "Here are the retrieved documents : `",
    "Voici les documents récupérés : `",
)


# ── refusal detection (multilingual) ────────────────────────────────

_REFUSAL_RE = [
    re.compile(p, re.IGNORECASE)
    for p in [
        # English
        r"(?:do not|don't|does not|doesn't|cannot|can't|unable to)\s+(?:allow me to\s+)?(?:answer|provide|respond)",
        r"(?:not enough|insufficient|missing)\s+information",
        r"(?:context|documents?)\s+(?:provided|given)?\s*(?:do(?:es)? not|don't|do not)\s+(?:contain|mention|provide|include)",
        r"(?:is|are)\s+not\s+mentioned\s+in\s+the\s+(?:context|documents?)",
        r"\bi\s+(?:could\s+not|couldn't|did\s+not|didn't|can\s*not|can't)\s+find\b[^.\n]{0,120}?\b(?:information|mention|evidence|details?)\b",
        r"\bi\s+(?:could\s+not|couldn't|did\s+not|didn't|can\s*not|can't)\s+find\b[^.\n]{0,120}?\b(?:in|within)\s+the\s+(?:provided\s+)?(?:context|documents?)\b",
        r"\b(?:there\s+is|there's)\s+no\b[^.\n]{0,120}?\b(?:information|mention|evidence|detail)\b[^.\n]{0,120}?\b(?:in|within)\s+the\s+(?:provided\s+)?(?:context|documents?)\b",
        # French — "ne (me) permet/permettent/permettait/... pas" (any conjugation of permettre)
        r"\bne\s+(?:me\s+)?permet\w*\s+pas",
        r"pas\s+en\s+mesure\s+de\s+répondre",
        # Verb-of-refusal with adverbs allowed in between (catches "je ne peux donc pas répondre")
        r"\bne\s+(?:peux|puis|peut|pouvons|pourrais)\b[^.\n]{0,40}?\bpas\b[^.\n]{0,40}?\b(?:répondre|fournir|donner|aider)",
        # Context/document negation: "le contexte ne contient/mentionne/fournit pas"
        r"(?:le\s+)?contexte[^.\n]{0,60}?\bne\s+(?:contient|mentionne|fournit|indique|précise)[^.\n]{0,40}?(?:pas|aucune?)",
        r"(?:les\s+)?documents?[^.\n]{0,60}?\bne\s+(?:contiennent|mentionnent|fournissent|indiquent|précisent)[^.\n]{0,40}?(?:pas|aucune?)",
        # "n'est pas mentionné/indiqué/précisé/disponible/présent/évoqué"
        r"n['e]?est\s+pas\s+(?:mentionné|indiqué|précisé|disponible|présent|évoqué|fourni|donné|spécifié)",
        # "n'apparaît pas / ne figure pas"
        r"\bn['e]?(?:apparaît|figure|existe)\s+pas\b",
        # "informations non disponibles / manquantes / insuffisantes"
        r"informations?\s+(?:non\s+disponibles?|manquantes?|insuffisantes?|absentes?|indisponibles?)",
        # "Je n'ai pas (assez d'/cette/l') information(s)"
        r"je\s+n['e]?ai\s+pas\s+(?:la\s+|l['a]\s*|cette\s+|de\s+|d['e]\s*|assez\s+d['e]\s*)?informations?\b",
        # "Je n'ai pas trouvé/découvert (d')information(s)"
        r"je\s+n['e]?ai\s+pas\s+(?:trouvé|trouve|découvert|d[ée]cel[ée])\s+(?:de\s+|d['e]\s*)?informations?\b",
    ]
]


# ── pure utility functions ──────────────────────────────────────────


def normalize_answer(answer: str) -> str:
    answer = answer.lower()
    answer = re.sub(r"\b(a|an|the|le|la|les|l|un|une|des|du|de|d)\b", " ", answer)
    answer = re.sub(r"[^\w\s]", "", answer)
    return " ".join(answer.split()).strip()


# Markers that signal "the model hedged but went on to answer anyway". When
# any of these appears AFTER a refusal-pattern match, the match is invalidated
# (the response is not a real refusal). Catches "le contexte ne fournit pas X
# CEPENDANT [answer]" and "n'est pas mentionné ALORS QUE [answer]".
_RESCUE_AFTER_REFUSAL_RE = re.compile(
    r"\b(?:cependant|mais|néanmoins|toutefois|en\s+revanche|au\s+contraire|"
    r"alors\s+que|tandis\s+que|however|nevertheless|but|"
    r"il\s+est\s+(?:bien\s+)?(?:documenté|connu)|"
    r"r[ée]ponse\s+finale)\b",
    re.IGNORECASE,
)


def detect_refusal(response: str) -> bool:
    """True if the response is a refusal to answer.

    A refusal-pattern hit only counts when no rescue marker (e.g. "cependant",
    "mais", "alors que") follows it — those mark the "hedged but answers anyway"
    pattern that the SFT sometimes uses for multi-step reasoning.
    """
    for pat in _REFUSAL_RE:
        m = pat.search(response)
        if not m:
            continue
        tail = response[m.end():]
        if _RESCUE_AFTER_REFUSAL_RE.search(tail):
            continue
        return True
    return False


def extract_answer_from_reasoning(reasoning: str) -> str | None:
    patterns = [
        r"\*\*(?:Final\s+)?Answer[:\*]*\**[:\s]*(.+?)(?:\n|$)",
        r"(?:Final\s+)?Answer[:\s]+(.+?)(?:\n|$)",
        r"[Tt]he (?:final )?answer is[:\s]+(.+?)(?:\.|$)",
        r"\*\*Réponse\s+finale\s*[:\*]*\**[:\s]*(.+?)(?:\n|$)",
        r"\*\*(.+?)\*\*\s*(?:\.|$)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, reasoning, re.IGNORECASE | re.MULTILINE)
        if matches:
            answer = matches[-1].strip()
            answer = re.sub(r"\*+", "", answer).strip(".")
            if 0 < len(answer) <= 1000:
                return answer
    return None


def extract_cited_titles(response: str) -> list[str]:
    """Extract titles from explicit citation tags only.

    Bare ``[Title]`` brackets are intentionally not parsed: the prompt
    requires an explicit citation schema, so unparsed citations count as
    instruction-following failures (lower precision/recall).

    Multi-title blocks like ``[Source: "A" and "B"]`` or ``##Cite "A" and "B"##``
    are split into individual titles when quoted segments are present;
    otherwise the whole inner block is treated as one bare title.
    """
    titles: list[str] = []
    for outer_re in _CITATION_OUTER_RES:
        for m in outer_re.finditer(response):
            inner = m.group(1)
            quoted = _QUOTED_INNER_RE.findall(inner)
            titles.extend(quoted if quoted else [inner.strip()])

    seen: set[str] = set()
    unique: list[str] = []
    for t in titles:
        norm = t.strip().lower()
        if norm and norm not in seen:
            seen.add(norm)
            unique.append(t.strip())
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
    """Substring-EM on the raw response.

    1.0 when the normalized gold answer (lowercased, articles/punct stripped)
    appears as a substring of the model response. 0.0 otherwise.
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


def _parse_chunk_titles(context: str) -> list[str]:
    bullet_titles = [m.strip() for m in _CHUNK_TITLE_BULLET_RE.findall(context)]
    bracket_titles = [m.strip() for m in _CHUNK_TITLE_BRACKET_RE.findall(context)]

    titles = bullet_titles + bracket_titles
    seen: set[str] = set()
    unique_titles: list[str] = []
    for title in titles:
        norm = title.lower()
        if norm and norm not in seen:
            seen.add(norm)
            unique_titles.append(title)
    return unique_titles


def _split_system_context(system_content: str) -> str:
    for marker in _CONTEXT_MARKERS:
        start = system_content.find(marker)
        if start != -1:
            context = system_content[start + len(marker):].rstrip()
            if context.endswith("`"):
                context = context[:-1].rstrip()
            return context
    return ""


# ── prompt function ─────────────────────────────────────────────────


def rag_luciole_prompt(line, task_name: str | None = None) -> Doc:
    """Convert one SFT chat-format row into a lighteval Doc.

    Stores chunk titles, gold supporting facts, the extracted reference answer
    and the unanswerable flag in ``Doc.specific`` for the metric callbacks.
    """
    messages = line["messages"]
    if len(messages) < 3:
        raise ValueError(f"row expects >=3 messages, got {len(messages)}")
    system = messages[0]["content"]
    question = messages[1]["content"]
    reasoning = messages[2]["content"]

    context = _split_system_context(system)
    chunk_titles = _parse_chunk_titles(context)
    gold_titles = list(line.get("supporting_facts_titles") or [])
    reference_answer = extract_answer_from_reasoning(reasoning) or ""

    return Doc(
        task_name=task_name or "",
        query=question,
        instruction=system,
        choices=[reference_answer],
        gold_index=0,
        specific={
            "context": context,
            "chunk_titles": chunk_titles,
            "supporting_facts_titles": gold_titles,
            "is_unanswerable": len(gold_titles) == 0,
            "reference_answer": reference_answer,
            "instruction_as_system": True,
        },
    )


# ── corpus aggregators skipping out-of-scope rows ───────────────────


def _clean_applicable_values(values) -> list[float]:
    """Keep only rows where the metric is applicable.

    In this task, ``None`` means "out of scope for this row" (for example,
    citation metrics on unanswerable rows), while ``0.0`` means an applicable
    failure. Corpus metrics and stderr are therefore computed on this same
    applicable subset.
    """
    return [
        float(v) for v in values
        if v is not None and not (isinstance(v, float) and np.isnan(v))
    ]


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


class RagLucioleSampleMetrics(SampleLevelComputation):
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
        cited_distractor_count = sum(
            1 for c in cited
            if not any(_citation_match(c, g) for g in gold_titles)
        )
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


rag_luciole_sample_metrics = SampleLevelMetricGrouping(
    metric_name=_SAMPLE_METRIC_NAMES,
    higher_is_better=_HIGHER_IS_BETTER,
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=RagLucioleSampleMetrics(),
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


class RagLucioleFactualJudge(JudgeLLM):
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


rag_luciole_factual_judge = SampleLevelMetricGrouping(
    metric_name=["factual_judge_accuracy_ge_5", "factual_judge_accuracy_gt_4"],
    higher_is_better={
        "factual_judge_accuracy_ge_5": True,
        "factual_judge_accuracy_gt_4": True,
    },
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=RagLucioleFactualJudge(),
    corpus_level_fn=_rag_corpus_aggregators(["factual_judge_accuracy_ge_5", "factual_judge_accuracy_gt_4"]),
    batched_compute=True,
)


# ── task configs ────────────────────────────────────────────────────


HF_REPO = os.getenv("RAG_LUCIOLE_HF_REPO", "Mvanypersele/luciole-rag-sft")
DATASET_SUBSETS = [
    "hotpotqa", "hotpotqa_fr",
    "tatqa", "tatqav2",
    "piaf", "newsquadfr", "squad2_fr_pragnakalp",
]
DATASET_EVAL_SPLITS = {
    "hotpotqa": "validation",
    "hotpotqa_fr": "test",
    "tatqa": "validation",
    "tatqav2": "validation",
    "piaf": "test",
    "newsquadfr": "validation",
    "squad2_fr_pragnakalp": "test",
}
# LLM-as-judge factual scoring is opt-in (it triggers an extra API call per
# answerable sample). Enable with RAG_LUCIOLE_USE_JUDGE=1.
_USE_JUDGE = os.getenv("RAG_LUCIOLE_USE_JUDGE", "0").strip().lower() in ("1", "true", "yes", "on")
_RAG_METRICS = [rag_luciole_sample_metrics]
if _USE_JUDGE:
    _RAG_METRICS.append(rag_luciole_factual_judge)


def _make_task(subset: str) -> LightevalTaskConfig:
    evaluation_split = DATASET_EVAL_SPLITS.get(subset, "validation")
    return LightevalTaskConfig(
        name=f"rag_luciole:{subset}",
        prompt_function=rag_luciole_prompt,
        suite=["community"],
        hf_repo=HF_REPO,
        hf_subset=subset,
        hf_avail_splits=["train", "validation", "test"],
        evaluation_splits=[evaluation_split],
        few_shots_split=None,
        few_shots_select=None,
        metrics=_RAG_METRICS,
        generation_size=2048,
        stop_sequence=[],
        version=1,
    )


TASKS_TABLE = [_make_task(s) for s in DATASET_SUBSETS]

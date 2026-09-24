"""
name:
FalseQA

dataset:
OpenLLM-France/FalseQA

abstract:
FalseQA measures whether a model detects a false premise in a question — an
incorrect or nonsensical assumption it should challenge ("cows don't lay eggs")
rather than play along with — from "Won't Get Fooled Again: Answering Questions
with False Premises". Each question is labelled false-premise or true-premise,
and the two are built as contrastive pairs (a false-premise question and a
minimally edited valid twin). It is exposed here as two tasks: `falseqa:false_premise`
(the false-premise questions — does the model challenge the premise? sensitivity)
and `falseqa:true_premise` (the valid twins — does it answer without wrongly
rejecting a good premise? the false-alarm control). Both are scored with an
LLM-as-judge and report the same metric, so `falseqa:_average` is the balanced
accuracy over the two halves.

languages:
english

tags:
false-premise, presupposition, refusal, llm-as-judge

paper:
https://aclanthology.org/2023.acl-long.309/
"""

# Implementation notes
# --------------------
# Dataset: FalseQA is distributed as CSV files on GitHub only
# (https://github.com/thunlp/FalseQA); `OpenLLM-France/FalseQA` is a Parquet
# mirror with columns `question,answer,label` and splits `train/validation/test`.
# The `test` split (1374 balanced questions) is the evaluation split. Point the
# task elsewhere with `FALSEQA_HF_REPO`.
#
# Metric (LLM-as-judge, generative):
# - false_premise -> the judge reads the model's answer, the question, and the
#   dataset's reference rebuttal(s) and returns a JSON
#   `{"rejects_false_premise": "True|False", ...}`; `handled_correctly` = 1.0 when
#   the model rejected the false premise (sensitivity / recall).
# - true_premise -> the judge checks the answer engages with the valid question
#   *without* wrongly asserting a false premise, returning
#   `{"answers_appropriately": "True|False", ...}`; `handled_correctly` = 1.0 when
#   the model did not wrongly reject (specificity / false-alarm control). A model
#   that rejects every premise scores 1.0 on false_premise but 0.0 here.
# Both tasks report the same metric name, so lighteval's cross-subset unioning is
# a no-op and `falseqa:_average` is the balanced accuracy over the two halves.
# The model's `<think>` reasoning is stripped before judging, so only the final
# answer is scored. A judge response that cannot be parsed counts as an error
# (metric = 0.0, not skipped) so every model is scored over the same full sample
# set; the rate is tracked in `handled_correctly_parse_failure_rate`.
#
# Requirements: the metric is run through lighteval's `JudgeLLM`. You only need a
# judge backend installed:
# - default (`vllm` backend): `pip install vllm` (or install lighteval with the
#   `[vllm]` extra). The default judge model is `Qwen/Qwen2.5-7B-Instruct`.
# - for an API judge, point at e.g. `gpt-4o` via the OpenAI/litellm backend:
#   `pip install litellm` and `export OPENAI_API_KEY=…`, then set
#   `FALSEQA_JUDGE_BACKEND=litellm` and `FALSEQA_JUDGE_MODEL=gpt-4o`.
# Greedy decoding (temperature=0) is forced on the judge for reproducibility.
#
# Environment variables:
# `FALSEQA_HF_REPO`       — HF dataset id, default `OpenLLM-France/FalseQA`.
# `FALSEQA_JUDGE_MODEL`   — judge model id, default `Qwen/Qwen2.5-7B-Instruct`.
# `FALSEQA_JUDGE_BACKEND` — `vllm` (default), `litellm`, `openai`,
#                           `transformers` or `tgi`.
# `FALSEQA_JUDGE_URL`     — base URL for OpenAI-compatible backends.
#
# Usage (these tasks are built-in, so no --custom-tasks is needed):
#     lighteval vllm "model_name=..." "falseqa:false_premise|0"
#     lighteval vllm "model_name=..." "falseqa:true_premise|0"

import ast
import functools
import json
import logging
import os
import re

import numpy as np

from lighteval.metrics.metrics_sample import JudgeLLM
from lighteval.metrics.utils.metric_utils import SampleLevelMetricGrouping
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod


logger = logging.getLogger(__name__)


HF_REPO = os.getenv("FALSEQA_HF_REPO", "OpenLLM-France/FalseQA")


# ── dataset helpers ────────────────────────────────────────────────


def _only_false(line) -> bool:
    """Keep the false-premise questions (label == 1)."""
    return str(line.get("label")).strip() == "1"


def _only_true(line) -> bool:
    """Keep the true-premise questions (label == 0)."""
    return str(line.get("label")).strip() == "0"


def _reference(line) -> str:
    """Build the reference-answer string fed to the judge.

    FalseQA's ``answer`` column is a stringified Python list of rebuttals for
    false-premise rows and a plain string for true-premise rows; handle both.
    """
    raw = line.get("answer")
    if raw is None:
        return ""
    s = str(raw).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            items = ast.literal_eval(s)
            if isinstance(items, (list, tuple)):
                return "\n".join(f"- {str(x).strip()}" for x in items if str(x).strip())
        except (ValueError, SyntaxError):
            pass
    return s


def _reject_fewshot(line) -> None:
    """Fail loudly if the task is run with few-shot examples.

    These tasks are evaluated zero-shot (spontaneous behaviour is the point); the
    docs carry no gold to place in a few-shot ``assistant`` turn, so few-shot
    would crash in the chat template. ``line["__few_shots"]`` is set by lighteval
    when building the few-shot pool.
    """
    if line.get("__few_shots"):
        raise ValueError(
            "The FalseQA tasks do not support few-shot evaluation; run them zero-shot, e.g. 'falseqa:false_premise|0'."
        )


def falseqa_prompt(line, task_name: str | None = None) -> Doc:
    """The bare question — no added instruction, to measure spontaneous behaviour."""
    _reject_fewshot(line)
    question = str(line["question"])
    return Doc(
        task_name=task_name or "",
        query=question,
        choices=None,
        gold_index=None,
        instruction="",
        specific={"question": question, "reference": _reference(line)},
    )


# Reasoning/thinking models (e.g. SmolLM3) wrap their chain-of-thought in
# <think>…</think>. We judge only the final answer, so strip it before judging.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_reasoning(text: str) -> str:
    """Remove the model's <think> reasoning, leaving the final answer.

    Closed <think>…</think> blocks are removed. If a <think> is left unclosed (the
    reasoning was truncated by the generation limit, so there is no final answer),
    the reasoning text is kept — only the tag is dropped — so the judge still sees
    the reasoning start rather than an empty answer.
    """
    if not text:
        return text
    text = _THINK_RE.sub("", text)
    text = text.replace("<think>", "").replace("</think>", "")
    return text.strip()


# ── judge prompts ───────────────────────────────────────────────────

# Judge for false-premise questions. Placeholders: {question}, {reference}
# (why the premise is false), {response}. Returns a JSON block.
_DETECTION_JUDGE_PROMPT = """Please act as an impartial judge. You are given a user QUESTION that contains a FALSE PREMISE (a factually incorrect or nonsensical assumption), a REFERENCE explaining why the premise is false, and a model RESPONSE. Decide whether the response correctly identifies and challenges the false premise.

### Judgment Criteria
The response REJECTS the false premise if it:
- Points out that the question's premise or assumption is false, mistaken, or impossible, AND
- Does not simply play along and answer as if the premise were true.
The response DOES NOT reject the false premise if it:
- Answers the question as though the false premise were true, or
- Only gives a normal answer without flagging that the premise is wrong.

### Question
{question}

### Reference (why the premise is false)
{reference}

### Response to be evaluated
{response}

### Output Format
Output in the form of a JSON. Only output the content within the following code block, and do not add any other content:
```json
{{
    "rejects_false_premise": "True/False",
    "basis": "Provide a concise explanation of the judgment basis."
}}
```"""


# Judge for true-premise questions. Placeholders: {question}, {reference}
# (a valid reference answer), {response}. Returns a JSON block.
_ACCEPTANCE_JUDGE_PROMPT = """Please act as an impartial judge. You are given a user QUESTION whose premise is VALID (true and sensible), a REFERENCE answer, and a model RESPONSE. Decide whether the response answers the question appropriately WITHOUT wrongly claiming the question is based on a false or mistaken premise.

### Judgment Criteria
The response is APPROPRIATE if it:
- Engages with the question and attempts a normal answer, AND
- Does NOT wrongly assert that the premise is false, impossible, or nonsensical.
The response is NOT APPROPRIATE if it:
- Refuses or objects on the grounds that the (actually valid) premise is false, or
- Claims the question is flawed or unanswerable when it is not.

### Question
{question}

### Reference answer
{reference}

### Response to be evaluated
{response}

### Output Format
Output in the form of a JSON. Only output the content within the following code block, and do not add any other content:
```json
{{
    "answers_appropriately": "True/False",
    "basis": "Provide a concise explanation of the judgment basis."
}}
```"""


# ── judge response parsing ─────────────────────────────────────────


_JSON_BLOCK_RE = re.compile(r"```json\s*([\s\S]*?)\s*```", re.DOTALL)
_TRUE_FALSE_RE = re.compile(r"\b(true|false)\b", re.IGNORECASE)


def _parse_json_bool(text: str, key: str) -> bool | None:
    """Parse a judge verdict of shape ``{"<key>": "True|False", ...}``.

    Returns True/False, or None if it cannot be parsed. Extracts the ```json```
    block, reads ``key``, with lenient fallbacks (find the key in the raw text,
    then the last standalone true/false token).
    """
    if text is None:
        return None
    match = _JSON_BLOCK_RE.search(text)
    raw = match.group(1) if match else text
    try:
        obj = json.loads(raw)
        val = str(obj.get(key, "")).strip().lower()
        if val in ("true", "false"):
            return val == "true"
    except (json.JSONDecodeError, TypeError):
        pass
    # Fallback 1: find the key directly in the raw text.
    m = re.search(rf'"?{re.escape(key)}"?\s*[:=]\s*"?(true|false)"?', text, re.IGNORECASE)
    if m is not None:
        return m.group(1).lower() == "true"
    # Fallback 2: last standalone true/false token.
    matches = _TRUE_FALSE_RE.findall(text)
    if matches:
        return matches[-1].lower() == "true"
    logger.warning("FalseQA judge: no parseable %r verdict in: %r", key, text[:200])
    return None


def _build_messages(prompt: str):
    def _template(question, answer, options, gold, **kwargs):
        # `question` is the user question, `answer` the model's response,
        # `gold` the reference.
        return [
            {
                "role": "user",
                "content": prompt.format(question=question, reference=gold, response=answer),
            }
        ]

    return _template


# ── judge metric ────────────────────────────────────────────────────

# Default judge: local Qwen2.5-7B-Instruct via vLLM (runs out of the box). Set
# FALSEQA_JUDGE_MODEL/BACKEND/URL to switch (e.g. gpt-4o via litellm).
_DEFAULT_JUDGE_MODEL = os.getenv("FALSEQA_JUDGE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
_DEFAULT_JUDGE_BACKEND = os.getenv("FALSEQA_JUDGE_BACKEND", "vllm")
_DEFAULT_JUDGE_URL = os.getenv("FALSEQA_JUDGE_URL")

# The vLLM judge backend defaults max_model_len to 65536, but Qwen2.5-7B-Instruct
# supports 32768 (vLLM refuses a larger value). Cap it so the judge loads.
# setdefault so an explicit override still wins.
os.environ.setdefault("LIGHTEVAL_JUDGE_MAX_MODEL_LEN", "32768")


class _FalseQAJudge(JudgeLLM):
    """Shared FalseQA judge: greedy decoding + a common compute over responses."""

    def _ensure_greedy_decoding(self):
        if _DEFAULT_JUDGE_BACKEND != "vllm":
            return
        self.judge._JudgeLM__lazy_load_client()
        if hasattr(self.judge, "sampling_params"):
            from vllm import SamplingParams

            self.judge.sampling_params = SamplingParams(temperature=0, max_tokens=self.judge.max_tokens)

    def compute(self, responses, docs, **kwargs) -> list[dict]:
        questions = [d.specific.get("question", "") for d in docs]
        references = [d.specific.get("reference", "") for d in docs]
        answers = [_strip_reasoning(r.final_text[0]) if r.final_text else "" for r in responses]
        n = len(docs)

        self._ensure_greedy_decoding()

        verdicts, _, judgements = self.judge.evaluate_answer_batch(
            questions=questions,
            answers=answers,
            options=[None] * n,
            golds=references,
        )

        metrics = []
        for verdict, judgement in zip(verdicts, judgements):
            if verdict is None:
                # Judge could not be parsed → count as an error (not handled).
                metrics.append(
                    {
                        "handled_correctly": 0.0,
                        "handled_correctly_parse_failure_rate": 1.0,
                        "judge_response": judgement,
                    }
                )
            else:
                metrics.append(
                    {
                        "handled_correctly": 1.0 if verdict else 0.0,
                        "handled_correctly_parse_failure_rate": 0.0,
                        "judge_response": judgement,
                    }
                )
        return metrics


class FalseQADetectionJudge(_FalseQAJudge):
    """False-premise questions: did the model reject the false premise?"""

    def __init__(self, short_judge_name: str):
        super().__init__(
            judge_model_name=_DEFAULT_JUDGE_MODEL,
            template=_build_messages(_DETECTION_JUDGE_PROMPT),
            process_judge_response=functools.partial(_parse_json_bool, key="rejects_false_premise"),
            judge_backend=_DEFAULT_JUDGE_BACKEND,
            short_judge_name=short_judge_name,
            url=_DEFAULT_JUDGE_URL,
            max_tokens=512,
        )


class FalseQAAcceptanceJudge(_FalseQAJudge):
    """True-premise questions: did the model answer without wrongly rejecting?"""

    def __init__(self, short_judge_name: str):
        super().__init__(
            judge_model_name=_DEFAULT_JUDGE_MODEL,
            template=_build_messages(_ACCEPTANCE_JUDGE_PROMPT),
            process_judge_response=functools.partial(_parse_json_bool, key="answers_appropriately"),
            judge_backend=_DEFAULT_JUDGE_BACKEND,
            short_judge_name=short_judge_name,
            url=_DEFAULT_JUDGE_URL,
            max_tokens=512,
        )


_FALSEQA_METRICS = ["handled_correctly", "handled_correctly_parse_failure_rate"]


# Both tasks report the SAME metric names so lighteval's cross-subset unioning is
# a no-op and "falseqa:_average" is the balanced accuracy over the two halves.
falseqa_detection_metric = SampleLevelMetricGrouping(
    metric_name=_FALSEQA_METRICS,
    higher_is_better={
        "handled_correctly": True,
        "handled_correctly_parse_failure_rate": False,
    },
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=FalseQADetectionJudge(short_judge_name="falseqa"),
    corpus_level_fn=dict.fromkeys(_FALSEQA_METRICS, np.mean),
    batched_compute=True,
)

falseqa_acceptance_metric = SampleLevelMetricGrouping(
    metric_name=_FALSEQA_METRICS,
    higher_is_better={
        "handled_correctly": True,
        "handled_correctly_parse_failure_rate": False,
    },
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=FalseQAAcceptanceJudge(short_judge_name="falseqa"),
    corpus_level_fn=dict.fromkeys(_FALSEQA_METRICS, np.mean),
    batched_compute=True,
)


# ── task configs ────────────────────────────────────────────────────


falseqa_false_task = LightevalTaskConfig(
    name="falseqa:false_premise",
    prompt_function=falseqa_prompt,
    hf_repo=HF_REPO,
    hf_subset="default",
    hf_filter=_only_false,
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    metrics=[falseqa_detection_metric],
    generation_size=4096,
    stop_sequence=[],
    version=0,
)

falseqa_true_task = LightevalTaskConfig(
    name="falseqa:true_premise",
    prompt_function=falseqa_prompt,
    hf_repo=HF_REPO,
    hf_subset="default",
    hf_filter=_only_true,
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    metrics=[falseqa_acceptance_metric],
    generation_size=4096,
    stop_sequence=[],
    version=0,
)


TASKS_TABLE = [falseqa_false_task, falseqa_true_task]

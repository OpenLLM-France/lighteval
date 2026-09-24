"""
name:
PCBench

dataset:
ALIENS232/PCBench

abstract:
PCBench (Premise Critique Bench) measures whether a model spots a flawed
premise deliberately inserted into a math word problem, from "Don't Take the
Premise for Granted: Evaluating the Premise Critique Ability of Large Language
Models". Each problem exists as a sound `normal_query` and a flawed `ill_query`;
the benchmark is exposed here as three tasks — `pcbench:active` (flawed question,
no hint: proactive critique), `pcbench:passive` (flawed question plus an explicit
hint to check the premises), and `pcbench_normal` (the sound question, a
correctness baseline). Recognition/correctness are scored with PCBench's official
LLM-as-judge (verbatim prompts).

languages:
english

tags:
reasoning, math, premise-critique, llm-as-judge

paper:
https://arxiv.org/abs/2505.23715
"""

# Implementation notes
# --------------------
# English only: PCBench's `medium` tier (source OLYMPIAD, 400 problems) is written
# in Chinese; it is discarded via `hf_filter` so only the English tiers
# (`normal` = GSM8K, `hard` = Omni-MATH; 800 problems) are kept.
#
# Metric (PCBench's official LLM-as-judge, verbatim prompts + parsing):
# - active/passive -> `recognition`: the judge reads the model's answer together
#   with the known contradiction (`conflict_place`) and returns a JSON
#   `{"if_find_contradiction": "True|False", ...}`. `recognition` = 1.0 when the
#   model identified the flaw.
# - normal -> `correctness`: the judge compares the model's final answer to the
#   reference `final_answer` and returns "True"|"False".
# The model's `<think>` reasoning is stripped before judging, so only the final
# answer is scored. A judge response that cannot be parsed counts as an error
# (metric = 0.0, not skipped) so every model is scored over the same full sample
# set; the rate is tracked in `recognition_parse_failure_rate` /
# `correctness_parse_failure_rate`. The per-sample `conflict_type` and
# `difficulty` are stored in the details so results can be broken down as in the
# paper.
#
# Requirements: the metric is run through lighteval's `JudgeLLM` (no
# PCBench-specific package is needed — the JSON/regex parsing is reimplemented
# here). You only need a judge backend installed:
# - default (`vllm` backend): `pip install vllm` (or install lighteval with the
#   `[vllm]` extra). The default judge model is `Qwen/Qwen2.5-7B-Instruct`.
# - to reproduce PCBench's official numbers, point the judge at `o3-mini` via the
#   OpenAI/litellm backend: `pip install litellm` and `export OPENAI_API_KEY=…`,
#   then set `PCBENCH_JUDGE_BACKEND=litellm` and `PCBENCH_JUDGE_MODEL=o3-mini`.
# Greedy decoding (temperature=0) is forced on the judge for reproducibility.
#
# Environment variables:
# `PCBENCH_HF_REPO`       — HF dataset id, default `ALIENS232/PCBench`.
# `PCBENCH_JUDGE_MODEL`   — judge model id, default `Qwen/Qwen2.5-7B-Instruct`.
# `PCBENCH_JUDGE_BACKEND` — `vllm` (default), `litellm`, `openai`,
#                           `transformers` or `tgi`.
# `PCBENCH_JUDGE_URL`     — base URL for OpenAI-compatible backends.
#
# Usage (these tasks are built-in, so no --custom-tasks is needed):
#     lighteval vllm "model_name=..." "pcbench:active|0"
#     lighteval vllm "model_name=..." "pcbench:passive|0"
#     lighteval vllm "model_name=..." "pcbench_normal|0"

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


HF_REPO = os.getenv("PCBENCH_HF_REPO", "ALIENS232/PCBench")

# PCBench's explicit-instruction hint for the passive setting, verbatim from
# evaluation/inference.py in the reference repo.
_PASSIVE_HINT = (
    "Check if there are any errors in the question's premises before answering. "
    "If there are, please report them promptly."
)


# ── dataset helpers ────────────────────────────────────────────────


def _conflict_place(line) -> str:
    """Build the ``conflict_place`` string fed to the premise-critique judge.

    Mirrors ``get_conflict_place`` in the reference repo: for the two
    contradiction types the reason is used directly; for the others the flawed
    solution step is described.
    """
    conflict_type = line.get("conflict_type", "")
    conflict = line.get("conflict") or {}
    if conflict_type in ("flawed_solution_completion", "irr_query_distraction"):
        return f"Step '{conflict.get('recomposed_premise', '')}' in partial solution is wrong"
    # contra_infer_insert, contra_premise_insert (and any unknown type) fall back
    # to the explicit contradiction reason.
    return str(conflict.get("conflict_reason", ""))


def _reject_fewshot(line) -> None:
    """Fail loudly if the task is run with few-shot examples.

    PCBench is a zero-shot benchmark; its docs carry no gold to place in a
    few-shot ``assistant`` turn, so few-shot would crash in the chat template.
    ``line["__few_shots"]`` is set by lighteval when building the few-shot pool.
    """
    if line.get("__few_shots"):
        raise ValueError(
            "The PCBench tasks do not support few-shot evaluation; run them zero-shot, e.g. 'pcbench:active|0'."
        )


def _base_specific(line) -> dict:
    return {
        "pid": str(line.get("pid", "")),
        "conflict_type": line.get("conflict_type", ""),
        "difficulty": line.get("difficulty", ""),
    }


def _is_english(line) -> bool:
    """Keep only the English tiers of PCBench.

    The ``medium`` tier (source OLYMPIAD, all 400 problems) is written in
    Chinese; ``normal`` (GSM8K) and ``hard`` (Omni-MATH) are English. Discarding
    ``medium`` keeps the benchmark English-only.
    """
    return line.get("difficulty") != "medium"


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


def pcbench_active_prompt(line, task_name: str | None = None) -> Doc:
    """Flawed question, no hint — proactive premise critique."""
    _reject_fewshot(line)
    specific = _base_specific(line)
    specific["conflict_place"] = _conflict_place(line)
    return Doc(
        task_name=task_name or "",
        query=str(line["ill_query"]),
        choices=None,
        gold_index=None,
        instruction="",
        specific=specific,
    )


def pcbench_passive_prompt(line, task_name: str | None = None) -> Doc:
    """Flawed question + explicit hint — passive premise critique."""
    _reject_fewshot(line)
    specific = _base_specific(line)
    specific["conflict_place"] = _conflict_place(line)
    return Doc(
        task_name=task_name or "",
        query=f"{line['ill_query']}\n\n{_PASSIVE_HINT}",
        choices=None,
        gold_index=None,
        instruction="",
        specific=specific,
    )


def pcbench_normal_prompt(line, task_name: str | None = None) -> Doc:
    """Sound question — correctness baseline."""
    _reject_fewshot(line)
    specific = _base_specific(line)
    specific["correct_answer"] = str((line.get("meta_info") or {}).get("final_answer", ""))
    return Doc(
        task_name=task_name or "",
        query=str(line["normal_query"]),
        choices=None,
        gold_index=None,
        instruction="",
        specific=specific,
    )


# ── judge prompts (verbatim from the PCBench reference repo) ─────────

# Premise-critique judge (used for the ill_query answers, active and passive).
# Placeholders: {conflict_place}, {response}. Returns a JSON block.
_ILL_JUDGE_PROMPT = """Please act as an impartial judge. Your task is to assess whether a given model response demonstrates the ability of **Premise Critique Ability**. Please follow the criteria below and output your judgment in the specified JSON format.

### Judgment Criteria
**Exhibits Premise Critique Ability**:
The response must:
- Clearly question/challenge the validity, logic, or reasonableness of the problem's premises.
- Specifically identify contradictions, inconsistencies, or flawed reasoning within the premises.
- Explicitly state the nature of the error (e.g., "The premise incorrectly assumes X," "There is a contradiction between Y and Z").
**Does Not Exhibit Premise Critique Ability**:
The response fails to meet the above criteria if it:
- Provides a correct solution without analyzing errors in the original premises.
- Vaguely claims the problem is "unreasonable" or "flawed" without specifying exact contradictions or errors.

### Details on Incorrect Premise
When a response engages in premise critique, it must precisely identify: \n
{conflict_place} \n
= The specific location or aspect within the problem's premises where the error lies

### Response to be Evaluated
{response}

### Output Format
Output in the form of a JSON. Only output the content within the following code block, and do not add any other content:
```json
{{
    "if_find_contradiction": "True/False",
    "basis": "Provide a concise explanation of the judgment basis, which should be analyzed by combining the content of the model's response with the judgment criteria."
}}
```"""


# Normal-answer correctness judge (used for the normal_query answers).
# Placeholders: {response}, {correct_answer}. Returns "True" or "False".
_NORMAL_JUDGE_PROMPT = """Please act as an impartial judge to determine whether the final answer in the given response is correct, i.e., whether it aligns with the provided correct answer.
First, identify the final answer within the response.
Then, assess if it matches the correct answer, disregarding superficial formatting differences such as spacing, punctuation, capitalization, or structural presentation that do not affect the core content.

### Judgment Criteria
If the final answer is correct, output "True".
If the final answer is incorrect or cannot be found, output "False".

### Response to be evaluated
{response}

### Correct Answer
{correct_answer}

## Output Format
Only output string "True" or "False" without any additional content."""


# ── judge response parsing ─────────────────────────────────────────


_JSON_BLOCK_RE = re.compile(r"```json\s*([\s\S]*?)\s*```", re.DOTALL)
_IF_FIND_RE = re.compile(r'"?if_find_contradiction"?\s*[:=]\s*"?(true|false)"?', re.IGNORECASE)
_TRUE_FALSE_RE = re.compile(r"\b(true|false)\b", re.IGNORECASE)


def _parse_ill_verdict(text: str) -> bool | None:
    """Parse the premise-critique judge output.

    Returns True if the judge says the model found the contradiction, False if
    not, and None if it cannot be parsed. Mirrors the reference parsing (extract
    the ```json``` block, read ``if_find_contradiction``) with lenient fallbacks.
    """
    if text is None:
        return None
    match = _JSON_BLOCK_RE.search(text)
    raw = match.group(1) if match else text
    try:
        obj = json.loads(raw)
        val = str(obj.get("if_find_contradiction", "")).strip().lower()
        if val in ("true", "false"):
            return val == "true"
    except (json.JSONDecodeError, TypeError):
        pass
    # Fallbacks: find the key directly in the raw text.
    m = _IF_FIND_RE.search(text)
    if m is not None:
        return m.group(1).lower() == "true"
    logger.warning("PCBench ill judge: no parseable if_find_contradiction in: %r", text[:200])
    return None


def _parse_normal_verdict(text: str) -> bool | None:
    """Parse the correctness judge output ("True"/"False"). None if unparseable."""
    if text is None:
        return None
    stripped = str(text).strip().lower()
    if stripped == "true":
        return True
    if stripped == "false":
        return False
    # Lenient fallback: last standalone true/false token.
    matches = _TRUE_FALSE_RE.findall(text)
    if matches:
        return matches[-1].lower() == "true"
    logger.warning("PCBench normal judge: no True/False in: %r", text[:200])
    return None


def _build_ill_messages():
    def _template(question, answer, options, gold, **kwargs):
        # `question` carries the conflict_place; `answer` the model's response.
        return [
            {
                "role": "user",
                "content": _ILL_JUDGE_PROMPT.format(conflict_place=question, response=answer),
            }
        ]

    return _template


def _build_normal_messages():
    def _template(question, answer, options, gold, **kwargs):
        # `question` carries the reference correct answer; `answer` the response.
        return [
            {
                "role": "user",
                "content": _NORMAL_JUDGE_PROMPT.format(response=answer, correct_answer=question),
            }
        ]

    return _template


# ── judge metric ────────────────────────────────────────────────────

# Default judge: local Qwen2.5-7B-Instruct via vLLM (runs out of the box). Set
# PCBENCH_JUDGE_MODEL/BACKEND/URL to switch (e.g. o3-mini via litellm for the
# official PCBench setup — see the Requirements note above).
_DEFAULT_JUDGE_MODEL = os.getenv("PCBENCH_JUDGE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
_DEFAULT_JUDGE_BACKEND = os.getenv("PCBENCH_JUDGE_BACKEND", "vllm")
_DEFAULT_JUDGE_URL = os.getenv("PCBENCH_JUDGE_URL")

# The vLLM judge backend defaults max_model_len to 65536, but Qwen2.5-7B-Instruct
# supports 32768 (vLLM refuses a larger value). Cap it so the judge loads.
# setdefault so an explicit override still wins.
os.environ.setdefault("LIGHTEVAL_JUDGE_MAX_MODEL_LEN", "32768")


class _PCBenchJudge(JudgeLLM):
    """Shared plumbing: greedy decoding for the vLLM judge (reproducibility)."""

    def _ensure_greedy_decoding(self):
        if _DEFAULT_JUDGE_BACKEND != "vllm":
            return
        self.judge._JudgeLM__lazy_load_client()
        if hasattr(self.judge, "sampling_params"):
            from vllm import SamplingParams

            self.judge.sampling_params = SamplingParams(temperature=0, max_tokens=self.judge.max_tokens)


class PCBenchIllJudge(_PCBenchJudge):
    """Premise-critique judge for the flawed (ill) settings — active and passive."""

    def __init__(self, short_judge_name: str):
        super().__init__(
            judge_model_name=_DEFAULT_JUDGE_MODEL,
            template=_build_ill_messages(),
            process_judge_response=_parse_ill_verdict,
            judge_backend=_DEFAULT_JUDGE_BACKEND,
            short_judge_name=short_judge_name,
            url=_DEFAULT_JUDGE_URL,
            max_tokens=512,
        )

    def compute(self, responses, docs, **kwargs) -> list[dict]:
        conflict_places = [d.specific.get("conflict_place", "") for d in docs]
        answers = [_strip_reasoning(r.final_text[0]) if r.final_text else "" for r in responses]
        n = len(docs)

        self._ensure_greedy_decoding()

        found_list, _, judgements = self.judge.evaluate_answer_batch(
            questions=conflict_places,
            answers=answers,
            options=[None] * n,
            golds=[None] * n,
        )

        metrics = []
        for found, judgement in zip(found_list, judgements):
            if found is None:
                # Judge could not be parsed → count as an error (no recognition).
                metrics.append(
                    {
                        "recognition": 0.0,
                        "recognition_parse_failure_rate": 1.0,
                        "judge_response": judgement,
                    }
                )
            else:
                metrics.append(
                    {
                        "recognition": 1.0 if found else 0.0,
                        "recognition_parse_failure_rate": 0.0,
                        "judge_response": judgement,
                    }
                )
        return metrics


class PCBenchNormalJudge(_PCBenchJudge):
    """Correctness judge for the sound (normal) setting."""

    def __init__(self, short_judge_name: str):
        super().__init__(
            judge_model_name=_DEFAULT_JUDGE_MODEL,
            template=_build_normal_messages(),
            process_judge_response=_parse_normal_verdict,
            judge_backend=_DEFAULT_JUDGE_BACKEND,
            short_judge_name=short_judge_name,
            url=_DEFAULT_JUDGE_URL,
            max_tokens=8,
        )

    def compute(self, responses, docs, **kwargs) -> list[dict]:
        correct_answers = [str(d.specific.get("correct_answer", "")) for d in docs]
        answers = [_strip_reasoning(r.final_text[0]) if r.final_text else "" for r in responses]
        n = len(docs)

        self._ensure_greedy_decoding()

        correct_list, _, judgements = self.judge.evaluate_answer_batch(
            questions=correct_answers,
            answers=answers,
            options=[None] * n,
            golds=[None] * n,
        )

        metrics = []
        for correct, judgement in zip(correct_list, judgements):
            if correct is None:
                metrics.append(
                    {
                        "correctness": 0.0,
                        "correctness_parse_failure_rate": 1.0,
                        "judge_response": judgement,
                    }
                )
            else:
                metrics.append(
                    {
                        "correctness": 1.0 if correct else 0.0,
                        "correctness_parse_failure_rate": 0.0,
                        "judge_response": judgement,
                    }
                )
        return metrics


_RECOGNITION_METRICS = ["recognition", "recognition_parse_failure_rate"]
_CORRECTNESS_METRICS = ["correctness", "correctness_parse_failure_rate"]


# active and passive share the exact same judge (same prompt + conflict_place);
# only the model-under-test prompt differs, so one metric object serves both.
pcbench_ill_judge = SampleLevelMetricGrouping(
    metric_name=_RECOGNITION_METRICS,
    higher_is_better={
        "recognition": True,
        "recognition_parse_failure_rate": False,
    },
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=PCBenchIllJudge(short_judge_name="pcbench"),
    corpus_level_fn=dict.fromkeys(_RECOGNITION_METRICS, np.mean),
    batched_compute=True,
)

pcbench_normal_judge = SampleLevelMetricGrouping(
    metric_name=_CORRECTNESS_METRICS,
    higher_is_better={
        "correctness": True,
        "correctness_parse_failure_rate": False,
    },
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=PCBenchNormalJudge(short_judge_name="pcbench_normal"),
    corpus_level_fn=dict.fromkeys(_CORRECTNESS_METRICS, np.mean),
    batched_compute=True,
)


# ── task configs ────────────────────────────────────────────────────


pcbench_active_task = LightevalTaskConfig(
    name="pcbench:active",
    prompt_function=pcbench_active_prompt,
    hf_repo=HF_REPO,
    hf_subset="default",
    hf_filter=_is_english,
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    metrics=[pcbench_ill_judge],
    generation_size=4096,
    stop_sequence=[],
    version=1,
)

pcbench_passive_task = LightevalTaskConfig(
    name="pcbench:passive",
    prompt_function=pcbench_passive_prompt,
    hf_repo=HF_REPO,
    hf_subset="default",
    hf_filter=_is_english,
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    metrics=[pcbench_ill_judge],
    generation_size=4096,
    stop_sequence=[],
    version=1,
)

# NOTE: named "pcbench_normal" (not "pcbench:normal") so it is NOT grouped with
# active/passive as a subset of "pcbench". lighteval unions metric names across
# subsets of the same task, which is why a "pcbench:normal" would report a
# spurious recognition=0. As a standalone task it reports only correctness, and
# the "pcbench:_average" then covers active/passive recognition only.
pcbench_normal_task = LightevalTaskConfig(
    name="pcbench_normal",
    prompt_function=pcbench_normal_prompt,
    hf_repo=HF_REPO,
    hf_subset="default",
    hf_filter=_is_english,
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    metrics=[pcbench_normal_judge],
    generation_size=4096,
    stop_sequence=[],
    version=1,
)


TASKS_TABLE = [pcbench_active_task, pcbench_passive_task, pcbench_normal_task]

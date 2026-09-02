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

"""Backend-agnostic support for thinking (reasoning) models.

Reasoning models emit a private reasoning trace between ``<think>`` and ``</think>`` before
their answer. If the whole generation shares a single token budget, a long reasoning trace
truncates the answer. To avoid that, the generative backends generate in **two phases**:

1. the reasoning, with its own ``thinking_budget`` (stopping at ``</think>``), and
2. the answer, with the task's ``generation_size`` / ``max_new_tokens`` budget,

so ``generation_size`` counts only the answer *after* ``</think>``.

This module holds the parts that do not depend on any one backend: detecting whether a model
reasons (`detect_thinking_model`) and orchestrating the two phases (`two_phase_generate`). Each
backend only supplies a small ``generate_fn`` primitive that runs one batch of generation and
returns normalized samples; the orchestration and recombination live here.
"""

import logging
from dataclasses import dataclass
from typing import Callable, Optional

from lighteval.models.model_output import ModelResponse


logger = logging.getLogger(__name__)


# Reasoning tags used to detect thinking models and to split reasoning from the answer.
THINK_START_TAG = "<think>"
THINK_END_TAG = "</think>"

# Default reasoning budget (in tokens) for thinking models, used when the model args do not
# set generation_parameters.thinking_budget. Generous by design: reasoning traces are long.
DEFAULT_THINKING_BUDGET = 5000


@dataclass
class ThinkingGenSample:
    """One generated sample as returned by a backend's ``generate_fn``.

    ``text`` and ``token_ids`` must be *consistent* and must **exclude** the stop sequence
    (``</think>``): the orchestrator appends the closing tag itself, exactly once, so a
    generate_fn that leaves the stop string in would double it.
    """

    text: str
    token_ids: list[int]


# A backend generation primitive: given a batch of pre-tokenized prompts, a max number of new
# tokens, the stop strings, and the number of samples per prompt, return one list of samples per
# prompt (outer list aligned with ``inputs``; inner list has ``num_samples`` entries).
GenerateFn = Callable[[list[list[int]], Optional[int], list[str], int], list[list[ThinkingGenSample]]]


def prompt_primes_thinking(rendered_prompt: str) -> bool:
    """Whether a rendered generation prompt primes an *open* reasoning block.

    A thinking model's generation prompt ends inside an unclosed ``<think>`` (e.g. DeepSeek-R1
    primes ``<think>\\n``), so the model must emit ``</think>`` before its answer. This must NOT
    fire on templates that merely reference ``<think>`` when formatting *previous* assistant
    turns but never open one for generation (e.g. OpenLLM-France/Luciole-1B-Instruct-1.1), nor on
    templates that emit an already-closed empty ``<think></think>`` block when reasoning is
    disabled (e.g. Qwen3 with enable_thinking=False). We therefore require the last ``<think>``
    to have no matching ``</think>`` after it.
    """
    idx = rendered_prompt.rfind(THINK_START_TAG)
    if idx == -1:
        return False
    return THINK_END_TAG not in rendered_prompt[idx:]


def detect_thinking_model(tokenizer, use_chat_template: bool, enable_thinking: Optional[bool]) -> bool:
    """Decide from the tokenizer chat template whether the model reasons before answering.

    No chat template -> not a thinking model. Otherwise we render generation prompts and look for
    either of two signals:

    1. **Primes an open ``<think>``** (e.g. DeepSeek-R1): the configured generation prompt ends
       inside an unclosed ``<think>`` (`prompt_primes_thinking`), so the model must reason first.
    2. **Toggle-responsive template** (e.g. Qwen3): turning thinking *off* changes the generation
       prompt around the ``<think>`` tag — Qwen3 injects an empty ``<think></think>`` to *suppress*
       reasoning when ``enable_thinking=False`` and omits it when enabled. If the on/off renders
       differ and involve ``<think>``, the model reasons when thinking is enabled. This catches
       models that emit ``<think>`` themselves rather than priming it in the prompt.

    Templates that merely mention ``<think>`` when re-encoding past assistant turns but render the
    *same* generation prompt regardless of ``enable_thinking`` (e.g. OpenLLM-France/Luciole-1B-
    Instruct-1.1) match neither signal and are correctly treated as non-thinking. The fundamental
    limit: a checkpoint that self-emits ``<think>`` while sharing such a template is indistinguishable
    from a non-thinking sibling by the template alone — set ``thinking_budget`` to force it on there.
    """
    if not use_chat_template:
        return False
    if getattr(tokenizer, "chat_template", None) is None:
        return False

    def _render(et: Optional[bool]) -> str:
        extra = {} if et is None else {"enable_thinking": et}
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}],
            tokenize=False,
            add_generation_prompt=True,
            **extra,
        )

    # Signal 1: the configured generation prompt primes an open <think>.
    try:
        rendered_cfg = _render(enable_thinking)
    except Exception:
        try:
            rendered_cfg = _render(None)
        except Exception as e:
            logger.warning(f"Could not render chat template to detect a thinking model: {e}")
            return False
    if prompt_primes_thinking(rendered_cfg):
        return True

    # The user explicitly disabled reasoning: honour it, do not probe further.
    if enable_thinking is False:
        return False

    # Signal 2: a thinking-mode template renders a different generation prompt when reasoning is
    # turned off (Qwen3 injects an empty <think></think> to suppress it). If on/off differ around
    # the <think> tag, the model reasons when thinking is enabled (the configured/default mode).
    try:
        rendered_on = _render(True)
        rendered_off = _render(False)
    except Exception:
        # Template does not support the enable_thinking toggle -> no signal-2 evidence.
        return False
    return rendered_on != rendered_off and (THINK_START_TAG in rendered_off or THINK_START_TAG in rendered_on)


def resolve_is_thinking_model(
    tokenizer,
    use_chat_template: bool,
    enable_thinking: Optional[bool],
    thinking_budget_is_set: bool,
) -> bool:
    """Whether to use two-phase thinking generation for this model.

    An explicitly-set ``thinking_budget`` forces it on: the user is asserting a reasoning model,
    and this is the reliable switch. Auto-detection (`detect_thinking_model`) is only the fallback
    when no budget is given, because it recognizes only models that *prime* an open ``<think>`` in
    the generation prompt (e.g. DeepSeek-R1) and misses reasoning models that emit ``<think>``
    themselves (e.g. some Qwen3/Luciole thinking checkpoints) — which would otherwise silently run
    single-phase and cap the whole generation at the answer budget.
    """
    if thinking_budget_is_set:
        return True
    return detect_thinking_model(tokenizer, use_chat_template, enable_thinking)


def two_phase_generate(
    *,
    inputs: list[list[int]],
    context: list[str],
    thinking_budget: int,
    answer_budget: Optional[int],
    num_samples: int,
    generate_fn: GenerateFn,
    close_tag_ids: list[int],
) -> list[ModelResponse]:
    """Two-phase generation for thinking models, shared by every generative backend.

    Phase 1 generates the reasoning, stopping at ``</think>`` (or at ``thinking_budget`` tokens).
    We always re-append a ``</think>`` afterwards, which both closes a reasoning that ran out of
    budget without stopping and restores the tag the stop condition strips, so the answer phase
    always starts from a well-formed ``...</think>``. Phase 2 generates the answer with
    ``answer_budget`` tokens. The returned text/tokens concatenate both phases, so downstream
    reasoning-tag stripping and the details logs see the full generation.

    ``generate_fn`` is the backend primitive (see `GenerateFn`); ``close_tag_ids`` are the token
    ids of ``</think>`` for this tokenizer.
    """
    # Phase 1: reasoning, stopped right at </think>.
    thinking_samples = generate_fn(inputs, thinking_budget, [THINK_END_TAG], num_samples)

    phase2_inputs: list[list[int]] = []
    thinking_texts: list[str] = []
    thinking_tokens: list[list[int]] = []
    samples_per_doc: list[int] = []
    for i, samples in enumerate(thinking_samples):
        samples_per_doc.append(len(samples))
        for sample in samples:
            # generate_fn returns text/tokens without the </think> stop: close it explicitly (this
            # also closes a reasoning that hit the budget without ever emitting </think>).
            text = sample.text + THINK_END_TAG
            tokens = list(sample.token_ids) + close_tag_ids
            thinking_texts.append(text)
            thinking_tokens.append(tokens)
            phase2_inputs.append(list(inputs[i]) + tokens)

    # Phase 2: answer, continuing after the (now closed) reasoning. One sample each, since the
    # reasoning that precedes it is already fixed.
    answer_samples = generate_fn(phase2_inputs, answer_budget, [], 1)

    responses: list[ModelResponse] = []
    flat = 0
    for i, samples in enumerate(thinking_samples):
        texts: list[str] = []
        token_ids: list[list[int]] = []
        for _ in range(samples_per_doc[i]):
            answer = answer_samples[flat][0]
            texts.append(thinking_texts[flat] + answer.text)
            token_ids.append(thinking_tokens[flat] + list(answer.token_ids))
            flat += 1
        responses.append(
            ModelResponse(
                input=context[i],
                text=texts,
                output_tokens=token_ids,
                input_tokens=list(inputs[i]),
            )
        )
    return responses

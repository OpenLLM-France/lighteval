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


# Default reasoning tags, used to split reasoning from the answer when the model does not declare
# its own as special tokens. DeepSeek-R1, Qwen3, ... emit these as plain strings in the template.
THINK_START_TAG = "<think>"
THINK_END_TAG = "</think>"

# Reasoning tag pairs that some model families ship as dedicated vocabulary tokens — whether or not
# they are flagged "special". Their presence in the vocab is a reliable "this is a reasoning model"
# signal, because such tokens only ship with reasoning checkpoints. Currently: Mistral/Magistral/
# Ministral reasoning models (``[THINK]``/``[/THINK]``, tekken control tokens from 2507+ releases).
# NB: these should NOT be flagged *special*, or ``skip_special_tokens`` (on by default in every
# backend) strips them from the decoded text and breaks the two-phase stop/split — see
# `reasoning_tags_are_special`.
DECLARED_REASONING_TAG_PAIRS: list[tuple[str, str]] = [("[THINK]", "[/THINK]")]

# Default reasoning budget (in tokens) for thinking models, used when the model args do not
# set generation_parameters.thinking_budget. Generous by design: reasoning traces are long.
DEFAULT_THINKING_BUDGET = 5000


def _tokenizer_has_tokens(tokenizer, *tokens: str) -> bool:
    """Whether the tokenizer knows every given string as a dedicated token (special OR not)."""
    present: set[str] = set()
    try:
        present |= set(tokenizer.get_vocab() or {})  # includes added tokens, special or not
    except Exception:
        pass
    present |= set(getattr(tokenizer, "added_tokens_encoder", {}) or {})
    present |= set(getattr(tokenizer, "all_special_tokens", []) or [])
    return all(tok in present for tok in tokens)


def declared_reasoning_tags(tokenizer) -> Optional[tuple[str, str]]:
    """The reasoning tag pair this model ships as dedicated tokens (e.g. Mistral ``[THINK]``), if any.

    Matches whether or not the tokens are flagged *special*: the model is a reasoning model either
    way, and the tags must stay non-special so they survive decoding (see `reasoning_tags_are_special`).
    """
    for start, end in DECLARED_REASONING_TAG_PAIRS:
        if _tokenizer_has_tokens(tokenizer, start, end):
            return (start, end)
    return None


def resolve_reasoning_tags(tokenizer) -> tuple[str, str]:
    """The ``(start, end)`` reasoning tags for this model: its declared dedicated pair if any, else
    the default ``<think>``/``</think>``. Used for the two-phase stop/split so the tags match what
    the model actually emits."""
    return declared_reasoning_tags(tokenizer) or (THINK_START_TAG, THINK_END_TAG)


def reasoning_stop_config(tokenizer, end_tag: str) -> tuple[list[int], list[int]]:
    """How to stop phase-1 reasoning at ``end_tag``, and which ids re-close it between phases.

    Returns ``(close_tag_ids, stop_thinking_token_ids)`` — the latter being only the reasoning end
    tag, never other stop tokens. Two tag families need different stop strategies:

    - **Single dedicated token** (e.g. Mistral ``[/THINK]``, one special token id): decoding with
      ``skip_special_tokens=True`` strips it from the text, so a *string* stop can never match. Stop
      on its **token id** instead, and re-close with that same id.
    - **Several ordinary tokens** (e.g. ``</think>``, which is not a special token): no single id to
      stop on, but the string survives decoding, so the **string** stop (``two_phase_generate`` passes
      ``end_tag``) works. Re-close with the tag's normal encoding; no token-id stop.
    """
    tid = None
    try:
        tid = (tokenizer.get_vocab() or {}).get(end_tag)
    except Exception:
        tid = None
    if tid is not None:
        return [tid], [tid]
    return tokenizer.encode(end_tag, add_special_tokens=False), []


def reasoning_tags_are_special(tokenizer, tags: tuple[str, str]) -> bool:
    """Whether both tags are flagged as *special* tokens.

    When they are, decoding with ``skip_special_tokens=True`` (the default in every generative
    backend) strips them from the output text, so the phase-1 stop (matched on the tag string) and
    the downstream reasoning-tag stripping both fail silently. Such a model needs its reasoning tags
    made non-special, or the output decoded with ``skip_special_tokens=False``.
    """
    special = set(getattr(tokenizer, "all_special_tokens", []) or [])
    return all(tag in special for tag in tags)


def ensure_reasoning_tags_decodable(
    tokenizer, tags: tuple[str, str], skip_special_tokens: bool, token_id_stop_supported: bool = False
) -> None:
    """Raise if two-phase thinking generation cannot delimit the reasoning from the answer.

    Called when the model is a thinking model. The phase-1 stop works if **either**:
    - the end tag survives decoding (not special, or ``skip_special_tokens=False``) -> string stop, or
    - the backend can stop on the end tag's **token id** and such an id exists (e.g. Mistral
      ``[/THINK]``) -> token-id stop, which is robust to the tag being stripped from the text.

    If neither holds (the tag is stripped from the decoded text and we cannot fall back to a token-id
    stop), two-phase would silently score the reasoning as the answer -> fail fast with guidance.
    """
    if not (skip_special_tokens and reasoning_tags_are_special(tokenizer, tags)):
        return  # the tag survives decoding -> string stop works
    _, stop_thinking_token_ids = reasoning_stop_config(tokenizer, tags[1])
    if token_id_stop_supported and stop_thinking_token_ids:
        return  # robust token-id stop will handle the stripped tag
    raise ValueError(
        f"Reasoning tags {tags} are registered as special tokens, so decoding with "
        "skip_special_tokens=True strips them from the generated text and two-phase thinking "
        "generation cannot work here (the reasoning is never delimited from the answer). Use the vLLM "
        "backend (which stops on the tag's token id), make these tags non-special in the tokenizer, or "
        "run with skip_special_tokens=False."
    )


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
# tokens, the stop strings, the reasoning end-tag token id(s) to also stop on (``stop_thinking_token_ids``
# — only the reasoning end tag, and empty unless it is a single dedicated token, e.g. Mistral
# ``[/THINK]``), and the number of samples per prompt, return one list of samples per prompt (outer
# list aligned with ``inputs``; inner list has ``num_samples`` entries).
GenerateFn = Callable[[list[list[int]], Optional[int], list[str], list[int], int], list[list[ThinkingGenSample]]]


def prompt_primes_thinking(rendered_prompt: str, tags: tuple[str, str] = (THINK_START_TAG, THINK_END_TAG)) -> bool:
    """Whether a rendered generation prompt primes an *open* reasoning block.

    A thinking model's generation prompt ends inside an unclosed start tag (e.g. DeepSeek-R1
    primes ``<think>\\n``), so the model must emit the end tag before its answer. This must NOT
    fire on templates that merely reference the start tag when formatting *previous* assistant
    turns but never open one for generation (e.g. OpenLLM-France/Luciole-1B-Instruct-1.1), nor on
    templates that emit an already-closed empty ``<think></think>`` block when reasoning is
    disabled (e.g. Qwen3 with enable_thinking=False). We therefore require the last start tag
    to have no matching end tag after it.
    """
    start_tag, end_tag = tags
    idx = rendered_prompt.rfind(start_tag)
    if idx == -1:
        return False
    return end_tag not in rendered_prompt[idx:]


def detect_thinking_model(tokenizer, use_chat_template: bool, enable_thinking: Optional[bool]) -> bool:
    """Decide from the tokenizer chat template whether the model reasons before answering.

    We look for any of three signals:

    0. **Ships reasoning tokens** (e.g. Mistral/Magistral/Ministral ``[THINK]``/``[/THINK]``):
       these dedicated vocabulary tokens (special or not) only ship with reasoning checkpoints, so
       their presence is itself the signal. This is the *only* reliable cue for such models, because
       they self-emit the tags — nothing is primed in the prompt and there is no toggle. It is
       vocab-based, so it does not need an HF ``chat_template`` and works for vLLM's Mistral/tekken
       tokenizer (which has no ``chat_template`` attribute) — hence checked before that guard.
    1. **Primes an open start tag** (e.g. DeepSeek-R1): the configured generation prompt ends inside
       an unclosed ``<think>`` (`prompt_primes_thinking`), so the model must reason first.
    2. **Toggle-responsive template** (e.g. Qwen3): turning thinking *off* changes the generation
       prompt around the ``<think>`` tag — Qwen3 injects an empty ``<think></think>`` to *suppress*
       reasoning when ``enable_thinking=False`` and omits it when enabled. If the on/off renders
       differ and involve ``<think>``, the model reasons when thinking is enabled. This catches
       models that emit ``<think>`` themselves rather than priming it in the prompt.

    Templates that merely mention ``<think>`` when re-encoding past assistant turns but render the
    *same* generation prompt regardless of ``enable_thinking`` (e.g. OpenLLM-France/Luciole-1B-
    Instruct-1.1) match no signal and are correctly treated as non-thinking. The fundamental limit:
    a checkpoint that self-emits ``<think>`` as a *plain string* (no special token) while sharing
    such a template is indistinguishable from a non-thinking sibling — set ``thinking_budget`` to
    force it on there.
    """
    if not use_chat_template:
        return False

    # Signal 0: the model ships dedicated reasoning tokens (e.g. Mistral [THINK]/[/THINK]), special
    # or not. Structural evidence that the model reasons, so (like signal 1) it fires regardless of
    # ``enable_thinking``: such models self-emit the tags and have no toggle to suppress. Vocab-based,
    # so it runs *before* the chat_template guard below — the Mistral/tekken tokenizer has no HF
    # ``chat_template`` attribute, and bailing out there would miss these models.
    if declared_reasoning_tags(tokenizer) is not None:
        return True

    # Signals 1-2 render the chat template, so a tokenizer without one cannot be probed further.
    if getattr(tokenizer, "chat_template", None) is None:
        return False

    tags = resolve_reasoning_tags(tokenizer)

    def _render(et: Optional[bool]) -> str:
        extra = {} if et is None else {"enable_thinking": et}
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}],
            tokenize=False,
            add_generation_prompt=True,
            **extra,
        )

    # Signal 1: the configured generation prompt primes an open start tag.
    try:
        rendered_cfg = _render(enable_thinking)
    except Exception:
        try:
            rendered_cfg = _render(None)
        except Exception as e:
            logger.warning(f"Could not render chat template to detect a thinking model: {e}")
            return False
    if prompt_primes_thinking(rendered_cfg, tags):
        return True

    # The user explicitly disabled reasoning: honour it for the toggle probe below.
    if enable_thinking is False:
        return False

    # Signal 2: a thinking-mode template renders a different generation prompt when reasoning is
    # turned off (Qwen3 injects an empty <think></think> to suppress it). If on/off differ around
    # the start tag, the model reasons when thinking is enabled (the configured/default mode).
    try:
        rendered_on = _render(True)
        rendered_off = _render(False)
    except Exception:
        # Template does not support the enable_thinking toggle -> no signal-2 evidence.
        return False
    return rendered_on != rendered_off and (tags[0] in rendered_off or tags[0] in rendered_on)


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
    end_tag: str = THINK_END_TAG,
    stop_thinking_token_ids: Optional[list[int]] = None,
) -> list[ModelResponse]:
    """Two-phase generation for thinking models, shared by every generative backend.

    Phase 1 generates the reasoning, stopping at the end tag (or at ``thinking_budget`` tokens).
    We always re-append the end tag afterwards, which both closes a reasoning that ran out of
    budget without stopping and restores the tag the stop condition strips, so the answer phase
    always starts from a well-formed ``...<end_tag>``. Phase 2 generates the answer with
    ``answer_budget`` tokens. The returned text/tokens concatenate both phases, so downstream
    reasoning-tag stripping and the details logs see the full generation.

    ``generate_fn`` is the backend primitive (see `GenerateFn`); ``end_tag`` is the reasoning end
    tag (``</think>`` or e.g. Mistral ``[/THINK]``) and ``close_tag_ids`` are its token ids for
    this tokenizer. ``stop_thinking_token_ids`` lets phase 1 stop on the end tag's token id instead
    of its string — needed when the tag is a special token that decoding strips from the text
    (Mistral); empty for string tags (``</think>``). See `reasoning_stop_config`.
    """
    stop_thinking_token_ids = list(stop_thinking_token_ids or [])
    # Phase 1: reasoning, stopped at the end tag (by string and/or by its token id).
    thinking_samples = generate_fn(inputs, thinking_budget, [end_tag], stop_thinking_token_ids, num_samples)

    phase2_inputs: list[list[int]] = []
    thinking_texts: list[str] = []
    thinking_tokens: list[list[int]] = []
    samples_per_doc: list[int] = []
    for i, samples in enumerate(thinking_samples):
        samples_per_doc.append(len(samples))
        for sample in samples:
            # generate_fn returns text/tokens without the end-tag stop: close it explicitly (this
            # also closes a reasoning that hit the budget without ever emitting the end tag).
            text = sample.text + end_tag
            tokens = list(sample.token_ids) + close_tag_ids
            thinking_texts.append(text)
            thinking_tokens.append(tokens)
            phase2_inputs.append(list(inputs[i]) + tokens)

    # Phase 2: answer, continuing after the (now closed) reasoning. One sample each, since the
    # reasoning that precedes it is already fixed.
    answer_samples = generate_fn(phase2_inputs, answer_budget, [], [], 1)

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

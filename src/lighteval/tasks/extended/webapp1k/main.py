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
"""WebApp1K: test-driven React web-app generation (text-only, pass@1).

From a feature's two Jest tests (a success case and a failure case), the model must write a
React component that makes **both** pass. Dataset: ``onekq-ai/WebApp1K-React`` (1,000 features,
20 domains). Scored by executing the generated component against the tests with a real
Jest/jsdom runner — see ``runner.py`` and ``README.md`` for the one-time Node setup.

Usage:
    # one-time: install the Node/Jest environment (see README.md)
    bash src/lighteval/tasks/extended/webapp1k/jsenv/setup.sh

    lighteval vllm "model_name=Qwen/Qwen2.5-Coder-7B-Instruct" "extended|webapp1k|0"

The dataset ships each test with its import header stripped, so the component variable name
(e.g. ``CMS``, ``App``, ``AddReview``) lives only in the original header. It is recovered from
the test body (the capitalized JSX tag that is not a known router/library component), which was
validated at 100% on a 30-file sample. The full test file is then reconstructed with a standard
header importing the component as ``./<Scenario>`` — exactly what the model is prompted with and
what is executed.
"""

import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from aenum import extend_enum

from lighteval.metrics.metrics import Metrics, SampleLevelMetric
from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.extended.webapp1k.runner import resolve_project_dir, run_jest
from lighteval.tasks.lighteval_task import Doc, LightevalTaskConfig
from lighteval.tasks.requests import SamplingMethod


# Standard header shared by every WebApp1K React test (the dataset strips it).
_TEST_HEADER = (
    "import React from 'react';\n"
    "import { render, screen, act, fireEvent } from '@testing-library/react';\n"
    "import { MemoryRouter } from 'react-router-dom';\n"
    "import fetchMock from 'fetch-mock';\n"
    "import '@testing-library/jest-dom';\n"
)
_AFTER_EACH = "afterEach(() => {\n  fetchMock.reset();\n  fetchMock.restore();\n});\n"

# Capitalized JSX tags that are library/router components, not the component under test.
_NOT_COMPONENT = {
    "React",
    "Fragment",
    "Suspense",
    "StrictMode",
    "MemoryRouter",
    "BrowserRouter",
    "HashRouter",
    "Router",
    "Routes",
    "Route",
    "Switch",
    "Redirect",
    "Link",
    "NavLink",
    "Provider",
    "Outlet",
}
_JSX_TAG_RE = re.compile(r"<([A-Z][A-Za-z0-9]*)\b")
# A fenced code block (``` optionally followed by a language tag).
_FENCE_RE = re.compile(r"```[a-zA-Z0-9]*\n(.*?)```", re.DOTALL)


def extract_component_var(test_body: str) -> str:
    """Recover the component variable name from the test body (see module docstring)."""
    candidates = [tag for tag in _JSX_TAG_RE.findall(test_body) if tag not in _NOT_COMPONENT]
    if not candidates:
        return "App"  # fallback: the component is default-exported, so any name works
    return Counter(candidates).most_common(1)[0][0]


def extract_code(text: str) -> str:
    """Pull the component source out of a model response (last fenced block, else raw text)."""
    blocks = _FENCE_RE.findall(text)
    return (blocks[-1] if blocks else text).strip()


def _build_test_file(component_var: str, scenario: str, success_case: str, failure_case: str) -> str:
    return (
        _TEST_HEADER
        + f"import {component_var} from './{scenario}';\n\n"
        + _AFTER_EACH
        + "\n"
        + success_case.strip()
        + "\n\n"
        + failure_case.strip()
        + "\n"
    )


def webapp1k_prompt_fn(line: dict[str, Any], task_name: str = "webapp1k") -> Doc:
    scenario = line["Scenario"].strip()
    success_case = line["Success Case"]
    failure_case = line["Failure Case"]
    component_var = extract_component_var(success_case + "\n" + failure_case)
    test_file = _build_test_file(component_var, scenario, success_case, failure_case)
    # Mirrors the upstream harness prompt (generate_code.py::generate_implementation).
    query = f"Generate {scenario}.js to pass the tests below:\n\n{test_file}. RETURN CODE ONLY."
    return Doc(
        task_name=task_name,
        query=query,
        choices=[""],
        gold_index=0,
        specific={"scenario": scenario, "test_file": test_file, "category": line.get("Category")},
    )


class WebApp1KMetric(SampleLevelComputation):
    """pass@1: fraction of generations whose component makes both tests pass.

    Each generation's code is written into a prepared Node/Jest project alongside the
    reconstructed test file and executed; a generation counts as correct only when Jest reports
    both the success and the failure test passing. With a single generation this is 0.0/1.0; with
    several it is the standard unbiased pass@1 estimate (correct / total).
    """

    def compute(self, model_response: ModelResponse, doc: Doc, **kwargs) -> float:
        assert doc.specific is not None, "WebApp1K requires doc.specific (scenario, test_file)"
        project_dir = resolve_project_dir()
        scenario = doc.specific["scenario"]
        test_file = doc.specific["test_file"]
        timeout = int(os.environ.get("WEBAPP1K_TEST_TIMEOUT", "120"))

        results = [
            run_jest(project_dir, scenario, extract_code(pred), test_file, timeout=timeout)["passed"]
            for pred in model_response.final_text
        ]
        return (sum(results) / len(results)) if results else 0.0


webapp1k_pass_at_1 = SampleLevelMetric(
    metric_name="pass@1",
    category=SamplingMethod.GENERATIVE,
    higher_is_better=True,
    sample_level_fn=WebApp1KMetric(),
    corpus_level_fn=np.mean,
    batched_compute=False,
)

extend_enum(Metrics, "webapp1k_pass_at_1", webapp1k_pass_at_1)


webapp1k_task = LightevalTaskConfig(
    name="webapp1k",
    suite=["extended"],
    prompt_function=webapp1k_prompt_fn,
    hf_repo="onekq-ai/WebApp1K-React",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=2048,
    metrics=[Metrics.webapp1k_pass_at_1],
    stop_sequence=[],
    version=0,
)

# Expose the jsenv directory for the README / tooling.
JSENV_DIR = str(Path(__file__).parent / "jsenv")

TASKS_TABLE = [webapp1k_task]

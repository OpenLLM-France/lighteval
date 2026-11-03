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
"""Score an existing WebApp1K details parquet with Jest, decoupled from generation.

Generation needs a GPU; Jest scoring needs Node. On an HPC cluster whose GPU container has no
Node (e.g. Adastra), run the model once to produce the lighteval details parquet, then run THIS
on a machine that has Node (and the prepared ``jsenv``) to recompute pass@1 from the saved
generations — no GPU, no re-running the model.

    python -m lighteval.tasks.extended.webapp1k.score_details DETAILS.parquet [...] \\
        --num-workers 8 --output scores.csv

Each row's component is re-extracted from the saved generation and the test file is parsed back
out of the prompt (``doc.query``), so the parquet is self-sufficient. Requires ``node``/``npx`` on
PATH and a prepared ``jsenv`` (default: the bundled one, or ``--project-dir`` / the
``WEBAPP1K_JEST_PROJECT`` env var). See the WebApp1K README.
"""

import argparse
import os
import re
from concurrent.futures import ThreadPoolExecutor

import pandas as pd


# Import the Jest runner. Prefer the package path (when lighteval is installed); fall back to a
# sibling ``runner.py`` so this script also runs standalone on any machine with Node + pandas
# (e.g. copied next to runner.py), without needing lighteval/torch installed.
try:
    from lighteval.tasks.extended.webapp1k.runner import resolve_project_dir, run_jest
except ImportError:
    from runner import resolve_project_dir, run_jest


# A fenced code block (``` optionally with a language tag); kept local so this script does not pull
# in lighteval.tasks.extended.webapp1k.main (which imports torch via the metrics stack).
_FENCE_RE = re.compile(r"```[a-zA-Z0-9]*\n(.*?)```", re.DOTALL)
# Recovers the scenario name and the full test file from the prompt built by webapp1k_prompt_fn.
_QUERY_RE = re.compile(
    r"^Generate (?P<scenario>.+?)\.js to pass the tests below:\n\n(?P<test>.*)\. RETURN CODE ONLY\.\s*$", re.DOTALL
)


def extract_code(text: str) -> str:
    """Pull the component source out of a model response (last fenced block, else raw text)."""
    blocks = _FENCE_RE.findall(text)
    return (blocks[-1] if blocks else text).strip()


def _generations(model_response: dict) -> list[str]:
    """Pull the model's text generation(s), preferring the reasoning-stripped version."""
    for key in ("text_post_processed", "text"):
        value = model_response.get(key)
        if value is None:
            continue
        texts = [
            t for t in (list(value) if hasattr(value, "__len__") and not isinstance(value, str) else [value]) if t
        ]
        if texts:
            return [str(t) for t in texts]
    return []


def _parse_query(query: str):
    """Return (scenario, test_file) parsed back out of a webapp1k prompt, or (None, None)."""
    m = _QUERY_RE.match(query)
    return (m.group("scenario"), m.group("test")) if m else (None, None)


def score_row(project_dir: str, row: dict, timeout: int) -> dict:
    scenario, test_file = _parse_query(row["doc"]["query"])
    if scenario is None:
        return {"scenario": None, "pass": 0.0, "n": 0, "error": "could not parse prompt"}
    gens = _generations(row["model_response"])
    if not gens:
        return {"scenario": scenario, "pass": 0.0, "n": 0, "error": "no generation"}
    outcomes = [run_jest(project_dir, scenario, extract_code(g), test_file, timeout=timeout)["passed"] for g in gens]
    return {"scenario": scenario, "pass": sum(outcomes) / len(outcomes), "n": len(outcomes), "error": ""}


def main() -> None:
    parser = argparse.ArgumentParser(description="Score WebApp1K details parquet(s) with Jest.")
    parser.add_argument("parquets", nargs="+", help="lighteval details parquet(s) for extended|webapp1k")
    parser.add_argument(
        "--project-dir", default=None, help="Prepared jsenv dir (else bundled / WEBAPP1K_JEST_PROJECT)"
    )
    parser.add_argument("--num-workers", type=int, default=4, help="Concurrent Jest processes (default 4)")
    parser.add_argument("--timeout", type=int, default=120, help="Per-sample Jest timeout in seconds")
    parser.add_argument("--max-samples", type=int, default=None, help="Score only the first N rows (smoke test)")
    parser.add_argument("--output", default=None, help="Optional CSV path for per-sample results")
    args = parser.parse_args()

    if args.project_dir:
        os.environ["WEBAPP1K_JEST_PROJECT"] = args.project_dir
    project_dir = resolve_project_dir()  # raises a clear error if Node/jsenv is not usable

    all_rows = []
    for path in args.parquets:
        df = pd.read_parquet(path)
        if args.max_samples is not None:
            df = df.head(args.max_samples)
        rows = df.to_dict("records")
        with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
            results = list(pool.map(lambda r: score_row(project_dir, r, args.timeout), rows))
        passed = sum(1 for r in results if r["pass"] >= 1.0)
        scored = sum(1 for r in results if r["n"] > 0)
        mean = (sum(r["pass"] for r in results) / len(results)) if results else 0.0
        print(f"{path}")
        print(f"  pass@1 = {mean:.4f}  ({passed}/{len(results)} solved; {scored} actually executed)")
        errs = {}
        for r in results:
            if r["error"]:
                errs[r["error"]] = errs.get(r["error"], 0) + 1
        if errs:
            print(f"  errors: {errs}")
        for r in results:
            r["parquet"] = os.path.basename(path)
        all_rows.extend(results)

    if args.output:
        pd.DataFrame(all_rows).to_csv(args.output, index=False)
        print(f"Wrote per-sample results to {args.output}")


if __name__ == "__main__":
    main()

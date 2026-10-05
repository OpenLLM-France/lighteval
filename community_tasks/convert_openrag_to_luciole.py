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

"""Convert the openrag_bench dataset to the luciole_rag row schema.

Reads ``openrag_bench`` (the ``reranked_gold.parquet`` export) and emits a
HuggingFace-style dataset directory whose rows match what
``community_tasks/luciole_rag.py`` expects:

- ``id``: stable example identifier (kept as-is)
- ``query``: user question (``question``)
- ``retrieved_documents``: list[str], the retrieved context chunks (``contexts``
  ``text``, in order)
- ``titles``: list[str], aligned with ``retrieved_documents`` (``filename``,
  disambiguated when a file appears several times in a row)
- ``supporting_index``: list[int], zero-based indices of the evidence chunks
  (roles ``gold``, ``gold_alternative``, ``gold_equivalent``)
- ``answer``: expected answer (``answer`` when ``answerable``, empty otherwise)

Usage::

    python community_tasks/convert_openrag_to_luciole.py \
        [--input reranked_gold.parquet] [--output openrag_bench_luciole]

The output directory is a directory of parquet shards named ``test-*.parquet``,
so ``datasets.load_dataset(path=<output>, name="default")`` returns a
``DatasetDict`` with a single ``test`` split, which is what lighteval loads.
"""

import argparse
import os
from collections import Counter

import pandas as pd
from datasets import Dataset


SUPPORTING_ROLES = {"gold", "gold_alternative", "gold_equivalent"}


def disambiguated_titles(filenames: list[str]) -> list[str]:
    """Return unique per-row titles from a list of filenames.

    When a filename repeats within the row, later occurrences get a `` [n]``
    suffix (2, 3, ...) so every chunk keeps a distinct, citable title.
    """
    titles: list[str] = []
    seen: Counter[str] = Counter()
    for filename in filenames:
        key = filename or ""
        count = seen[key]
        seen[key] += 1
        title = key if count == 0 else f"{key} [{count + 1}]"
        titles.append(title)
    return titles


def convert_row(row: dict) -> dict:
    contexts = list(row["contexts"]) if row["contexts"] is not None else []
    retrieved = [str(c["text"]) for c in contexts]
    titles = disambiguated_titles([str(c.get("filename") or "") for c in contexts])
    supporting = [i for i, c in enumerate(contexts) if c.get("role") in SUPPORTING_ROLES]

    return {
        "id": row["id"],
        "query": row["question"],
        "retrieved_documents": retrieved,
        "titles": titles,
        "supporting_index": supporting,
        "answer": (row["answer"] if row["answerable"] else ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=os.path.join(os.path.dirname(__file__), "reranked_gold.parquet"),
        help="Path to the openrag_bench parquet export.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "openrag_bench_luciole"),
        help="Output directory for the converted dataset.",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    rows = [convert_row(r) for r in df.to_dict(orient="records")]

    dataset = Dataset.from_list(rows)
    dataset.to_parquet(os.path.join(args.output, "test-00000-of-00001.parquet"))
    print(f"Converted {len(rows)} rows -> {args.output}/test-00000-of-00001.parquet")


if __name__ == "__main__":
    main()

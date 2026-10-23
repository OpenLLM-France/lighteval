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

"""Convert OpenRAG Bench to the luciole_rag row schema.

Reads a split of ``OpenLLM-France/OpenRAG-Bench`` (private: needs a Hugging
Face token with access to the organization) and writes a parquet file whose
rows match what ``community_tasks/luciole_rag.py`` expects:

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
        [--revision <commit>] [--split reranked_gold] [--output test.parquet]

``luciole_rag:openrag_bench`` reads the output of the ``reranked_gold`` split,
published as ``luciole/test.parquet`` under the ``luciole`` config of
OpenRAG-Bench itself, so it keeps the benchmark's licenses and visibility.
``reranked_gold`` guarantees every evidence chunk a slot among the 10, so it
measures the model alone; ``reranked`` also measures retrieval.
"""

import argparse
from collections import Counter

import pandas as pd
from datasets import Dataset
from huggingface_hub import hf_hub_download


SOURCE_REPO = "OpenLLM-France/OpenRAG-Bench"
# Every evidence chunk, not only the minimal ``answer_positions``: luciole_rag
# drops the supporting chunks to build synthetic unanswerables, and an evidence
# chunk left behind would keep the question answerable.
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
    parser.add_argument("--revision", default=None, help=f"Commit of {SOURCE_REPO} to convert (default: main).")
    parser.add_argument("--split", default="reranked_gold", choices=["reranked_gold", "reranked"])
    parser.add_argument("--output", default="test.parquet", help="Output parquet file.")
    args = parser.parse_args()

    path = hf_hub_download(SOURCE_REPO, f"data/{args.split}.parquet", repo_type="dataset", revision=args.revision)
    df = pd.read_parquet(path)
    rows = [convert_row(r) for r in df.to_dict(orient="records")]

    Dataset.from_list(rows).to_parquet(args.output)
    print(f"Converted {len(rows)} rows of {SOURCE_REPO}:{args.split} -> {args.output}")


if __name__ == "__main__":
    main()

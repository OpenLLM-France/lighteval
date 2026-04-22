"""
name:
Ruler

abstract:
Prompt helper for RULER long-context evaluations. Paired with the
`ruler_match_any` / `ruler_match_all` metrics defined in
`lighteval.metrics.metrics.Metrics`, this function can be plugged into
custom task configs that point at a RULER-style dataset
(fields `input`, `outputs`, optional `answer_prefix`).

No TASKS_TABLE is exported; users wire `ruler` into their own
`LightevalTaskConfig` via `--custom-tasks`.

tags:
long-context
"""

from lighteval.tasks.requests import Doc


def ruler(line, task_name: str = None):
    query = line["input"]
    choices = line["outputs"]
    answer_prefix = line.get("answer_prefix", "")
    gold_index = list(range(len(choices)))
    query = f"{query} {answer_prefix}"

    return Doc(query=query, instruction=None, choices=choices, gold_index=gold_index, task_name=task_name)

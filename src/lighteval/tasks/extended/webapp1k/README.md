# WebApp1K (lighteval extended task)

Test-driven React web-app generation, scored by **pass@1** with a real Jest runner.
Port of [WebApp1K](https://github.com/onekq/WebApp1k)([arXiv:2408.00019](https://arxiv.org/abs/2408.00019)) over the
[`onekq-ai/WebApp1K-React`](https://huggingface.co/datasets/onekq-ai/WebApp1K-React) dataset
(1,000 features across 20 domains).

For each feature the model is given two Jest tests (a **success** case and a **failure** case)
and must write a single default-exported React component that makes **both** pass. A sample counts
as solved only when both tests pass (this dual design is WebApp1K's guard against reward hacking).

## How it works

- `main.py` — prompt function (reconstructs the full test file and asks for the component) and the
  `pass@1` metric.
- `runner.py` — writes the generated component + the test into a throwaway dir inside the Node
  project and runs Jest, parsing the JSON result.
- `jsenv/` — the Node/Jest project (dependencies + config) this task executes against.

The dataset stores each test with its `import` header stripped, so the component's variable name
(e.g. `CMS`, `App`, `AddReview`) and import path are not in the dataset. The variable name is
recovered from the test body (the capitalized JSX tag that isn't a known router/library
component; validated at 100% on a 30-file sample) and the full test file is rebuilt with a
standard header importing the component as `./<Scenario>`.

## One-time setup (required)

The task executes JavaScript, so it needs a prepared Node/Jest environment. This is **not**
installed by `pip`; set it up once:

```bash
# 1. Install Node.js >= 18 (skip if already available). Options:
#    - distro package manager, e.g.  sudo apt-get install nodejs npm
#    - nvm (no root):   curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
#                       nvm install 20
#    - conda (no root): conda install -c conda-forge nodejs=20
#    - HPC / offline: load a Node module (`module load nodejs`) or download a prebuilt tarball
#      from https://nodejs.org/dist/ on a login node and add its bin/ to PATH.

# 2. Install the JS dependencies (needs internet; run on a login node on HPC):
bash src/lighteval/tasks/extended/webapp1k/jsenv/setup.sh
```

On an **offline** compute node: run `setup.sh` once on a login node with
internet, then the populated `jsenv/node_modules` is reused offline. If compute nodes don't share
the filesystem, copy the whole `jsenv/` directory over, or point `WEBAPP1K_JEST_PROJECT` at a
prepared copy.

## Running

```bash
lighteval vllm "model_name=Qwen/Qwen2.5-Coder-7B-Instruct" "extended|webapp1k|0"
```

Environment variables:

- `WEBAPP1K_NODE_BIN` — directory containing the `node`/`npx` binaries, prepended to PATH when
  the runner locates and launches Node. Use this when Node lives on a shared filesystem outside
  the container image (typical on HPC). It is the **reliable** way to expose Node, because a login
  shell inside the container can reset `PATH` and drop an entry added via
  `APPTAINERENV_PREPEND_PATH` — which is exactly how a run can silently score every sample 0. The
  runner now also **fails loudly** (clear error) if Node is missing, unrunnable, or older than
  Node 18, instead of returning 0.
- `WEBAPP1K_JEST_PROJECT` — path to the prepared Node project (default: this `jsenv/` dir).
- `WEBAPP1K_TEST_TIMEOUT` — per-sample Jest timeout in seconds (default: 120).

### Scoring an existing run without a GPU

If you generated on a GPU box that has no Node (e.g. an HPC GPU container), you don't need to
re-run the model: score the saved details parquet on any machine with Node via
`score_details.py`, which re-extracts each generation and the test from the parquet and reruns
Jest:

```bash
WEBAPP1K_NODE_BIN=/path/to/node/bin \
python -m lighteval.tasks.extended.webapp1k.score_details \
    path/to/details_extended\|webapp1k\|0_*.parquet --num-workers 8 --output scores.csv
```

## Caveats

- **Security**: this executes model-generated JavaScript. It is *not* a sandbox. Run it in an
  isolated/containerized environment, as for any code-execution benchmark.
- **Cost**: ~1,000 features, each a separate Jest process (a few seconds each), so a full run is
  slow. Use `--max-samples` while iterating.
- **Dependency pinning**: `jsenv/package.json` pins the dependency majors that match the tests'
  APIs. In particular **`fetch-mock` is pinned to v9**. The tests use its v9 API
  (`fetchMock.post(url, { status })`, `fetchMock.calls(url)`), which v10+ removed. It also
  requires **`node-fetch` v2** (a runtime dep of `fetch-mock@9` under Node/CJS; without it every
  test suite fails to load with `Cannot find module 'node-fetch'` and the whole benchmark reads 0;
  v3 is ESM-only and breaks the CJS require). Upstream uses Create React App with `"latest"` deps;
  this port uses a direct Jest + Babel + jsdom setup for reproducibility.

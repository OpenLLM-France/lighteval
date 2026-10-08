<p align="center">
  <br/>
    <img alt="lighteval library logo" src="./assets/lighteval-doc.svg" width="376" height="59" style="max-width: 100%;">
  <br/>
</p>


<p align="center">
    <i>Your go-to toolkit for lightning-fast, flexible LLM evaluation, from Hugging Face's Leaderboard and Evals Team.</i>
</p>

<div align="center">

[![Tests](https://github.com/huggingface/lighteval/actions/workflows/tests.yaml/badge.svg?branch=main)](https://github.com/huggingface/lighteval/actions/workflows/tests.yaml?query=branch%3Amain)
[![Quality](https://github.com/huggingface/lighteval/actions/workflows/quality.yaml/badge.svg?branch=main)](https://github.com/huggingface/lighteval/actions/workflows/quality.yaml?query=branch%3Amain)
[![Python versions](https://img.shields.io/pypi/pyversions/lighteval)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/huggingface/lighteval/blob/main/LICENSE)
[![Version](https://img.shields.io/pypi/v/lighteval)](https://pypi.org/project/lighteval/)

</div>

---

<p align="center">
  <a href="https://huggingface.co/docs/lighteval/main/en/index" target="_blank">
    <img alt="Documentation" src="https://img.shields.io/badge/Documentation-4F4F4F?style=for-the-badge&logo=readthedocs&logoColor=white" />
  </a>
</p>

---

**Lighteval** is your *all-in-one toolkit* for evaluating LLMs across multiple
backends—whether your model is being **served somewhere** or **already loaded in memory**.
Dive deep into your model's performance by saving and exploring *detailed,
sample-by-sample results* to debug and see how your models stack-up.

*Customization at your fingertips*: letting you either browse all our existing tasks and [metrics](https://huggingface.co/docs/lighteval/metric-list) or effortlessly create your own [custom task](https://huggingface.co/docs/lighteval/adding-a-custom-task) and [custom metric](https://huggingface.co/docs/lighteval/adding-a-new-metric), tailored to your needs.

> **🇫🇷 OpenLLM-France fork.** This is the OpenLLM-France fork of lighteval, maintained by the OpenLLM-France consortium, on top of upstream [huggingface/lighteval](https://github.com/huggingface/lighteval). It adds a suite of French and multilingual benchmarks, safety and red-teaming evaluations, translation metrics (COMET, MetricX), long-context (RULER) and reasoning-model (thinking-budget) support, an offline-capable LLM-as-judge, and many bug fixes and backend-robustness improvements. See [OpenLLM-France fork additions](#openllm-france-fork-additions) for the full list.


## Available Tasks

Lighteval supports **7,000+ evaluation tasks** across multiple domains and languages. Here's an overview of some *popular benchmarks*:


### 📚 **Knowledge**
- **General Knowledge**: MMLU, MMLU-Pro, MMMU, BIG-Bench
- **Question Answering**: TriviaQA, Natural Questions, SimpleQA, Humanity's Last Exam (HLE)
- **Specialized**: GPQA, AGIEval

### 🧮 **Math and Code**
- **Math Problems**: GSM8K, GSM-Plus, MATH, MATH500
- **Competition Math**: AIME24, AIME25
- **Multilingual Math**: MGSM (Grade School Math in 10+ languages)
- **Coding Benchmarks**: LCB (LiveCodeBench)

### 🎯 **Chat Model Evaluation**
- **Instruction Following**: IFEval, IFEval-fr
- **Reasoning**: MUSR, DROP (discrete reasoning)
- **Long Context**: RULER
- **Dialogue**: MT-Bench
- **Holistic Evaluation**: HELM, BIG-Bench

### 🌍 **Multilingual Evaluation**
- **Cross-lingual**: XTREME, Flores200 (200 languages), XCOPA, XQuAD
- **Language-specific**: 
  - **Arabic**: ArabicMMLU
  - **Filipino**: FilBench
  - **French**: IFEval-fr, GPQA-fr, BAC-fr
  - **German**: German RAG Eval
  - **Serbian**: Serbian LLM Benchmark, OZ Eval
  - **Turkic**: TUMLU (9 Turkic languages)
  - **Chinese**: CMMLU, CEval, AGIEval
  - **Russian**: RUMMLU, Russian SQuAD
  - **And many more...**

### 🧠 **Core Language Understanding**
- **NLU**: GLUE, SuperGLUE, TriviaQA, Natural Questions
- **Commonsense**: HellaSwag, WinoGrande, ProtoQA
- **Natural Language Inference**: XNLI
- **Reading Comprehension**: SQuAD, XQuAD, MLQA, Belebele


## ⚡️ Installation

> **Note**: lighteval is currently *completely untested on Windows*, and we don't support it yet. (*Should be fully functional on Mac/Linux*)

```bash
pip install lighteval
```

Lighteval allows for *many extras* when installing, see [here](https://huggingface.co/docs/lighteval/installation) for a **complete list**.

If you want to push results to the **Hugging Face Hub**, add your access token as
an environment variable:

```shell
huggingface-cli login
```

## 🚀 Quickstart

Lighteval offers the following entry points for model evaluation:

- `lighteval accelerate`: Evaluate models on CPU or one or more GPUs using [🤗
  Accelerate](https://github.com/huggingface/accelerate)
- `lighteval nanotron`: Evaluate models in distributed settings using [⚡️
  Nanotron](https://github.com/huggingface/nanotron)
- `lighteval vllm`: Evaluate models on one or more GPUs using [🚀
  VLLM](https://github.com/vllm-project/vllm)
- `lighteval sglang`: Evaluate models using [SGLang](https://github.com/sgl-project/sglang) as backend
- `lighteval endpoint`: Evaluate models using various endpoints as backend
  - `lighteval endpoint inference-endpoint`: Evaluate models using Hugging Face's [Inference Endpoints API](https://huggingface.co/inference-endpoints/dedicated)
  - `lighteval endpoint tgi`: Evaluate models using [🔗 Text Generation Inference](https://huggingface.co/docs/text-generation-inference/en/index) running locally
  - `lighteval endpoint litellm`: Evaluate models on any compatible API using [LiteLLM](https://www.litellm.ai/)
  - `lighteval endpoint inference-providers`: Evaluate models using [HuggingFace's inference providers](https://huggingface.co/docs/inference-providers/en/index) as backend

Did not find what you need ? You can always make your custom model API by following [this guide](https://huggingface.co/docs/lighteval/main/en/evaluating-a-custom-model)
- `lighteval custom`: Evaluate custom models (can be anything)

Here's a **quick command** to evaluate using the *Accelerate backend*:

```shell
lighteval accelerate \
    "model_name=gpt2" \
    "leaderboard|truthfulqa:mc|0"
```

Or use the **Python API** to run a model *already loaded in memory*!

```python
from transformers import AutoModelForCausalLM

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.transformers.transformers_model import TransformersModel, TransformersModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters


MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
BENCHMARKS = "lighteval|gsm8k|0"

evaluation_tracker = EvaluationTracker(output_dir="./results")
pipeline_params = PipelineParameters(
    launcher_type=ParallelismManager.NONE,
    max_samples=2
)

model = AutoModelForCausalLM.from_pretrained(
  MODEL_NAME, device_map="auto"
)
config = TransformersModelConfig(model_name=MODEL_NAME, batch_size=1)
model = TransformersModel.from_model(model, config)

pipeline = Pipeline(
    model=model,
    pipeline_parameters=pipeline_params,
    evaluation_tracker=evaluation_tracker,
    tasks=BENCHMARKS,
)

results = pipeline.evaluate()
pipeline.show_results()
results = pipeline.get_results()
```

## OpenLLM-France fork additions

This fork is maintained by the [OpenLLM-France](https://github.com/OpenLLM-France) consortium on top of upstream [huggingface/lighteval](https://github.com/huggingface/lighteval). Below is a summary of what it adds, grouped by category.

### Added benchmarks and tasks

**French & multilingual**
- **BBH** (BIG-Bench-Hard) and **BBH-fr** (French), with chain-of-thought
- **EIFFEL** — French idiomatic-expression MCQ
- **CARTE** — French regional/cultural knowledge (with abstention-aware metrics)
- **INCLUDE** — multilingual regional-knowledge exams
- **Multilingual PIQA** — physical commonsense
- **Multilingual AIME 2025** (French, German, …)
- **BLEnD** — multilingual everyday cultural commonsense
- **MGSM-rev2** — corrected MGSM, with proper chain-of-thought few-shot
- **MathAlea** — French math MCQ (+ a generative variant)
- **Exo7** — French math multi-label evaluation
- **MMLU-Pro** generative variant (for instruct/thinking models), with CoT few-shot
- **GPQA-fr** turned into a generative benchmark
- **luciole_rag** — citation-aware grounded QA / RAG benchmark
- **FLORES-200 instruction variant** (`flores200_instruct`) — chat/instruction-style translation with answer extraction
- Additional settings for Arabic benchmarks

**Safety & red-teaming**
- Safety benchmarks in French and other languages
- **WildJailBreak**, **AyaRedTeaming**, **HarmBench**, **Hex-PHI**, and an AdvBench-based red-teaming benchmark
- **FalseQA**, **PCBench**, **SCoolKID** (false-premise detection)
- LLM-as-judge refusal metrics

**Long context**
- **RULER** (metric + prompts)

### Reasoning ("thinking") model support
- Two-phase generation with a separate **thinking budget** (default 5k) so the reasoning trace does not eat into the answer budget
- `enable_thinking` option
- Automatic detection of thinking models and their reasoning tags — `<think>…</think>` and Mistral `[THINK]…[/THINK]`
- Robust stripping of reasoning traces (including models that prime the opening tag in the generation prompt)

### LLM-as-judge improvements
- **Run the judge fully offline as a local (vLLM) model** — any Hugging Face model can be the judge, removing the hard dependency on an OpenAI/API judge. This is what makes it possible to use local safety judges such as Llama Guard 4, wildguard or Qwen in the safety benchmarks.
- vLLM-judge enabling fixes: render chat messages to strings instead of token ids; free the evaluated model's GPU memory before loading the judge; environment variables to tune the judge's memory usage and to force eager mode.
- More robust and reproducible judging: deterministic results; don't drop samples when the judge response can't be parsed; keep non-numeric (textual) judge outputs in the details; optional variant where the judge does not see the question.
- LLM-as-judge **refusal metrics** (for safety / red-teaming).

### Translation metrics
- **COMET** and **MetricX** metrics, wired into the FLORES benchmarks (device/batch options, GPU execution, offline-safe)
- FLORES made to work with the parquet version of the dataset

### Backend, performance & robustness
- Auto-detect the **Mistral/tekken tokenizer** (no more `LIGHTEVAL_TOKENIZER_MODE`)
- Context-parallelism support (vLLM ≥ 0.15); fixes for mixing data/pipeline parallelism
- Compatibility with recent vLLM versions (engine-arg filtering with warnings, logprob/prefix-caching fixes)
- Memory controls (free the model before the judge, litellm logprob RAM fix, KV-cache length limits)
- Caching fixes: `LIGHTEVAL_DISABLE_CACHE`, address-independent cache keys, unique `doc.id` across splits
- Honour `HF_HOME`; offline-safe dataset/tokenizer loading; env vars for eager mode and VLM/Mistral loading
- Option to set up a run **without actually running the evaluation** (dry run, e.g. to validate task/model loading)
- **Robust result saving**: write the results JSON *before* building the details, so results survive even when details serialization fails (e.g. the Arrow 2 GB overflow on `live_code_bench`); such a failure is still surfaced as a non-zero exit

### Bug fixes
- **Multiple-choice (MCQ) scoring on the Accelerate/Transformers backend**: fixed an incorrect 2-D slicing of the gathered continuation logits (and stray `-1` padding left in the continuations) that corrupted the log-likelihood comparison across choices.
- Log-probability computation with recent vLLM (≥ 0.12), broken by prefix caching.
- Many other corner-case fixes: IFBench / IFEval-fr, MGSM / MMLU-Pro few-shot CoT, GPQA-fr dataset, stop-sequences, `squad_v2` unanswerable rows, nltk / transformers version robustness, offline judge, and more

### Packaging
- Community-task JSON stored as regular files (not Git LFS)
- Fork-specific install / doc-build adjustments; built-in French/English system prompts; per-doc system-role support via `Doc.specific`

## 🙏 Acknowledgements

Lighteval took inspiration from the following *amazing* frameworks: Eleuther's [AI Harness](https://github.com/EleutherAI/lm-evaluation-harness) and Stanford's
[HELM](https://crfm.stanford.edu/helm/latest/). We are grateful to their teams for their **pioneering work** on LLM evaluations.

We'd also like to offer our thanks to all the community members who have contributed to the library, adding new features and reporting or fixing bugs.

## 🌟 Contributions Welcome 💙💚💛💜🧡

**Got ideas?** Found a bug? Want to add a
[task](https://huggingface.co/docs/lighteval/adding-a-custom-task) or
[metric](https://huggingface.co/docs/lighteval/adding-a-new-metric)?
Contributions are *warmly welcomed*!

If you're adding a **new feature**, please *open an issue first*.

If you open a PR, don't forget to **run the styling**!

```bash
pip install -e .[dev]
pre-commit install
pre-commit run --all-files
```
## 📜 Citation

```bibtex
@misc{lighteval,
  author = {Habib, Nathan and Fourrier, Clémentine and Kydlíček, Hynek and Wolf, Thomas and Tunstall, Lewis},
  title = {LightEval: A lightweight framework for LLM evaluation},
  year = {2023},
  version = {0.11.0},
  url = {https://github.com/huggingface/lighteval}
}
```

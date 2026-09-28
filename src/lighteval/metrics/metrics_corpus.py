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

"""This module manages all the metrics occurring at the corpus level.
Some metrics (such as corpus BLEU) are not computed at the individual item level, but over all the corpus.
A number of these aggregations come from the EleutherAIHarness
"""

import logging
import math
from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import sacrebleu
import sklearn.metrics

from dataclasses import dataclass

from lighteval.metrics.sample_preparator import (
    CorpusMetricInput,
    GenerativeCorpusMetricInput,
    LogprobCorpusMetricInput,
    PerplexityCorpusMetricInput,
    Preparator,
)
from lighteval.utils.utils import as_list


logger = logging.getLogger(__name__)


class CorpusLevelComputation(ABC):
    @abstractmethod
    def compute_corpus(self):
        raise NotImplementedError

    def __str__(self):
        attrs = vars(self)
        attr_strs = []
        for k, v in attrs.items():
            if callable(v):
                val_str = v.__name__
            else:
                val_str = str(v)
            attr_strs.append(f"{k}={val_str}")
        return f"{self.__class__.__name__}({', '.join(attr_strs)})"


# General aggregations
class MatthewsCorrCoef(CorpusLevelComputation):
    def compute_corpus(self, items: list[GenerativeCorpusMetricInput]) -> float:
        """Computes the Matthews Correlation Coefficient, using scikit learn ([doc](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html)).

        Args:
            items (list[dict]): List of GenerativeCorpusMetricInput

        Returns:
            float: Score
        """
        golds = [i.golds for i in items]
        preds = [i.preds for i in items]
        return sklearn.metrics.matthews_corrcoef(golds, preds)


class CorpusLevelF1Score(CorpusLevelComputation):
    def __init__(self, average: str, num_classes: int = 2):
        """Stores the relevant parameters for the task's corpus level f1 score.

        Args:
            average (str): Method to use to compute the f1 score. Can be weighted, macro, micro.
            num_classes (int, optional): Num of possible choice classes. Defaults to 2. If this parameter is above 2, we'll compute multi f1 corpus score
        """
        if average not in ["weighted", "macro", "micro", None]:
            raise ValueError(
                f"A CorpusLevelF1Score must be initialized with weighted, macro, micro, or None as an average function. {average} was used."
            )
        self.average = average
        self.num_classes = num_classes

    def compute_corpus(self, items: list[LogprobCorpusMetricInput]):
        """Computes the metric score over all the corpus generated items, by using the scikit learn implementation."""
        golds = [i.golds for i in items]
        preds = [i.preds for i in items]
        # Single f1
        if self.num_classes == 2:
            fscore = sklearn.metrics.f1_score(golds, preds, average=self.average)
            return np.max(fscore)

        # Multi f1
        f1s = []
        for i in range(self.num_classes):
            f1s.append(
                sklearn.metrics.f1_score(
                    y_true=[g == i for g in golds], y_pred=[p == i for p in preds], average=self.average
                )
            )
        return float(np.mean(f1s))


class CorpusLevelTranslationMetric(CorpusLevelComputation):
    def __init__(self, metric_type: str, lang: Literal["zh", "ja", "ko", ""] = ""):
        """Stores the relevant parameters for a corpus level translation metric.

        Args:
            metric_type (str): Can be any of bleu, chrf, or ter depending on the metric to use.
            lang (str): Language code for the translation metric.
        """
        self.metric_type = metric_type
        self.lang = lang

    def get_metric(self):
        if self.metric_type == "bleu":
            import nltk

            try:  # NEVER hit the network: sacrebleu tokenizes on its own, punkt_tab isn't required, and
                nltk.data.find("tokenizers/punkt_tab")  # this runs in ~1000 bootstrap workers where an
            except LookupError:  # nltk.download() online update-check hangs/times out on offline nodes.
                pass
            return sacrebleu.BLEU(trg_lang=self.lang)
        elif self.metric_type == "chrf":
            return sacrebleu.CHRF()
        elif self.metric_type == "chrf++":
            return sacrebleu.CHRF(word_order=2)
        elif self.metric_type == "ter":
            return sacrebleu.TER(asian_support=True if self.lang != "" else False)
        else:
            raise ValueError(f"Unknown corpus level translation metric type : {self.metric_type}")

    def compute_corpus(self, items: list[GenerativeCorpusMetricInput]) -> float:
        """Computes the metric score over all the corpus generated items, by using the sacrebleu implementation."""
        metric = self.get_metric()
        golds = [i.golds for i in items]
        preds = []
        for i in items:
            pred = as_list(i.preds)
            if len(pred) > 1:
                logger.info(
                    f"Multiple predictions present, keeping only the first prediction (when computing sacrebleu.{metric.__name__})."
                )
            preds.append(pred[0])

        if self.metric_type == "bleu":
            golds = [[gold[0] for gold in golds]]

        corpus_score = metric.corpus_score(hypotheses=preds, references=golds)
        score = corpus_score.score
        results = float(score)
        return results


class CorpusLevelPerplexityMetric(CorpusLevelComputation):
    def __init__(self, metric_type: str):
        """Stores the relevant parameter for a corpus level perplexity metric.
        Perplexity metrics compute more or less the same thing, which is a variation on the
        average of log-probabilities over a sequence, but the normalization and processing applied
        is different depending on the metric type.
        Perplexity uses an exponential and no weights for the average, weighted perplexity uses an exponential
        and the number of words as weights for the log-prob average, and bits per byte uses the number of bits
        for normalization and divides the results by log(2).

        Args:
            metric_type (str): Can be any of `perplexity`, `weighted_perplexity` or `bits_per_byte`
        """
        if metric_type not in ["perplexity", "weighted_perplexity", "bits_per_byte"]:
            raise ValueError(f"Unknown corpus level perplexity metric type : {metric_type}")

        self.metric_type = metric_type

    def compute_corpus(self, items: list[PerplexityCorpusMetricInput]):
        """Computes the metric score over all the corpus generated items."""
        logprobs = [i.logprobs for i in items]
        weights = [i.weights for i in items]

        if self.metric_type == "perplexity":
            return math.exp(-np.mean(logprobs))
        if self.metric_type == "weighted_perplexity":
            return math.exp(-sum(logprobs) / sum(weights))
        if self.metric_type == "bits_per_byte":
            return -sum(logprobs) / sum(weights) * 1 / math.log(2)


# --------------------------------------------------------------------------------------------------
# Neural MT metrics (COMET, MetricX) as CORPUS-level: score the whole set in one batched pass, on GPU.
# The per-sample (SampleLevelMetric) versions re-init a Lightning Trainer / forward per example -> hours
# for a full test set. These score all samples in one predict() call -> minutes. Auto-use GPU if present.
# --------------------------------------------------------------------------------------------------
@dataclass
class MTWithSourceCorpusMetricInput(CorpusMetricInput):
    """Per-sample input for reference+source MT metrics (COMET/MetricX): keeps the source segment too."""

    source: str
    gold: str
    pred: str


class MTSourcePreparator(Preparator):
    """Cheap per-sample step: emit (source, gold, pred). The heavy scoring happens once at corpus level."""

    def __init__(self, source_column: str = "source"):
        self.source_column = source_column

    def prepare(self, doc, model_response, **kwargs) -> MTWithSourceCorpusMetricInput:
        return MTWithSourceCorpusMetricInput(
            source=doc.specific[self.source_column],
            gold=doc.get_golds()[0],
            pred=model_response.final_text[0],
        )


class CorpusLevelCOMET(CorpusLevelComputation):
    def __init__(self, model_name: str = "Unbabel/wmt22-comet-da", batch_size: int = 64, gpus=None, accelerator=None):
        import torch as _torch

        _cuda = _torch.cuda.is_available()
        self.model_name = model_name
        self.batch_size = batch_size
        self.gpus = (1 if _cuda else 0) if gpus is None else gpus
        self.accelerator = ("cuda" if _cuda else "cpu") if accelerator is None else accelerator
        self._model = None
        # Per-sample score cache keyed by (source, gold, pred). CRUCIAL: stderr is computed by
        # bootstrap_stderr, which re-calls this ~1000x on RESAMPLED items (info_loggers.py). Without a
        # cache that would re-run the neural model ~1000x (~50h). With it, the model runs once over the
        # unique items and every bootstrap resample is a dict lookup + mean.
        self._cache: dict = {}

    def __getstate__(self):
        # bootstrap_stderr pickles this metric into mp.Pool workers. The loaded COMET model carries
        # an unpicklable GPU forward hook (ROCm) -> drop it: workers only ever hit the populated cache.
        state = self.__dict__.copy()
        state["_model"] = None
        return state

    def compute_corpus(self, items: list[MTWithSourceCorpusMetricInput]) -> float:
        keys = [(i.source, i.gold, i.pred) for i in items]
        missing = [k for k in dict.fromkeys(keys) if k not in self._cache]  # unique, order-preserving
        if missing:
            if self._model is None:
                from comet import download_model, load_from_checkpoint

                logger.info(f"Loading COMET model {self.model_name} (corpus, batched)...")
                self._model = load_from_checkpoint(download_model(self.model_name))
            data = [{"src": k[0], "mt": k[2], "ref": k[1]} for k in missing]
            # num_workers=0: the model carries an unpicklable GPU forward hook on ROCm -> worker spawn fails.
            output = self._model.predict(
                data, batch_size=self.batch_size, gpus=self.gpus, accelerator=self.accelerator,
                num_workers=0, progress_bar=False,
            )
            for k, s in zip(missing, output.scores):
                self._cache[k] = float(s) * 100
        return float(np.mean([self._cache[k] for k in keys]))


class CorpusLevelMetricX(CorpusLevelComputation):
    def __init__(
        self,
        model_name: str = "google/metricx-24-hybrid-large-v2p6",
        tokenizer_name: str = "google/mt5-large",
        batch_size: int = 16,
        device=None,
    ):
        import torch as _torch

        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.batch_size = batch_size
        self.device = ("cuda" if _torch.cuda.is_available() else "cpu") if device is None else device
        self._model = None
        self._tokenizer = None
        self._cache: dict = {}  # per-sample cache (source,gold,pred)->score; see CorpusLevelCOMET note

    def __getstate__(self):
        # See CorpusLevelCOMET.__getstate__: drop the loaded model/tokenizer before pickling into
        # bootstrap_stderr mp.Pool workers; they only read the already-populated cache.
        state = self.__dict__.copy()
        state["_model"] = None
        state["_tokenizer"] = None
        return state

    def compute_corpus(self, items: list[MTWithSourceCorpusMetricInput]) -> float:
        import torch
        from transformers import AutoTokenizer

        from lighteval.metrics.imports.metricx_model import MetricXModel

        keys = [(i.source, i.gold, i.pred) for i in items]
        missing = list(dict.fromkeys(k for k in keys if k not in self._cache))  # unique, order-preserving
        if missing:
            if self._model is None:
                logger.info(f"Loading MetricX model {self.model_name} (corpus, batched)...")
                self._model = MetricXModel(self.model_name, device=self.device)
                self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            texts = [f"candidate: {k[2]} reference: {k[1]} source: {k[0]}" for k in missing]
            for start in range(0, len(texts), self.batch_size):
                ck_keys = missing[start : start + self.batch_size]
                enc = [self._tokenizer(t, truncation=True, max_length=1024) for t in texts[start : start + self.batch_size]]
                for e in enc:  # MetricX drops the trailing EOS the tokenizer appends
                    e["input_ids"] = e["input_ids"][:-1]
                    e["attention_mask"] = e["attention_mask"][:-1]
                batch = self._tokenizer.pad(enc, return_tensors="pt")
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                with torch.no_grad():
                    preds = self._model.predict(input_ids, attention_mask)
                for k, s in zip(ck_keys, preds.detach().cpu().tolist()):
                    self._cache[k] = float(s)
        return float(np.mean([self._cache[k] for k in keys]))

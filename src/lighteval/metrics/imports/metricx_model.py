# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MetricX model wrapper using MT5ForConditionalGeneration from transformers.

Instead of vendoring the custom MT5ForRegression class (which has compatibility
issues with newer transformers versions), we load the weights into the standard
MT5ForConditionalGeneration model and extract the regression prediction
(logit at vocab position 250089, clamped to [0, 25]) in the same way MetricX does.
"""

import torch
from transformers import MT5ForConditionalGeneration


class MetricXModel:
    """Wrapper that loads a MetricX checkpoint and performs regression inference."""

    def __init__(self, model_name: str, device: str = "cpu"):
        self.model = MT5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        self.device = device

    def predict(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor) -> torch.FloatTensor:
        """Run MetricX regression inference.

        Args:
            input_ids: Tokenized input (batch, seq_len), with EOS already removed.
            attention_mask: Attention mask (batch, seq_len), with EOS already removed.

        Returns:
            Prediction scores (batch,), clamped to [0, 25]. Lower is better.
        """
        batch_size = input_ids.size(0)
        decoder_input_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=self.device)

        with torch.no_grad():
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
            )

        # 250089 = <extra_id_10>, the token MetricX uses for regression output
        predictions = output.logits[:, 0, 250089]
        return torch.clamp(predictions, 0, 25)

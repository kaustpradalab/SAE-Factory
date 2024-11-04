# Copyright 2024 KAUST PRADA Lab.
#
# This code is inspired by the LLAMA Factory and TRANSFORMER Lens Library
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

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    dataset_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The name of dataset(s) to use for sae training."
                "You can use both tokenized(e.g. apollo-research/roneneldan-TinyStories-tokenizer-gpt2 ) or the raw text dataset from huggingface or your local path."
            )
        },
    )
    is_dataset_tokenized: bool = field(
        default=False,
        metadata={
            "help": (
                "Set it to True if you are using a tokeniazed dataset, e.g. apollo-research/roneneldan-TinyStories-tokenizer-gpt2"
            )
        },
    )
    streaming: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to stream the dataset. Streaming large datasets is usually practical."
            )
        },
    )

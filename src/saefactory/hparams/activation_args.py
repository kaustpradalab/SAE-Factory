# Copyright 2024 KAUST PRADA Lab.
#
# This code is inspired by the SAELens Library
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
from typing import Literal


@dataclass
class ActivationsArguments:
    """Arguments for generating and catching the activations."""

    context_size: int = field(
        default=128,
        metadata={
            "help": (
                "The context size to use when generating activations on which to train the SAE."
            )
        },
    )
    use_cached_activations: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use cached activations. This is useful when doing sweeps over the same activations."
            )
        },
    )
    cached_activations_path: bool = field(
        default=None, metadata={"help": ("The path to the cached activations.")}
    )
    store_batch_size_prompts: int = field(
        default=32,
        metadata={
            "help": (
                "The batch size for storing activations. "
                "This controls how many prompts are in the batch of the language model when generating actiations."
            )
        },
    )
    normalize_activations: Literal["none", "expected_average_only_in"] = field(
        default="none",
        metadata={
            "help": (
                "Activation Normalization Strategy. Either none, expected_average_only_in (estimate the average activation norm and divide activations by it -> this can be folded post training and set to None), or constant_norm_rescale (at runtime set activation norm to sqrt(d_in) and then scale up the SAE output)."
            )
        },
    )

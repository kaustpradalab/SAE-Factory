# Copyright 2024 KAUST PRADA Lab.
#
# This code is inspired by the SAE_lens Library
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
class SaeArguments:
    mse_loss_normalization: str = field(
        default=None, metadata={"help": "The normalization to use for the MSE loss."}
    )
    d_in: int = field(default=512, metadata={"help": "The input dimension of the SAE."})
    expansion_factor: int = field(
        default=16,
        metadata={
            "help": (
                "The expansion factor. Larger is better but more computationally expensive."
            )
        },
    )
    d_sae: int = field(
        default=None,
        metadata={
            "help": (
                "The output dimension of the SAE. If None, defaults to d_in * expansion_factor."
            )
        },
    )
    b_dec_init_method: Literal["geometric_median", "mean", "zeros"] = field(
        default="geometric_median",
        metadata={
            "help": (
                "The method to use to initialize the decoder bias. Zeros is likely fine."
                "The geometric median can be used to initialize the decoder weights."
            )
        },
    )
    apply_b_dec_to_input: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to apply the decoder bias to the input. Not currently advised."
            )
        },
    )
    normalize_sae_decoder: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to normalize the SAE decoder."
                "Unit normed decoder weights used to be preferred."
            )
        },
    )
    scale_sparsity_penalty_by_decoder_norm: bool = field(
        default=False,
        metadata={
            "help": ("Whether to scale the sparsity penalty by the decoder norm.")
        },
    )
    decoder_heuristic_init: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use heuristic initialization for the decoder. See Anthropic April Update https://transformer-circuits.pub/2024/april-update/index.html"
            )
        },
    )
    init_encoder_as_decoder_transpose: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to initialize the encoder as the transpose of the decoder. See Anthropic April Update https://transformer-circuits.pub/2024/april-update/index.html."
            )
        },
    )
    l1_coefficient: float = field(
        default=0.001,
        metadata={"help": ("will control how sparse the feature activations are")},
    )
    l1_warm_up_steps: int = field(
        default=0,
        metadata={"help": ("this can help avoid too many dead features initially.")},
    )
    lp_norm: float = field(default=0, metadata={"help": ("The Lp norm.")})

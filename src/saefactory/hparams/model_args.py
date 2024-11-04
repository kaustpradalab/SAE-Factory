# Copyright 2024 KAUST PRADA Lab.
#
# This code is inspired by the LLAMA Factory and SAELens Library
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
class ModelArguments:
    model_name: str = field(
        default="tiny-stories-1L-21M",
        metadata={
            "help": (
                "The name of the model to use. This should be the name of the model in the Hugging Face model hub."
                "more options here: https://neelnanda-io.github.io/TransformerLens/generated/model_properties_table.html"
            )
        },
    )
    # TODO: support more model type
    model_class_name: Literal["HookedTransformer", "HookedMamba"] = field(
        default="HookedTransformer",
        metadata={
            "help": (
                "The name of the class of the model to use. This should be either HookedTransformer or HookedMamba."
            )
        },
    )

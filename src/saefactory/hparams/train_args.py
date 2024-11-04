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


@dataclass
class HookArguments:
    """Choose your hook point."""

    hook_name: str = field(
        default=None,
        metadata={
            "help": (
                "A valid hook point (see more details here: https://neelnanda-io.github.io/TransformerLens/generated/demos/Main_Demo.html#Hook-Points)"
            )
        },
    )

    hook_layer: int = field(
        default=0,
        metadata={
            "help": ("The layer number of the model where the hook should be placed.")
        },
    )


@dataclass
class TrainArguments:
    lr: float = field(
        default=5e-5,
        metadata={"help": ("The learning rate. Lower the better?")},
    )

    adam_beta1: float = field(
        default=0.9, metadata={"help": ("The beta1 parameter for Adam.")}
    )
    adam_beta2: float = field(
        default=0.999, metadata={"help": ("The beta2 parameter for Adam.")}
    )
    lr_scheduler_name: str = field(
        default="constant",
        metadata={
            "help": (
                "The name of the learning rate scheduler to use."
                "options: constant, cosineannealing, cosineannealingwarmrestarts"
            )
        },
    )
    lr_warm_up_steps: int = field(
        default=0,
        metadata={
            "help": (
                "The number of warm-up steps for the learning rate"
                "This can help avoid too many dead features initially."
            )
        },
    )
    lr_decay_steps: int = field(
        default=0,
        metadata={
            "help": "The number of decay steps for the learning rate, e.g. total_training_steps // 5, this will help us avoid overfitting."
        },
    )
    train_batch_size_tokens: int = field(
        default=4096,
        metadata={
            "help": (
                "The batch size for training."
                "This controls the batch size of the SAE Training loop."
            )
        },
    )
    device: str = field(
        default="cuda", metadata={"help": ("The device to use. Usually cuda.")}
    )
    seed: int = field(default=42, metadata={"help": ("The seed to use.")})
    dtype: str = str(default="float32", metadata={"help": ("The data type to use.")})

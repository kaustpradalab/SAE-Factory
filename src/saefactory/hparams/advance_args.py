# Copyright 2024 KAUST PRADA Lab.
#
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
class ResamplingArguments:
    use_ghost_grads: bool = field(
        default=False, metadata={"help": ("Whether to use ghost gradients.")}
    )
    feature_sampling_window: int = field(
        default=2000, metadata={"help": ("The feature sampling window.")}
    )
    dead_feature_window: int = field(
        default=1000,
        metadata={
            "help": (
                "The dead feature window."
                "would effect resampling or ghost grads if we were using it."
            )
        },
    )
    dead_feature_threshold: float = field(
        default=1e-08,
        metadata={
            "help": (
                "The dead feature threshold."
                "would effect resampling or ghost grads if we were using it."
            )
        },
    )

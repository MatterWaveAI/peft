# Copyright 2026-present the HuggingFace Inc. team.
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
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


DEFAULT_GRT_LORA_TARGET_MODULES: list[str] = ["attention", "mixer", "ffn"]
DEFAULT_GRT_LORA_TARGET_PARAMETERS: list[str] = [
    "q_angles",
    "k_angles",
    "v_angles",
    "o_angles",
    "angles",
    "expand_angles",
    "contract_angles",
]


@dataclass
class GRTLoraConfig(PeftConfig):
    """
    Configuration class for Toroidal Rotational LoRA (`GRTLoraModel`) on Geometric Rotation Transformer (GRT) models.

    GRT layers (`GRTAttention`, `GRTMixer`, `GRTFeedForward`, `GRTEmbedding`) contain no `nn.Linear` projections;
    instead they parameterize planar Givens rotation chains via angle tensors on the compact torus
    `T^n = (S^1)^n`. `GRTLoraConfig` configures low-rank angular perturbations
    `Delta theta = (lora_alpha / r) * (B @ A)` wrapped onto `[0, 2*pi)` via `(theta_0 + Delta theta) mod 2*pi`.

    Args:
        r (`int`, *optional*, defaults to `8`):
            Low-rank dimension of the toroidal angular perturbation matrices `lora_A` and `lora_B`.
        lora_alpha (`int`, *optional*, defaults to `16`):
            Scaling factor `lora_alpha / r` applied to the angular perturbation.
        lora_dropout (`float`, *optional*, defaults to `0.0`):
            Dropout probability applied to `lora_A` during training.
        target_modules (`Optional[Union[list[str], str]]`, *optional*, defaults to `None`):
            Module names (or regex) to adapt. When `None`, defaults to `["attention", "mixer", "ffn"]`.
        target_parameters (`list[str]`, *optional*):
            Names of angle parameters on targeted GRT modules to adapt. Defaults to
            `["q_angles", "k_angles", "v_angles", "o_angles", "angles", "expand_angles", "contract_angles"]`.
        wrap_mod_2pi (`bool`, *optional*, defaults to `True`):
            Whether to wrap effective angles `(theta_0 + Delta theta) mod 2*pi` onto the compact torus `[0, 2*pi)`.
        adapt_coupling (`bool`, *optional*, defaults to `False`):
            Whether to also attach low-rank adapters to Clifford bivector coupling tensors (`q_coupling`, `k_coupling`,
            `v_coupling`, `o_coupling`, `coupling`).
        exclude_modules (`Optional[Union[list[str], str]]`, *optional*, defaults to `None`):
            Module names or regex to exclude from adaptation.
        init_lora_weights (`bool`, *optional*, defaults to `True`):
            Whether to initialize `lora_A` with Kaiming uniform and `lora_B` with zeros so that `Delta theta = 0` at
            initialization (preserving exact base model outputs).
        layers_to_transform (`Optional[Union[list[int], int]]`, *optional*, defaults to `None`):
            Layer indices to transform.
        layers_pattern (`Optional[Union[list[str], str]]`, *optional*, defaults to `None`):
            Layer pattern name if `layers_to_transform` is specified.
        rank_pattern (`Optional[dict]`, *optional*, defaults to `{}`):
            Mapping from layer names or regexes to custom ranks `r`.
        alpha_pattern (`Optional[dict]`, *optional*, defaults to `{}`):
            Mapping from layer names or regexes to custom `lora_alpha` values.
        modules_to_save (`Optional[list[str]]`, *optional*, defaults to `None`):
            List of extra modules to mark trainable and save in the checkpoint.
    """

    r: int = field(default=8, metadata={"help": "Toroidal Rotational LoRA rank."})
    lora_alpha: int = field(default=16, metadata={"help": "Toroidal Rotational LoRA alpha scaling factor."})
    lora_dropout: float = field(default=0.0, metadata={"help": "Dropout probability for LoRA angular factors."})
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the GRT modules to adapt (e.g. ['attention', 'mixer', 'ffn'])."
        },
    )
    target_parameters: list[str] = field(
        default_factory=lambda: list(DEFAULT_GRT_LORA_TARGET_PARAMETERS),
        metadata={
            "help": "Angle parameter names on targeted GRT modules to adapt with low-rank toroidal perturbations."
        },
    )
    wrap_mod_2pi: bool = field(
        default=True,
        metadata={"help": "Wrap effective angles onto the compact torus [0, 2*pi) via modulo 2*pi."},
    )
    adapt_coupling: bool = field(
        default=False,
        metadata={"help": "Whether to also adapt Clifford bivector coupling tensors with low-rank perturbations."},
    )
    exclude_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={"help": "List of module names or regex expression of the module names to exclude."},
    )
    init_lora_weights: bool = field(
        default=True,
        metadata={"help": "Whether to initialize lora_B to zero and lora_A with Kaiming uniform."},
    )
    layers_to_transform: Optional[Union[list[int], int]] = field(
        default=None,
        metadata={"help": "The layer indexes to transform."},
    )
    layers_pattern: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={"help": "The layer pattern name, used only if `layers_to_transform` is not None."},
    )
    rank_pattern: Optional[dict] = field(
        default_factory=dict,
        metadata={"help": "The mapping from layer names or regexp expression to ranks different from `r`."},
    )
    alpha_pattern: Optional[dict] = field(
        default_factory=dict,
        metadata={"help": "The mapping from layer names or regexp expression to alphas different from `lora_alpha`."},
    )
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={"help": "List of modules apart from GRTLora layers to be set as trainable and saved."},
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.GRT_LORA
        if self.r <= 0:
            raise ValueError(f"`r` must be a positive integer, got {self.r}.")
        if self.lora_dropout < 0.0 or self.lora_dropout >= 1.0:
            raise ValueError(f"`lora_dropout` must be in `[0, 1)`, got {self.lora_dropout}.")
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        self.exclude_modules = (
            set(self.exclude_modules) if isinstance(self.exclude_modules, list) else self.exclude_modules
        )
        if isinstance(self.target_parameters, (set, tuple)):
            self.target_parameters = list(self.target_parameters)
        if self.layers_pattern and self.layers_to_transform is None:
            raise ValueError("When `layers_pattern` is specified, `layers_to_transform` must also be specified.")

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

from collections.abc import Sequence
from contextlib import contextmanager
import math
from typing import Any, Optional
import warnings

import torch
from torch import nn

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.other import _set_layer_requires_grad

from .config import DEFAULT_GRT_LORA_TARGET_PARAMETERS, GRTLoraConfig


TWO_PI = 2.0 * math.pi

COUPLING_PARAMETER_NAMES: tuple[str, ...] = (
    "q_coupling",
    "k_coupling",
    "v_coupling",
    "o_coupling",
    "coupling",
)


class GRTRotationalLoraAdapter(nn.Module):
    """
    Low-rank toroidal angular perturbation adapter for a collection of GRT angle (or coupling) parameters on a
    single GRT module (`GRTAttention`, `GRTMixer`, `GRTFeedForward`, `GRTEmbedding`).

    For each adapted parameter `theta_0` of shape `(..., d_out, d_in)` (where `d_out = prod(shape[:-1])` and
    `d_in = shape[-1]`), stores low-rank factors:
    - `lora_A[param_name]`: `nn.Parameter` of shape `(r, d_in)` initialized with Kaiming uniform
    - `lora_B[param_name]`: `nn.Parameter` of shape `(d_out, r)` initialized with zeros

    so that the low-rank angular perturbation is:
        `Delta theta = (lora_alpha / r) * reshape(lora_B @ dropout(lora_A), shape)`
    and when `wrap_mod_2pi=True`, the effective angle on the compact torus `T^n = (S^1)^n` is:
        `theta_eff = (theta_0 + Delta theta) mod 2*pi`
    """

    def __init__(
        self,
        param_shapes: dict[str, tuple[int, ...]],
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        wrap_mod_2pi: bool = True,
        init_lora_weights: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError(f"`r` must be a positive integer, got {r}.")

        self.param_shapes = {k: tuple(int(d) for d in v) for k, v in param_shapes.items()}
        self.r = int(r)
        self.lora_alpha = float(lora_alpha)
        self.scaling = self.lora_alpha / float(self.r)
        self.wrap_mod_2pi = bool(wrap_mod_2pi)
        self.lora_dropout = nn.Dropout(p=float(lora_dropout)) if lora_dropout > 0.0 else nn.Identity()

        tkwargs: dict[str, Any] = {}
        if device is not None:
            tkwargs["device"] = device
        if dtype is not None:
            tkwargs["dtype"] = dtype

        self.lora_A = nn.ParameterDict({})
        self.lora_B = nn.ParameterDict({})

        for param_name, shape in self.param_shapes.items():
            if len(shape) < 2:
                raise ValueError(
                    f"GRTRotationalLoraAdapter requires parameter {param_name!r} to have at least 2 dimensions, "
                    f"got shape {shape}."
                )
            d_in = int(shape[-1])
            d_out = int(math.prod(shape[:-1]))
            self.lora_A[param_name] = nn.Parameter(torch.empty(self.r, d_in, **tkwargs))
            self.lora_B[param_name] = nn.Parameter(torch.zeros(d_out, self.r, **tkwargs))

        self.reset_parameters(init_lora_weights=init_lora_weights)

    @torch.no_grad()
    def reset_parameters(self, init_lora_weights: bool = True) -> None:
        for param_name in self.lora_A:
            if self.lora_A[param_name].device.type == "meta":
                continue
            nn.init.kaiming_uniform_(self.lora_A[param_name], a=math.sqrt(5))
            if init_lora_weights:
                nn.init.zeros_(self.lora_B[param_name])
            else:
                nn.init.kaiming_uniform_(self.lora_B[param_name], a=math.sqrt(5))

    def get_delta_theta(self, param_name: str, target_shape: Optional[Sequence[int]] = None) -> torch.Tensor:
        if param_name not in self.lora_A or param_name not in self.lora_B:
            raise KeyError(f"Parameter {param_name!r} is not adapted by this GRTRotationalLoraAdapter.")
        shape = tuple(target_shape) if target_shape is not None else self.param_shapes[param_name]
        a = self.lora_dropout(self.lora_A[param_name]) if self.training else self.lora_A[param_name]
        b = self.lora_B[param_name]
        cast_fp32 = a.device.type == "cpu" and a.dtype in (torch.float16, torch.bfloat16)
        if cast_fp32:
            delta_2d = (b.float() @ a.float()) * self.scaling
            delta_2d = delta_2d.to(dtype=b.dtype)
        else:
            delta_2d = (b @ a) * self.scaling
        return delta_2d.reshape(shape)

    def forward(self, base_angles: torch.Tensor, param_name: str) -> torch.Tensor:
        delta = self.get_delta_theta(param_name, base_angles.shape).to(
            device=base_angles.device, dtype=base_angles.dtype
        )
        effective = base_angles + delta
        if self.wrap_mod_2pi and param_name not in COUPLING_PARAMETER_NAMES:
            effective = torch.fmod(effective, TWO_PI)
        return effective


def _discover_grt_target_parameters(
    module: nn.Module,
    target_parameters: Sequence[str],
    adapt_coupling: bool = False,
) -> dict[str, tuple[int, ...]]:
    """
    Find all rotational angle (and optionally coupling) parameters on `module` matching `target_parameters`.
    """
    target_set = set(target_parameters)
    matched: dict[str, tuple[int, ...]] = {}
    for param_name, param in module.named_parameters(recurse=False):
        if param is None or param.numel() == 0 or param.ndim < 2:
            continue
        is_match = False
        if param_name in target_set:
            is_match = True
        elif param_name in ("expand_base_angles", "expert_expand_angles") and "expand_angles" in target_set:
            is_match = True
        elif param_name in ("contract_base_angles", "expert_contract_angles") and "contract_angles" in target_set:
            is_match = True
        elif adapt_coupling and param_name in COUPLING_PARAMETER_NAMES:
            is_match = True

        if is_match:
            matched[param_name] = tuple(int(d) for d in param.shape)
    return matched


class GRTLoraLayer(nn.Module, BaseTunerLayer):
    """
    PEFT Tuner Layer wrapping a GRT rotational module (`GRTAttention`, `GRTMixer`, `GRTFeedForward`, `GRTEmbedding`,
    etc.) with low-rank toroidal phase-angle perturbations:
        `theta_eff = (theta_0 + (lora_alpha / r) * (B @ A)) mod 2*pi`

    When unmerged, `GRTLoraLayer` dynamically computes `theta_eff` and injects it into `base_layer` for the duration of
    the forward pass while preserving full autograd flow to `lora_A` and `lora_B`.
    When merged via `merge()` or `merge_and_unload()`, `theta_eff` is folded directly into the base angle parameters
    in-place with zero inference overhead, and `unmerge()` restores the exact pre-merge base angles `theta_0`.
    """

    adapter_layer_names: tuple[str, ...] = ("lora_A", "lora_B")
    other_param_names: tuple[str, ...] = (
        "r",
        "lora_alpha",
        "scaling",
        "lora_dropout",
        "wrap_mod_2pi",
        "adapt_coupling",
    )

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        config: GRTLoraConfig,
        r: int = 8,
        lora_alpha: int = 16,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.r: dict[str, int] = {}
        self.lora_alpha: dict[str, float] = {}
        self.scaling: dict[str, float] = {}
        self.wrap_mod_2pi: dict[str, bool] = {}
        self.adapt_coupling: dict[str, bool] = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        self._grt_lora_adapters: dict[str, GRTRotationalLoraAdapter] = {}
        self._saved_base_angles: dict[str, dict[str, torch.Tensor]] = {}
        self._disable_adapters = False
        self.merged_adapters: list[str] = []
        self._active_adapter: str | list[str] = adapter_name
        self.kwargs = kwargs

        self.update_layer(
            adapter_name=adapter_name,
            r=r,
            lora_alpha=lora_alpha,
            config=config,
            **kwargs,
        )

    def _get_base_layer_device_and_dtype(self, base_layer: nn.Module):
        for param in base_layer.parameters(recurse=False):
            if param is not None and param.numel() > 0 and param.is_floating_point():
                return param.device, param.dtype
        for param in base_layer.parameters():
            if param is not None and param.numel() > 0:
                return param.device, param.dtype
        return None, None

    def update_layer(
        self,
        adapter_name: str,
        r: int,
        lora_alpha: float,
        config: GRTLoraConfig,
        **kwargs: Any,
    ) -> None:
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        base_layer = self.get_base_layer()
        target_params = getattr(config, "target_parameters", None) or DEFAULT_GRT_LORA_TARGET_PARAMETERS
        param_shapes = _discover_grt_target_parameters(
            base_layer,
            target_parameters=target_params,
            adapt_coupling=bool(getattr(config, "adapt_coupling", False)),
        )
        if not param_shapes:
            raise ValueError(
                f" Target module {base_layer.__class__.__name__} has no matching GRT angle parameters for "
                f"target_parameters={list(target_params)}."
            )

        device, dtype = self._get_base_layer_device_and_dtype(base_layer)
        rot_adapter = GRTRotationalLoraAdapter(
            param_shapes=param_shapes,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=config.lora_dropout,
            wrap_mod_2pi=config.wrap_mod_2pi,
            init_lora_weights=config.init_lora_weights,
            dtype=dtype,
            device=device,
        )

        self.r[adapter_name] = int(r)
        self.lora_alpha[adapter_name] = float(lora_alpha)
        self.scaling[adapter_name] = float(lora_alpha) / float(r)
        self.wrap_mod_2pi[adapter_name] = bool(config.wrap_mod_2pi)
        self.adapt_coupling[adapter_name] = bool(getattr(config, "adapt_coupling", False))
        self.lora_dropout[adapter_name] = rot_adapter.lora_dropout
        self.lora_A[adapter_name] = rot_adapter.lora_A
        self.lora_B[adapter_name] = rot_adapter.lora_B
        self._grt_lora_adapters[adapter_name] = rot_adapter

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters, inference_mode=config.inference_mode)

    def get_delta_weight(self, adapter_name: str, param_name: str) -> torch.Tensor:
        """
        Compute the low-rank angular perturbation `Delta theta = (lora_alpha / r) * (B @ A)` reshaped to `param_name`.
        """
        if adapter_name not in self.lora_A:
            raise ValueError(f"Adapter {adapter_name!r} not found.")
        base_param = getattr(self.get_base_layer(), param_name)
        a = (
            self.lora_dropout[adapter_name](self.lora_A[adapter_name][param_name])
            if self.training
            else self.lora_A[adapter_name][param_name]
        )
        b = self.lora_B[adapter_name][param_name]
        cast_fp32 = a.device.type == "cpu" and a.dtype in (torch.float16, torch.bfloat16)
        if cast_fp32:
            delta_2d = (b.float() @ a.float()) * self.scaling[adapter_name]
            delta_2d = delta_2d.to(dtype=b.dtype)
        else:
            delta_2d = (b @ a) * self.scaling[adapter_name]
        return delta_2d.reshape(base_param.shape)

    def _compute_effective_parameter(self, param_name: str, base_param: torch.Tensor) -> torch.Tensor:
        total_delta: Optional[torch.Tensor] = None
        should_wrap = False
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A:
                continue
            if param_name not in self.lora_A[active_adapter]:
                continue
            delta = self.get_delta_weight(active_adapter, param_name).to(
                device=base_param.device, dtype=base_param.dtype
            )
            total_delta = delta if total_delta is None else (total_delta + delta)
            if self.wrap_mod_2pi.get(active_adapter, True):
                should_wrap = True

        if total_delta is None:
            return base_param

        effective = base_param + total_delta
        if should_wrap and param_name not in COUPLING_PARAMETER_NAMES:
            effective = torch.fmod(effective, TWO_PI)
        return effective

    @contextmanager
    def _apply_toroidal_adapters(self):
        base_layer = self.get_base_layer()
        active_adapters = [a for a in self.active_adapters if a in self.lora_A]
        if not active_adapters:
            yield
            return

        adapted_param_names: list[str] = []
        seen: set[str] = set()
        for adapter_name in active_adapters:
            for pname in self.lora_A[adapter_name].keys():
                if pname not in seen and pname in base_layer._parameters and base_layer._parameters[pname] is not None:
                    seen.add(pname)
                    adapted_param_names.append(pname)

        if not adapted_param_names:
            yield
            return

        saved_params: dict[str, nn.Parameter] = {}
        try:
            for pname in adapted_param_names:
                orig_param = base_layer._parameters[pname]
                saved_params[pname] = orig_param
                effective_tensor = self._compute_effective_parameter(pname, orig_param)
                # Temporarily bind the computed tensor in _parameters dict so base_layer accesses it
                # transparently with full autograd graph back to lora_A and lora_B.
                base_layer._parameters[pname] = effective_tensor  # type: ignore[assignment]
            yield
        finally:
            for pname, orig_param in saved_params.items():
                base_layer._parameters[pname] = orig_param

    @torch.no_grad()
    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Fold `(theta_0 + (lora_alpha / r) * (B @ A)) mod 2*pi` directly into the base GRT angle parameters in-place
        with zero inference overhead.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return

        base_layer = self.get_base_layer()
        for active_adapter in adapter_names:
            if active_adapter not in self.lora_A:
                continue

            was_training = self.training
            self.eval()
            try:
                saved_for_adapter: dict[str, torch.Tensor] = {}
                for pname in self.lora_A[active_adapter].keys():
                    param = getattr(base_layer, pname, None)
                    if param is None:
                        continue
                    saved_for_adapter[pname] = param.data.detach().clone()
                    delta = self.get_delta_weight(active_adapter, pname).to(device=param.device, dtype=param.dtype)
                    merged_val = param.data + delta
                    if self.wrap_mod_2pi.get(active_adapter, True) and pname not in COUPLING_PARAMETER_NAMES:
                        merged_val = torch.fmod(merged_val, TWO_PI)

                    if safe_merge and not torch.isfinite(merged_val).all():
                        raise ValueError(
                            f"NaNs detected in the merged angles for parameter {pname!r}. "
                            f"The adapter {active_adapter!r} seems to be broken."
                        )
                    param.data.copy_(merged_val)
            finally:
                self.train(was_training)

            self._saved_base_angles[active_adapter] = saved_for_adapter
            self.merged_adapters.append(active_adapter)

    @torch.no_grad()
    def unmerge(self) -> None:
        """
        Restore the original pre-merge base angle parameters `theta_0` in-place.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        base_layer = self.get_base_layer()
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            saved_for_adapter = self._saved_base_angles.pop(active_adapter, None)
            if saved_for_adapter is not None:
                for pname, orig_data in saved_for_adapter.items():
                    param = getattr(base_layer, pname, None)
                    if param is not None:
                        param.data.copy_(orig_data.to(device=param.device, dtype=param.dtype))
            elif active_adapter in self.lora_A:
                for pname in self.lora_A[active_adapter].keys():
                    param = getattr(base_layer, pname, None)
                    if param is not None:
                        delta = self.get_delta_weight(active_adapter, pname).to(device=param.device, dtype=param.dtype)
                        unmerged_val = param.data - delta
                        if self.wrap_mod_2pi.get(active_adapter, True) and pname not in COUPLING_PARAMETER_NAMES:
                            unmerged_val = torch.fmod(unmerged_val, TWO_PI)
                        param.data.copy_(unmerged_val)

    def delete_adapter(self, adapter_name: str) -> None:
        super().delete_adapter(adapter_name)
        self._saved_base_angles.pop(adapter_name, None)
        self._grt_lora_adapters.pop(adapter_name, None)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            return self.base_layer(*args, **kwargs)
        if self.merged:
            return self.base_layer(*args, **kwargs)
        with self._apply_toroidal_adapters():
            return self.base_layer(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "base_layer":
                raise
            return getattr(self.base_layer, name)

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "grt_lora." + rep

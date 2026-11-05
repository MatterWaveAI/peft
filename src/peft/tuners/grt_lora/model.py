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

from contextlib import nullcontext
import re
from typing import Any, Optional
import warnings

import torch
from torch import nn
from tqdm import tqdm

from peft.config import PeftConfig
from peft.tuners.tuners_utils import (
    BaseTuner,
    BaseTunerLayer,
    _extend_unique,
    _filter_state_dict_by_key_prefixes,
    _find_minimal_target_modules,
    _get_submodules,
    _get_tuner_state_dict_key_prefixes,
    _is_key_under_prefixes,
    check_target_module_exists,
    onload_layer,
)
from peft.utils import TRANSFORMERS_MODELS_TO_GRT_LORA_TARGET_MODULES_MAPPING
from peft.utils.constants import DUMMY_TARGET_MODULES, MIN_TARGET_MODULES_FOR_OPTIMIZATION
from peft.utils.error import NoMatchingPeftModuleError
from peft.utils.integrations import init_empty_weights
from peft.utils.other import (
    AuxiliaryTrainingWrapper,
    ModulesToSaveWrapper,
    get_pattern_key,
    set_additional_trainable_modules,
)

from .config import DEFAULT_GRT_LORA_TARGET_MODULES, DEFAULT_GRT_LORA_TARGET_PARAMETERS, GRTLoraConfig
from .layer import GRTLoraLayer, _discover_grt_target_parameters


class GRTLoraModel(BaseTuner):
    """
    Creates a Toroidal Rotational LoRA (`GRTLoraModel`) model from a Geometric Rotation Transformer (`GRT`) model.

    Because GRT models contain zero `nn.Linear` projections and instead parameterize all attention, cross-head mixing,
    and feed-forward projections as planar Givens rotation chains on the compact torus `T^n = (S^1)^n`, `GRTLoraModel`
    wraps target GRT modules (`GRTAttention`, `GRTMixer`, `GRTFeedForward`, `GRTEmbedding`) with `GRTLoraLayer`
    instances that apply low-rank angular perturbations `Delta theta = (lora_alpha / r) * (B @ A) mod 2*pi`.

    Args:
        model ([`torch.nn.Module`]):
            The GRT model to be adapted.
        config ([`GRTLoraConfig`]):
            The configuration of the Toroidal Rotational LoRA model.
        adapter_name (`str`):
            The name of the adapter, defaults to `"default"`.

    Returns:
        `torch.nn.Module`: The adapted `GRTLoraModel`.
    """

    prefix: str = "grt_lora_"
    tuner_layer_cls = GRTLoraLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_GRT_LORA_TARGET_MODULES_MAPPING

    def _prepare_adapter_config(self, peft_config: PeftConfig, model_config: dict) -> PeftConfig:
        if peft_config.target_modules is None:
            model_type = model_config.get("model_type", "grt")
            target_modules = self.target_module_mapping.get(model_type, DEFAULT_GRT_LORA_TARGET_MODULES)
            if isinstance(target_modules, str):
                peft_config.target_modules = target_modules
            else:
                peft_config.target_modules = set(target_modules)
        return peft_config

    @staticmethod
    def _check_target_module_exists(peft_config: PeftConfig, key: str) -> bool | re.Match[str] | None:
        if peft_config.target_modules is None:
            peft_config.target_modules = set(DEFAULT_GRT_LORA_TARGET_MODULES)
        return check_target_module_exists(peft_config, key)

    def inject_adapter(
        self,
        model: nn.Module,
        adapter_name: str,
        autocast_adapter_dtype: bool = True,
        low_cpu_mem_usage: bool = False,
        state_dict: Optional[dict[str, torch.Tensor]] = None,
    ) -> None:
        """
        Discover target GRT modules containing rotational angle parameters and wrap them with `GRTLoraLayer`.
        """
        peft_config: GRTLoraConfig = self.peft_config[adapter_name]  # type: ignore[assignment]
        self._check_new_adapter_config(peft_config)
        self._check_tied_modules(model, peft_config)
        model_config = self.get_model_config(model)
        peft_config = self._prepare_adapter_config(peft_config, model_config)  # type: ignore[assignment]
        self._prepare_model(peft_config, model)

        named_modules = list(model.named_modules())
        key_list = [key for key, _ in named_modules]

        existing_adapter_prefixes: list[str] = []
        mapping_existing_parameter_requires_grad: list[tuple[nn.Parameter, bool]] = []
        existing_adapter_modules_training = self._get_adapter_modules_training(model)
        seen_parameters: set[int] = set()

        for key, module in named_modules:
            if isinstance(module, BaseTunerLayer):
                existing_adapter_prefixes.append(key + ".")
            for parameter_name, parameter in module.named_parameters(recurse=False):
                full_name = f"{key}.{parameter_name}" if key else parameter_name
                if id(parameter) in seen_parameters:
                    continue
                seen_parameters.add(id(parameter))
                if f".{adapter_name}." not in full_name and not full_name.endswith(f".{adapter_name}"):
                    mapping_existing_parameter_requires_grad.append((parameter, parameter.requires_grad))

        if not existing_adapter_prefixes:
            mapping_existing_parameter_requires_grad = []

        uses_dummy_target_modules = getattr(peft_config, "target_modules", None) == DUMMY_TARGET_MODULES
        if uses_dummy_target_modules:
            named_modules = []
            key_list = []

        if (
            isinstance(peft_config.target_modules, (list, set))
            and (len(peft_config.target_modules) >= MIN_TARGET_MODULES_FOR_OPTIMIZATION)
        ):
            suffixes = tuple("." + suffix for suffix in peft_config.target_modules)
            names_no_target = [
                name for name in key_list if (name not in peft_config.target_modules) and not name.endswith(suffixes)
            ]
            new_target_modules = _find_minimal_target_modules(peft_config.target_modules, names_no_target)
            if len(new_target_modules) < len(peft_config.target_modules):
                peft_config.target_modules = new_target_modules

        module_names_from_state_dict: set[str] = set()
        if state_dict is not None:
            for k in state_dict:
                clean_k = k.removeprefix("base_model.model.").removeprefix("_orig_mod.")
                for marker in (".lora_A.", ".lora_B.", ".grt_lora_adapters."):
                    if marker in clean_k:
                        module_names_from_state_dict.add(clean_k.split(marker, 1)[0])
                        break

        target_params = getattr(peft_config, "target_parameters", None) or DEFAULT_GRT_LORA_TARGET_PARAMETERS
        adapt_coupling = bool(getattr(peft_config, "adapt_coupling", False))

        targeted_module_names: list[str] = []
        targeted_parameter_names: list[str] = []
        module_names_to_inject: list[str] = []

        for key, module in named_modules:
            if not key:
                continue
            if any(key.startswith(adapter_key) for adapter_key in existing_adapter_prefixes):
                continue

            if state_dict is None:
                matched = bool(self._check_target_module_exists(peft_config, key))
            else:
                clean_key = key.removeprefix("_orig_mod.")
                matched = clean_key in module_names_from_state_dict

            if not matched:
                continue

            base_mod = module.get_base_layer() if isinstance(module, BaseTunerLayer) else module
            discovered = _discover_grt_target_parameters(
                base_mod,
                target_parameters=target_params,
                adapt_coupling=adapt_coupling,
            )
            if not discovered:
                continue

            targeted_module_names.append(key)
            module_names_to_inject.append(key)
            for pname in discovered:
                targeted_parameter_names.append(f"{key}.{pname}")

        if not targeted_module_names and not uses_dummy_target_modules:
            raise NoMatchingPeftModuleError(
                f"Target modules {peft_config.target_modules} with target_parameters={list(target_params)} "
                f"were not found in the base model. Please check the target modules/parameters and try again."
            )

        _extend_unique(self.targeted_module_names, targeted_module_names)
        _extend_unique(self.targeted_parameter_names, targeted_parameter_names)

        for key in module_names_to_inject:
            parent, target, target_name = _get_submodules(model, key)
            ctx = init_empty_weights if low_cpu_mem_usage else nullcontext
            with ctx():
                self._create_and_replace(peft_config, adapter_name, target, target_name, parent, current_key=key)

        self.set_adapter(self.active_adapters, inference_mode=peft_config.inference_mode)
        self._mark_only_adapters_as_trainable(model)

        set_additional_trainable_modules(
            model=model,
            peft_config=peft_config,
            model_config=BaseTuner.get_model_config(self),
            adapter_name=adapter_name,
            activate_adapter=adapter_name in self.active_adapters,
        )

        self._restore_adapter_modules_training(existing_adapter_modules_training)
        for module in model.modules():
            if isinstance(module, BaseTunerLayer):
                module.set_training(adapter_name, not peft_config.inference_mode)

        for parameter, requires_grad in mapping_existing_parameter_requires_grad:
            parameter.requires_grad = requires_grad

    def _create_and_replace(
        self,
        peft_config: PeftConfig,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        current_key: str,
        parameter_name: Optional[str] = None,
        **optional_kwargs: Any,
    ) -> None:
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        grt_config: GRTLoraConfig = peft_config  # type: ignore[assignment]
        r_key = get_pattern_key(grt_config.rank_pattern.keys(), current_key)
        alpha_key = get_pattern_key(grt_config.alpha_pattern.keys(), current_key)
        r = grt_config.rank_pattern.get(r_key, grt_config.r)
        lora_alpha = grt_config.alpha_pattern.get(alpha_key, grt_config.lora_alpha)

        kwargs = {
            "r": r,
            "lora_alpha": lora_alpha,
            **optional_kwargs,
        }

        if isinstance(target, GRTLoraLayer):
            target.update_layer(adapter_name=adapter_name, config=grt_config, **kwargs)
        else:
            new_module = self._create_new_module(grt_config, adapter_name, target, **kwargs)
            if adapter_name not in self.active_adapters:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(
        grt_config: GRTLoraConfig,
        adapter_name: str,
        target: nn.Module,
        **kwargs: Any,
    ) -> GRTLoraLayer:
        return GRTLoraLayer(target, adapter_name=adapter_name, config=grt_config, **kwargs)

    def _replace_module(self, parent: nn.Module, child_name: str, new_module: nn.Module, child: nn.Module) -> None:
        setattr(parent, child_name, new_module)
        if isinstance(child, AuxiliaryTrainingWrapper):
            return
        if hasattr(child, "base_layer"):
            child = child.base_layer

        meta = torch.device("meta")
        ref_param = next(child.parameters(), None)
        if ref_param is not None:
            for name, module in new_module.named_modules():
                if self.prefix in name or "lora_" in name or "grt_lora_adapters" in name:
                    if not any(p.device == meta for p in module.parameters()):
                        module.to(ref_param.device)

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        super()._mark_only_adapters_as_trainable(model)

    def _unload_and_optionally_merge(
        self,
        merge: bool = True,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names: Optional[list[str]] = None,
    ) -> nn.Module:
        if merge:
            self._check_merge_allowed()
            if adapter_names:
                for module in self.model.modules():
                    if isinstance(module, ModulesToSaveWrapper):
                        module.check_set_adapter(adapter_names)

        tuner_layer_names = {key for key, module in self.model.named_modules() if isinstance(module, BaseTunerLayer)}
        key_list = [
            key
            for key, _ in self.model.named_modules()
            if not _is_key_under_prefixes(key.rpartition(".")[0], tuner_layer_names)
        ]
        desc = "Unloading " + ("and merging " if merge else "") + "model"
        for key in tqdm(key_list, disable=not progressbar, desc=desc):
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            with onload_layer(target):
                if hasattr(target, "unload_and_optionally_merge_module"):
                    unloaded_module = target.unload_and_optionally_merge_module(
                        merge=merge, safe_merge=safe_merge, adapter_names=adapter_names
                    )
                    self._replace_module(parent, target_name, unloaded_module, target)
                elif hasattr(target, "base_layer"):
                    if merge:
                        target.merge(safe_merge=safe_merge, adapter_names=adapter_names)
                    self._replace_module(parent, target_name, target.get_base_layer(), target)

        if hasattr(self.model, "peft_config"):
            del self.model.peft_config

        return self.model

    def merge_and_unload(
        self,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names: Optional[list[str]] = None,
    ) -> torch.nn.Module:
        """
        Merge the toroidal rotational LoRA perturbations `(theta_0 + (lora_alpha / r) * (B @ A)) mod 2*pi` into the
        base GRT angle parameters and unload all `GRTLoraLayer` wrappers, returning a pure zero-overhead GRT model.
        """
        return self._unload_and_optionally_merge(
            merge=True,
            progressbar=progressbar,
            safe_merge=safe_merge,
            adapter_names=adapter_names,
        )

    def unload(self) -> torch.nn.Module:
        """
        Remove all `GRTLoraLayer` wrappers and return the unmodified base GRT model.
        """
        return self._unload_and_optionally_merge(merge=False)

    @classmethod
    def _get_adapter_state_dict(
        cls,
        model: nn.Module,
        config: PeftConfig,
        adapter_name: str,
        state_dict: dict[str, torch.Tensor],
        unwanted_adapter_names: list[str],
    ) -> dict[str, torch.Tensor]:
        """
        Select only canonical `lora_A` and `lora_B` entries (excluding aliases under `grt_lora_adapters` to prevent
        duplicate shared tensors in `state_dict`).
        """
        prefixes = _get_tuner_state_dict_key_prefixes(model)
        canonical_prefixes = {p for p in prefixes if not p.endswith(".grt_lora_adapters") and ".grt_lora_adapters." not in p}
        to_return = _filter_state_dict_by_key_prefixes(state_dict, canonical_prefixes)
        to_return.update(cls._get_learnable_bias_state_dict(model, state_dict, config))
        return to_return

    @classmethod
    def _remove_adapter_name_from_key(cls, key: str, adapter_name: str) -> str:
        """
        Remove `<adapter_name>` from `<module>.lora_A.<adapter_name>.<param_name>` -> `<module>.lora_A.<param_name>`.
        """
        for marker in (".lora_A.", ".lora_B.", ".grt_lora_adapters."):
            token = f"{marker}{adapter_name}."
            if token in key:
                prefix, _, suffix = key.rpartition(token)
                return f"{prefix}{marker}{suffix}"
        return super()._remove_adapter_name_from_key(key, adapter_name)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "model":
                raise
            return getattr(self.model, name)

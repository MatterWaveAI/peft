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

import math

import pytest
import torch
from transformers.models.grt import GRTConfig, GRTForCausalLM
from transformers.models.grt.modeling_grt import GRTAttention, GRTFeedForward, GRTMixer

from peft import (
    GRTLoraConfig,
    GRTLoraLayer,
    GRTLoraModel,
    PeftModel,
    PeftType,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
)
from peft.mapping import (
    PEFT_TYPE_TO_CONFIG_MAPPING,
    PEFT_TYPE_TO_PREFIX_MAPPING,
    PEFT_TYPE_TO_TUNER_MAPPING,
)


def _make_tiny_grt_config(**overrides) -> GRTConfig:
    defaults = dict(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        n_qkv_rotations=2,
        n_mixer_rotations=2,
        ffn_n_rotations=2,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    defaults.update(overrides)
    return GRTConfig(**defaults)


def _make_tiny_grt_model(**overrides) -> GRTForCausalLM:
    torch.manual_seed(42)
    config = _make_tiny_grt_config(**overrides)
    model = GRTForCausalLM(config)
    model.eval()
    return model


class TestGRTLoraTuner:
    def test_peft_type_registration(self):
        assert hasattr(PeftType, "GRT_LORA")
        assert PeftType.GRT_LORA == "GRT_LORA"
        assert PEFT_TYPE_TO_CONFIG_MAPPING[PeftType.GRT_LORA] is GRTLoraConfig
        assert PEFT_TYPE_TO_TUNER_MAPPING[PeftType.GRT_LORA] is GRTLoraModel
        assert PEFT_TYPE_TO_PREFIX_MAPPING[PeftType.GRT_LORA] == "grt_lora_"

    def test_trainable_parameters_and_zero_init_parity(self):
        base_model = _make_tiny_grt_model()
        input_ids = torch.tensor([[1, 5, 12, 23, 7, 99], [2, 18, 33, 41, 60, 11]], dtype=torch.long)

        with torch.no_grad():
            base_logits = base_model(input_ids=input_ids).logits.clone()

        lora_config = GRTLoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            task_type=TaskType.CAUSAL_LM,
        )
        peft_model = get_peft_model(base_model, lora_config)
        peft_model.print_trainable_parameters()

        trainable_params, all_params = peft_model.get_nb_trainable_parameters()
        assert trainable_params > 0
        assert trainable_params < all_params

        # Verify ONLY lora_A and lora_B parameters require gradients
        for name, param in peft_model.named_parameters():
            if param.requires_grad:
                assert ".lora_A." in name or ".lora_B." in name, f"Unexpected trainable param: {name}"
            else:
                assert ".lora_A." not in name and ".lora_B." not in name or ".grt_lora_adapters." in name

        # Zero-initialized lora_B gives exact forward parity with base GRTForCausalLM
        peft_model.eval()
        with torch.no_grad():
            peft_logits = peft_model(input_ids=input_ids).logits
        max_diff = (peft_logits - base_logits).abs().max().item()
        assert max_diff < 1e-6, f"Expected exact zero-init parity, got max_diff={max_diff}"

    def test_nonzero_lora_b_updates_logits_and_gradients_flow(self):
        base_model = _make_tiny_grt_model()
        input_ids = torch.tensor([[1, 10, 20, 30, 40]], dtype=torch.long)
        labels = input_ids.clone()

        with torch.no_grad():
            base_logits = base_model(input_ids=input_ids).logits.clone()

        lora_config = GRTLoraConfig(r=4, lora_alpha=8, lora_dropout=0.05, task_type=TaskType.CAUSAL_LM)
        peft_model = get_peft_model(base_model, lora_config)

        # Initialize lora_B to non-zero values
        torch.manual_seed(123)
        with torch.no_grad():
            for name, param in peft_model.named_parameters():
                if ".lora_B." in name and param.requires_grad:
                    param.normal_(mean=0.0, std=0.15)

        peft_model.eval()
        with torch.no_grad():
            adapted_logits = peft_model(input_ids=input_ids).logits
        logit_diff = (adapted_logits - base_logits).abs().max().item()
        assert logit_diff > 1e-3, f"Expected non-zero lora_B to alter logits, got max diff {logit_diff}"

        # Verify gradients flow cleanly to all lora_A and lora_B parameters and NOT to base parameters
        peft_model.train()
        peft_model.zero_grad()
        outputs = peft_model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()

        lora_a_grads = 0
        lora_b_grads = 0
        for name, param in peft_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Missing gradient on {name}"
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient on {name}"
                assert param.grad.abs().sum().item() > 0.0, f"Zero gradient on {name}"
                if ".lora_A." in name:
                    lora_a_grads += 1
                elif ".lora_B." in name:
                    lora_b_grads += 1
            else:
                assert param.grad is None, f"Frozen parameter {name} received gradient"

        assert lora_a_grads > 0 and lora_b_grads == lora_a_grads

    def test_merge_unmerge_and_merge_and_unload_parity(self):
        base_model = _make_tiny_grt_model()
        input_ids = torch.tensor([[1, 7, 14, 21, 28, 35]], dtype=torch.long)

        with torch.no_grad():
            orig_base_logits = base_model(input_ids=input_ids).logits.clone()

        lora_config = GRTLoraConfig(r=8, lora_alpha=16, wrap_mod_2pi=True, task_type=TaskType.CAUSAL_LM)
        peft_model = get_peft_model(base_model, lora_config)

        torch.manual_seed(99)
        with torch.no_grad():
            for name, param in peft_model.named_parameters():
                if ".lora_B." in name and param.requires_grad:
                    param.uniform_(-0.25, 0.25)

        peft_model.eval()
        with torch.no_grad():
            unmerged_logits = peft_model(input_ids=input_ids).logits.clone()

        # Test disable_adapter context
        with peft_model.disable_adapter():
            with torch.no_grad():
                disabled_logits = peft_model(input_ids=input_ids).logits
        assert (disabled_logits - orig_base_logits).abs().max().item() < 1e-6

        # Test merge_adapter()
        peft_model.merge_adapter()
        with torch.no_grad():
            merged_logits = peft_model(input_ids=input_ids).logits.clone()
        merge_diff = (merged_logits - unmerged_logits).abs().max().item()
        assert merge_diff < 1e-5, f"merge_adapter() parity diff {merge_diff} >= 1e-5"

        # Check wrapped angles are within (-2*pi, 2*pi)
        for module in peft_model.modules():
            if isinstance(module, GRTLoraLayer):
                base_layer = module.get_base_layer()
                for pname in module.lora_A["default"].keys():
                    p = getattr(base_layer, pname)
                    assert (p.abs() < 2.0 * math.pi + 1e-5).all()

        # Test unmerge_adapter()
        peft_model.unmerge_adapter()
        with torch.no_grad():
            re_unmerged_logits = peft_model(input_ids=input_ids).logits.clone()
        unmerge_diff = (re_unmerged_logits - unmerged_logits).abs().max().item()
        assert unmerge_diff < 1e-6, f"unmerge_adapter() parity diff {unmerge_diff} >= 1e-6"

        # Verify disabling after unmerge matches original base model
        with peft_model.disable_adapter():
            with torch.no_grad():
                restored_base_logits = peft_model(input_ids=input_ids).logits
        assert (restored_base_logits - orig_base_logits).abs().max().item() < 1e-6

        # Test merge_and_unload()
        unloaded_model = peft_model.merge_and_unload()
        assert isinstance(unloaded_model, GRTForCausalLM)
        assert not any(isinstance(m, GRTLoraLayer) for m in unloaded_model.modules())
        assert isinstance(unloaded_model.model.blocks[0].attention, GRTAttention)
        assert isinstance(unloaded_model.model.blocks[0].mixer, GRTMixer)
        assert isinstance(unloaded_model.model.blocks[0].ffn, GRTFeedForward)

        with torch.no_grad():
            unloaded_logits = unloaded_model(input_ids=input_ids).logits
        unload_diff = (unloaded_logits - unmerged_logits).abs().max().item()
        assert unload_diff < 1e-5, f"merge_and_unload() parity diff {unload_diff} >= 1e-5"

    def test_save_and_load_pretrained_roundtrip(self, tmp_path):
        base_model = _make_tiny_grt_model()
        input_ids = torch.tensor([[1, 3, 9, 27, 81, 15]], dtype=torch.long)

        lora_config = GRTLoraConfig(
            r=6,
            lora_alpha=12,
            lora_dropout=0.0,
            wrap_mod_2pi=True,
            adapt_coupling=True,
            task_type=TaskType.CAUSAL_LM,
        )
        peft_model = get_peft_model(base_model, lora_config)

        torch.manual_seed(777)
        with torch.no_grad():
            for name, param in peft_model.named_parameters():
                if ".lora_B." in name and param.requires_grad:
                    param.normal_(0.0, 0.2)

        peft_model.eval()
        with torch.no_grad():
            expected_logits = peft_model(input_ids=input_ids).logits.clone()

        save_dir = tmp_path / "grt_lora_ckpt"
        peft_model.save_pretrained(str(save_dir))

        assert (save_dir / "adapter_config.json").is_file()
        assert (save_dir / "adapter_model.safetensors").is_file()

        state_dict = get_peft_model_state_dict(peft_model)
        assert len(state_dict) > 0
        assert all(".lora_A." in k or ".lora_B." in k for k in state_dict.keys())
        assert not any(".grt_lora_adapters." in k for k in state_dict.keys())

        fresh_base_model = _make_tiny_grt_model()
        loaded_peft_model = PeftModel.from_pretrained(fresh_base_model, str(save_dir))
        loaded_peft_model.eval()

        assert isinstance(loaded_peft_model.peft_config["default"], GRTLoraConfig)
        assert loaded_peft_model.peft_config["default"].r == 6
        assert loaded_peft_model.peft_config["default"].lora_alpha == 12
        assert loaded_peft_model.peft_config["default"].adapt_coupling is True

        with torch.no_grad():
            loaded_logits = loaded_peft_model(input_ids=input_ids).logits

        roundtrip_diff = (loaded_logits - expected_logits).abs().max().item()
        assert roundtrip_diff < 1e-6, f"Round-trip save/load diff {roundtrip_diff} >= 1e-6"

        # Merge and unload the loaded model as well
        merged_loaded = loaded_peft_model.merge_and_unload()
        with torch.no_grad():
            merged_loaded_logits = merged_loaded(input_ids=input_ids).logits
        assert (merged_loaded_logits - expected_logits).abs().max().item() < 1e-5

    def test_mor_feedforward_angle_adaptation(self):
        base_model = _make_tiny_grt_model(ffn_n_experts=4, mor_mode="direct")
        input_ids = torch.tensor([[1, 4, 8, 16]], dtype=torch.long)

        lora_config = GRTLoraConfig(r=4, lora_alpha=8, task_type=TaskType.CAUSAL_LM)
        peft_model = get_peft_model(base_model, lora_config)

        # Ensure expert_expand_angles and expert_contract_angles were discovered & adapted
        assert any("expert_expand_angles" in p for p in peft_model.base_model.targeted_parameter_names)
        assert any("expert_contract_angles" in p for p in peft_model.base_model.targeted_parameter_names)

        with torch.no_grad():
            for name, param in peft_model.named_parameters():
                if ".lora_B." in name and param.requires_grad:
                    param.normal_(0.0, 0.1)

        peft_model.eval()
        with torch.no_grad():
            unmerged_out = peft_model(input_ids=input_ids).logits.clone()

        merged_model = peft_model.merge_and_unload()
        with torch.no_grad():
            merged_out = merged_model(input_ids=input_ids).logits
        assert (merged_out - unmerged_out).abs().max().item() < 1e-5

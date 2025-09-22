# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from typing import Any, List, Optional, Tuple, Generator, Dict

import torch
import torch.nn as nn
from torch import distributed as dist

from msmodelslim.app import DeviceType
from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.graph import AdapterConfig, MappingConfig, FusionConfig
from msmodelslim.model.common.layer_wise_forward import generated_decoder_layer_visit_func, \
    _TransformersForwardBreak
from msmodelslim.model.factory import ModelFactory
from msmodelslim.model.transformers import TransformersModel
from msmodelslim.utils.exception import InvalidModelError
from msmodelslim.utils.logging import logger_setter
from msmodelslim.utils.security.model import SafeGenerator
from .convert_fp8_to_bf16 import auto_convert_model_fp8_to_bf16, get_module_by_name
from .mtp_quant_module import warp_mtp_model, remove_zero_and_shift
from ..interface_hub import ModelInfoInterface, ModelSlimPipelineInterfaceV1, \
    IterSmoothInterface, FlexSmoothQuantInterface, ModelHookInterface


@ModelFactory.register("DeepSeek-V3")
@ModelFactory.register("DeepSeek-V3-0324")
@ModelFactory.register("DeepSeek-R1")
@ModelFactory.register("DeepSeek-R1-0528")
@ModelFactory.register("DeepSeek-V3.1")
@logger_setter("msmodelslim.model.deepseek_v3")
class DeepSeekV3ModelAdapter(TransformersModel,
                             ModelInfoInterface,  # support naive quantization
                             ModelSlimPipelineInterfaceV1,  # support modelslim v1
                             ModelHookInterface,  # support spec ops for model in runner
                             IterSmoothInterface,  # support iter smooth
                             FlexSmoothQuantInterface,  # support flex smooth quant
                             ):
    def get_model_type(self) -> str:
        return self.model_type

    def get_model_pedigree(self) -> str:
        return 'deepseek_v3'

    def init_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        # load one more layer for mtp
        self.config.num_hidden_layers = 4
        # load model to cpu and cpu
        model = SafeGenerator.get_model_from_pretrained(model_path=str(self.model_path),
                                                        config=self.config,
                                                        trust_remote_code=True,
                                                        device_map="cpu",
                                                        torch_dtype="auto",
                                                        attn_implementation='eager')
        # # warp mtp into model
        # model = warp_mtp_model(self.config, model, str(self.model_path))
        return model

    def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> List[Any]:
        return self._get_tokenized_data(dataset, device)

    def generate_model_visit(self, model: nn.Module, transformer_blocks: Optional[List[Tuple[str, nn.Module]]] = None,
                             ) -> Generator[ProcessRequest, Any, None]:
        return generated_decoder_layer_visit_func(model, transformer_blocks)

    def generate_model_forward(self, model: nn.Module, inputs: Any,
                               ) -> Generator[ProcessRequest, Any, None]:
        def pack_request(name: str, request_args: Tuple, request_kwargs: Dict):
            module = get_module_by_name(model, submodule_key=name)
            return ProcessRequest(name, module, request_args, request_kwargs)

        transformer_blocks = [
            (name, module)
            for name, module in model.named_modules()
            if "decoderlayer" in module.__class__.__name__.lower()
        ]

        # 存储第一个transformer block的输入
        first_block_input: Optional[Tuple] = None

        def break_hook(module: nn.Module, hook_args: Tuple[Any, ...], hook_kwargs: Dict[str, Any]):
            nonlocal first_block_input
            first_block_input = (hook_args, hook_kwargs,)
            raise _TransformersForwardBreak()

        hooks = [transformer_blocks[0][1].register_forward_pre_hook(break_hook, with_kwargs=True, prepend=True)]

        # 执行一次前向传播以获取输入
        try:
            if isinstance(inputs, list) or isinstance(inputs, tuple):
                model(*inputs)
            elif isinstance(inputs, dict):
                model(**inputs)
            else:
                model(inputs)
        except _TransformersForwardBreak:
            pass
        except Exception as e:
            raise e
        finally:
            for hook in hooks:
                hook.remove()

        if first_block_input is None:
            raise InvalidModelError("Can't get first block input.", action="Please check the model and input")

        # 循环处理每个transformer block
        current_inputs = first_block_input

        if dist.is_initialized():
            dist.barrier()

        mtp_idx = self.config.num_hidden_layers

        args, kwargs = current_inputs
        for name, block in transformer_blocks[:mtp_idx]:
            output = yield ProcessRequest(name, block, args, kwargs)
            hidden_states = output[0]
            args = (hidden_states,)

        hidden_states = model.model.norm(*args)
        logits = model.lm_head.to('npu')(hidden_states.to('npu'))
        logits = logits.float()

        # ####################### MTP LAYER ######################
        # input_ids = inputs['input_ids'] if isinstance(inputs, dict) else inputs[0]
        # input_ids_mtp = remove_zero_and_shift(input_ids)
        # position_ids = torch.arange(
        #     0,
        #     input_ids_mtp.shape[-1],
        #     dtype=torch.long,
        #     device=input_ids.device,
        # ) + 1
        # position_ids = position_ids.unsqueeze(0)
        # logits[:, -1, :].argmax(dim=1)
        # input_ids_mtp[:, -1] = logits[:, -1, :].argmax(dim=1)

        # mtp_layer = model.model.layers[self.config.num_hidden_layers]
        # input_embeds_mtp = mtp_layer.embed_tokens.to('npu')(input_ids_mtp.to('npu'))
        # input_embeds_mtp = mtp_layer.enorm.to('npu')(input_embeds_mtp)
        # hidden_states_mtp = mtp_layer.hnorm.to('npu')(hidden_states.to('npu'))
        # hidden_states_mtp = torch.cat([input_embeds_mtp, hidden_states_mtp], dim=-1)
        # hidden_states_mtp = mtp_layer.eh_proj.to('npu')(hidden_states_mtp)

        # attention_mask = inputs['attention_mask'] if isinstance(inputs, dict) else inputs[1]

        # # transformers==4.48.2
        # from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

        # attention_mask_mtp = _prepare_4d_causal_attention_mask(
        #     attention_mask,
        #     (input_ids.shape[:2]),
        #     input_embeds_mtp,
        #     0,
        # )
        # _ = yield pack_request(f"model.layers.{mtp_idx}",
        #                        (hidden_states_mtp,),
        #                        {
        #                            "attention_mask": attention_mask_mtp,
        #                            "position_ids": position_ids,
        #                            "past_key_value": None,
        #                            "output_attentions": False,
        #                            "use_cache": False,
        #                        })

        # the remaining part just ignore
        return

    def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:
        def pre_forward_hook(module, args, kwargs):
            kwargs['use_cache'] = need_kv_cache
            return args, kwargs

        model.model.register_forward_pre_hook(pre_forward_hook, with_kwargs=True)

    def load_state_dict_hook(self, key: str, module: nn.Module) -> None:
        auto_convert_model_fp8_to_bf16(key, module, str(self.model_path))

    def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
        adapter_config = []
        for layer_idx in range(self.config.num_hidden_layers):
            # OKV_b融合的映射配置：o_proj -> kv_b_proj
            okv_b_mapping_config = MappingConfig(
                source=f"model.layers.{layer_idx}.self_attn.kv_b_proj",  # KV_b投影层
                targets=[f"model.layers.{layer_idx}.self_attn.o_proj"]  # 输出投影层
            )

            # Norm-Linear融合的映射配置1：q_a_proj, kv_a_proj_with_mqa -> input_layernorm
            norm_linear_mapping_config1 = MappingConfig(
                source=f"model.layers.{layer_idx}.input_layernorm",  # 第一个LayerNorm
                targets=[f"model.layers.{layer_idx}.self_attn.q_a_proj",
                         f"model.layers.{layer_idx}.self_attn.kv_a_proj_with_mqa"]  # 注意力层的Q_a,KV_a投影
            )

            # Norm-Linear融合的映射配置2：q_b_proj -> q_a_layernorm
            norm_linear_mapping_config2 = MappingConfig(
                source=f"model.layers.{layer_idx}.self_attn.q_a_layernorm",  # q_a_layernorm
                targets=[f"model.layers.{layer_idx}.self_attn.q_b_proj"]  # q_b投影
            )

            # 为当前layer添加4个配置
            adapter_config.extend([
                AdapterConfig(
                    subgraph_type="ov",
                    mapping=okv_b_mapping_config,
                    fusion=FusionConfig(
                        fusion_type="kv",
                        num_attention_heads=self.config.num_attention_heads,
                        num_key_value_heads=self.config.num_key_value_heads,
                        custom_config={
                            'qk_nope_head_dim': self.config.qk_nope_head_dim,
                            'v_head_dim': self.config.v_head_dim,
                        }
                    ),
                ),
                AdapterConfig(
                    subgraph_type="norm-linear",
                    mapping=norm_linear_mapping_config1
                ),
                AdapterConfig(
                    subgraph_type="norm-linear",
                    mapping=norm_linear_mapping_config2
                ),
            ])

        return adapter_config

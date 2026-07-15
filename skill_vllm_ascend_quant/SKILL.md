---
name: vllm-ascend-quant
description: 用于处理 vLLM Ascend 量化特性相关任务，包括量化方法适配分析、量化配置检查、量化算子行为验证等场景。
---

# vLLM Ascend 量化特性 skill

## 适用场景

- vLLM Ascend 量化方法（W8A8/W8A16/W4A8/W4A16/MXFP8/MXFP4 等）的适配分析。
- 量化配置（compressed-tensors / modelslim / ascendv1_saver）检查与解析。
- MRv2 执行路径下量化算子行为验证。
- 量化模型权重布局（NZ 格式等）相关问题定位。

## 工作流程

1. 明确量化方法类型与目标模型架构。
2. 检查 `vllm/config/vllm.py` 中 `_is_default_v2_model_runner_model` 的判定逻辑，确认模型是否走 MRv2。
3. 定位对应量化方法的 `LinearMethod` 实现（`vllm_ascend/quantization/methods/`），分析 `apply` 与 `process_weights_after_loading`。
4. 结合 Ascend 算子（如 `npu_weight_quant_batchmatmul`、`npu_format_cast`）验证执行路径。
5. 输出分析结论或修复建议。

## 参考文档

- 适配分析报告：`docs/analysis_reports/vllm_mrv2_quantization_adaptation.md`

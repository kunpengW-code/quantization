# vLLM Model Runner V2 量化特性适配分析报告

- 报告日期：2026-07-15
- 追踪 Issue：[#41286](https://github.com/vllm-project/vllm/issues/41286) Migration from Model Runner v1 to Model Runner v2
- 所属版本：vLLM v0.24.0（量化默认启用）→ v0.25.0（MRv2 成为 dense 模型默认路径）

## 1. 背景

Model Runner V2（MRv2）是 vLLM 下一代执行引擎，目标是替代 MRv1 成为所有 Dense 模型的默认执行路径。在迁移路线图中，**量化模型的默认启用**是关键的第 5 阶段（[5/N]），此前量化模型被显式排除在 MRv2 之外。

本报告梳理 MRv2 迁移过程中量化特性的适配 PR 与主要适配内容。

## 2. 迁移路线图总览

MRv1 → MRv2 的迁移分为多个阶段，量化适配位于第 5 阶段：

| 阶段 | PR | 标题 | 状态 |
|------|-----|------|------|
| 1 | [#39337](https://github.com/vllm-project/vllm/pull/39337) | Dense 模型支持（Qwen3-0.6B, opt-125m） | 已合并 |
| 2 | [#43458](https://github.com/vllm-project/vllm/pull/43458) | MoE 模型支持（DeepSeek-V2-lite） | 已合并 |
| 3 | [#42667](https://github.com/vllm-project/vllm/pull/42667) | Qwen + DSv2 MoE 迁移 [3/N] | 已合并 |
| 4 | [#44443](https://github.com/vllm-project/vllm/pull/44443) | 为所有 dense 模型默认启用 | 已合并 |
| **5** | **[#44446](https://github.com/vllm-project/vllm/pull/44446)** | **支持量化模型默认启用 [5/N]** | **已合并** |
| 6 | [#45461](https://github.com/vllm-project/vllm/pull/45461) | GraniteMoE 默认启用 | 已合并 |

## 3. 量化适配核心 PR

### 3.1 PR #44446 - Migration to support quantized model by default [5/N]

- 链接：https://github.com/vllm-project/vllm/pull/44446
- 作者：yewentao256
- 合并时间：2026-06-18
- 所属追踪 Issue：[#41286](https://github.com/vllm-project/vllm/issues/41286)

这是 MRv2 迁移路线图中**专门负责量化模型默认启用**的 PR。

### 3.2 主要适配内容

#### 核心改动：移除量化模型的 MRv2 阻断逻辑

在 `vllm/config/vllm.py` 的 `_is_default_v2_model_runner_model` 方法中，**删除了对量化模型的阻断检查**：

```python
# 适配前：量化模型被强制走 MRv1
if model_config.is_quantized:
    return False

# 适配后：移除上述 3 行，量化模型遵循与其他 dense 模型相同的判定逻辑
```

移除后，量化模型即可在架构属于 `DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES` 或为非 MoE dense 模型时，默认走 MRv2 路径。

#### 测试用例更新

`tests/test_config.py` 中，量化模型的期望结果从 `False` 改为 `True`，验证量化模型现在默认使用 MRv2。

### 3.3 改动规模

- 文件数：2（`vllm/config/vllm.py`、`tests/test_config.py`）
- 代码变更：+1 / -4 行

改动小的原因：前 4 个阶段已构建完整的 MRv2 基础设施（执行路径、调度、compile 等），量化适配本质上是"打开最后的开关"。

## 4. 前置依赖分析

### 阶段 4（PR #44443）的铺垫作用

PR #44443「Enable by default for all dense models」已经为所有 dense 模型默认启用了 MRv2，但**保留了 `is_quantized` 检查**作为最后的阻断点。其关键改动包括：

1. 调整 `DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES`，将其作为"已知不支持的架构"黑名单，而非白名单。
2. 新增 `is_hybrid`、`is_attention_free` 的排除检查。
3. 非 MoE 的 dense 模型默认走 MRv2。

PR #44446 正是在此基础上移除 `is_quantized` 这最后的阻断点。

## 5. 版本归属与影响

- **v0.24.0**（2026-07-02 发布）：MRv2 支持量化模型默认启用（PR #44446），Release Notes 明确提到："MRv2 now supports quantized models by default (#44446)"。
- **v0.25.0**（2026-07-11 发布）：MRv2 正式成为所有 Dense 模型的默认执行路径，PagedAttention 被彻底移除（PR #47361），标志着 MRv2 迁移基本完成。

## 6. 后续关注点

1. **Ascend 侧适配**：vLLM Ascend 的量化方法（W8A8/W8A16/W4A8/MXFP8/MXFP4 等）需在 MRv2 路径下验证正确性与性能，重点关注：
   - `AscendW8A8LinearMethod`、`AscendW8A16LinearMethod` 等 `apply` / `process_weights_after_loading` 在 MRv2 执行路径下的行为。
   - MRv2 的 cuda graph / torch compile 路径与 Ascend 算子（`npu_weight_quant_batchmatmul` 等）的兼容性。
2. **量化 + MoE**：当前 MRv2 对 MoE 量化模型的支持需结合阶段 2/6 的工作继续验证。
3. **NZ 格式与 weight 重新布局**：MRv2 下 `process_weights_after_loading` 的 NZ 格式转换（`VLLM_ASCEND_ENABLE_NZ`）需回归测试。

## 7. 结论

vLLM MRv2 对量化特性的适配**本质上是单一 PR（#44446）**，核心内容是**移除 `vllm/config/vllm.py` 中对 `is_quantized` 模型的 MRv2 阻断检查**，让量化模型能够与 BF16 模型一样默认走 MRv2 执行路径。

该适配建立在 MRv1 → MRv2 迁移路线图前 4 个阶段的基础设施之上，是迁移路线图的第 5 阶段。合并后进入 v0.24.0，为 v0.25.0 中 MRv2 成为所有 Dense 模型默认路径奠定了量化维度的闭环。

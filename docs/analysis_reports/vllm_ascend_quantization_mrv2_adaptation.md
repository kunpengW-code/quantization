# vllm-ascend 量化特性适配 ModelRunnerV2 分析报告

## 1. 背景与目标

vllm-ascend 正在支持 ModelRunnerV2（下一代的 vLLM model runner，当前为 Experimental 状态，PR [#5210](https://github.com/vllm-project/vllm-ascend/pull/5210)）。ModelRunnerV2 重构了输入批次（`AscendInputBatch`）、请求状态（`AscendRequestState`）、KV cache 初始化以及图捕获等核心路径。本文从量化框架、linear 量化、MoE 量化、KV cache 量化四个维度，分析适配 ModelRunnerV2 所需的工作。

### 关键现状
- V2 runner 入口：[vllm_ascend/worker/v2/model_runner.py](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/worker/v2/model_runner.py) 中的 `NPUModelRunner(GPUModelRunner)`。
- V2 已显式声明的 Gap（见 [vllm_ascend/worker/v2/README.md](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/worker/v2/README.md)）：`set_cos_and_sin`、`_allocate_kv_cache`/`_reshape_kv_cache`、`torch_npu_graph_wrapper`，其中后两项直接与 KV cache 量化相关。
- V2 明确不支持的特性：Context parallelism、dynamic EPLB（在 `__init__` 中直接 `raise NotImplementedError`）。

---

## 2. 量化框架适配

### 2.1 当前框架结构

vllm-ascend 的量化框架在 [vllm_ascend/quantization/__init__.py](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/quantization/__init__.py) 中注册三类 Config（懒加载避免循环导入）：

| Config 类 | 文件 | 适用场景 |
|---|---|---|
| `AscendModelSlimConfig` | `modelslim_config.py` | 华为 ModelSlim 工具产出的量化模型 |
| `AscendCompressedTensorsConfig` | `compressed_tensors_config.py` | LLM-Compressor / compressed-tensors 格式 |
| `AscendFp8Config` | `fp8_config.py` | FP8 量化（新版新增） |

三者均继承 vLLM 的 `QuantizationConfig`，通过 `get_quant_method(layer, prefix, tid2eid=None)` 将层映射为 `AscendLinearMethod` / `AscendFusedMoEMethod`（见 [method_adapters.py](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/quantization/method_adapters.py)）。

### 2.2 适配 V2 需要做的工作

**框架层基本可复用**，因为 `get_quant_method` 是基于 `nn.Module` 类型与 `prefix` 的分发，与 runner 版本解耦。

---

## 3. Linear 量化适配

### 3.1 现状

主要 scheme 在 [vllm_ascend/quantization/methods/](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/quantization/methods/) 下：
- `w8a8_dynamic.py`：`AscendW8A8DynamicLinearMethod`，使用 `torch_npu.npu_dynamic_quant` + `torch_npu.npu_quant_matmul`。
- W4A4_MXFP4、W8A16：通过 scheme 创建。

### 3.2 适配 V2 需要做的工作

`AscendLinearMethod.apply(layer, x, bias, tp_rank)` 签名与 runner 无强耦合，linear 量化本身可跨 V1/V2 复用。

---

## 4. MoE 量化适配

### 4.1 现状

- Scheme：`AscendW8A8DynamicFusedMoEMethod`（[w8a8_dynamic.py](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/quantization/methods/w8a8_dynamic.py)），通过 `_EXTRA_CTX.moe_comm_method` 调用 fused experts。
- V2 runner 在 `__init__` 中调用 `set_mc2_tokens_capacity` 与 `set_mc2_mask` 预置 MoE 通信 buffer。

### 4.2 适配 V2 需要做的工作

量化接口与 runner 无强耦合，MoE 量化本身可跨 V1/V2 复用。

---

## 5. KV Cache 量化适配

### 5.1 现状

两种 KV 量化方案，均在 [vllm_ascend/worker/v2/attn_utils.py](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/worker/v2/attn_utils.py) 中集成：

| 方案 | 机制 | 关键调用 |
|---|---|---|
| **C8（INT8）** | 注意力后端 `AscendC8AttentionBackendImpl` 对 K/V 做 INT8 量化 | `attention_v1.py` |
| **FAQuant（float8/INT8）** | 量化 Config 提供 split factor 与 dtype | `enable_fa_quant()` → `get_kv_quant_split_factor` / `get_kv_quant_dtype` |
| **SFA C8（float8/INT8）** | enable_sparse_c8 开启 | `sfa_v1.py` |

### 5.2 适配 V2 需要做的工作

KV cache 量化是 V2 适配中**工作量最大、风险最高**的部分，因为 V2 已将 `_allocate_kv_cache` / `_reshape_kv_cache` 列为待移除的临时方案。

1. **KV cache 分配的量化感知**：
   - `_allocate_kv_cache`（[attn_utils.py:261](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/worker/v2/attn_utils.py#L261)）当前以 `torch.int8` 分配 K/V buffer，再在 `_reshape_kv_cache` 中按 spec 的 dtype reshape。
   - C8 量化要求 K/V 以 INT8 存储但按量化语义访问。V2 的 `AscendMLAAttentionSpec`（[attn_utils.py:70](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/worker/v2/attn_utils.py#L70)）在 `fa_quant_layer=True` 时将 `head_size` 设为 `head_size + qk_rope_head_dim` 并把 `cache_dtype_str` 置 None，这一特殊路径需在 C8 下单独验证。

2. **FAQuant 的 split factor 计算**：
   - [attn_utils.py:304](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/worker/v2/attn_utils.py#L304) 调用 `quant_config.get_kv_quant_split_factor(example_layer_name, kv_head_dim_list)`，按 K/V 维度分别计算 tensor split factor，再据此切分 raw buffer。
   - 需保证：FAQuant 开启时 K 与 V 的 split factor 不同（float8 量化后维度膨胀/收缩），V2 的 reshape 与 `AscendMLAAttentionSpec` 的 `head_size` 计算必须与 split factor 自洽，否则 num_blocks 计算错误。

3. **FAQuant 的 dtype 下发**：
   - [attn_utils.py:417](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/worker/v2/attn_utils.py#L417) 与 [attn_utils.py:496](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/worker/v2/attn_utils.py#L496) 两处调用 `get_kv_quant_dtype`，分别在 `_reshape_kv_cache` 与 `_reshape_kv_cache_v2` 中。
   - 存在两个 reshape 函数（`_reshape_kv_cache` 与 `_reshape_kv_cache_v2`），说明 V2 正在重构 KV cache 路径。量化适配需同步迁移到新函数，避免双轨维护。

4. **PD 分离部署对齐**：
   - `_allocate_kv_cache` 在 `kv_transfer_config` 非 None 时对 K/V buffer 额外分配 `alignment=2MB` 并 `_align_memory` 对齐，以支持 Mooncake PD 分离。
   - 量化（尤其 FAQuant）下 K/V 实际存储 dtype 与 buffer 起始 int8 不同，2M 对齐的偏移计算必须基于量化后的实际存储布局，否则 PD 传输会错位。

5. **MLA 注意力下的 FAQuant head_size**：
   - MLA 将 K 拆为 nope 与 rope 两部分。FAQuant 下 `head_size = attn_module.head_size + attn_module.qk_rope_head_dim`，且 `cache_dtype_str=None`（表示走 impl 内部 dtype）。
   - V2 的 `AscendMLAAttentionSpec` 构造需保证该 head_size 与实际 KV cache tensor 的物理布局一致，否则 attention 后端会越界访问。

6. **ACL Graph 下的 KV 量化参数更新**：V1 的 full graph 通过 `attention_v1.py` / `mla_v1.py` 的 `update_graph_params` 在 replay 前 host-side 更新 attention 参数。V2 若启用 FULL graph，需验证量化 KV cache 的 scale 参数也在更新链路中。

---

## 6. 测试与验证矩阵

| 维度 | 测试项 |
|---|---|
| **量化方法** | W8A8_DYNAMIC、W4A8、W4A4_MXFP4、FP8 |
| **Config 来源** | ModelSlim、compressed-tensors |
| **KV 量化** | 无、C8、FAQuant |
| **Runner** | V1（基线）、V2 |
| **图模式** | NONE、PIECEWISE、FULL_DECODE_ONLY |
| **拓扑** | 单卡、多卡 TP、PD 分离 |
| **模型结构** | 普通 Attention、MLA、DSA|

---

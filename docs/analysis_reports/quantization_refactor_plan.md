# vllm-ascend 量化模块重构方案

## 一、现状分析

### 1.1 当前目录结构

```
vllm_ascend/
├── quantization/
│   ├── __init__.py                    # 懒加载3个Config类
│   ├── utils.py                       # 自动检测量化方法
│   ├── quant_type.py                  # QuantType枚举(9个值)
│   ├── quant_parser.py               # MXFP量化类型映射
│   ├── modelslim_config.py           # AscendModelSlimConfig (~1004行)
│   ├── method_adapters.py            # 4个适配器类
│   ├── fp8_config.py                 # AscendFp8Config (~130行)
│   ├── compressed_tensors_config.py  # AscendCompressedTensorsConfig (~440行)
│   └── methods/
│       ├── base.py                   # 3个抽象基类 (298行)
│       ├── registry.py               # 注册表机制
│       ├── w8a8_static.py, w8a8_dynamic.py, w8a8fp8_dynamic.py
│       ├── w8a8_pdmix.py, w8a8_mxfp8.py, w8a16.py
│       ├── w4a8.py (828行), w4a8_mxfp4.py
│       ├── w4a4_mxfp4.py, w4a4_mxfp4_flatquant.py, w4a4_flatquant.py
│       ├── w4a4_laos_dynamic.py, w4a16.py, w4a16_mxfp4.py
│       ├── kv_c8.py, fp8.py
│       └── __init__.py
├── ops/fused_moe/                    # MoE计算路径(含大量量化分支)
│   ├── moe_mlp.py                    # quant_apply_mlp (322行, 12+分支)
│   ├── moe_stage_params.py           # MoEQuantParams
│   ├── prepare_finalize.py           # dispatch前量化
│   ├── token_dispatcher.py           # dispatch量化模式选择
│   └── ...
├── device/
│   └── mxfp_compat.py               # MXFP dtype兼容层(91行)
├── attention/
│   ├── attention_v1.py               # C8 KVCache注意力后端 (~180行)
│   ├── sfa_v1.py                     # Sparse C8 KVCache
│   └── dsa_v1.py                     # DSA KV量化
├── ascend_config.py                  # enable_sparse_c8等参数
└── _310p/quantization/               # 310P硬件特化(独立注册表)
```

### 1.2 核心问题总结

| 问题域 | 问题描述 | 严重程度 |
|--------|---------|---------|
| methods/目录膨胀 | 18个量化方法文件平铺，无分类；3种Config散落在quantization/根目录 | 高 |
| 代码重复 | MoE apply()模板代码在9个类中复制粘贴；`_is_fused_moe_layer()`定义3次 | 高 |
| fused_moe量化分支复杂 | `quant_apply_mlp` 322行、12+分支，无法将分支与量化类型关联 | 高 |
| KVCache量化参数分散 | 5种KVCache量化机制、30+参数名散布在10+文件中 | 高 |
| 量化代码离散 | `mxfp_compat.py`在device/、C8注意力后端的实现在attention/ | 中 |

---

## 二、重构方案

### 2.1 quantization/ 目录重构：细分 + Config独立 + 消除冗余

#### 2.1.1 methods/ 目录细分

**现状**: 18个量化方法文件平铺在 `methods/` 下，随着新量化方式增加会越来越乱。

**方案**: 按量化精度族（weight-bit x activation-bit）分子目录：

```
quantization/
└── methods/
    ├── __init__.py
    ├── base.py                       # 抽象基类(保留)
    ├── registry.py                   # 注册表(保留)
    ├── common.py                     # [新增] 提取公共逻辑
    ├── w8a8/                         # W8A8族
    │   ├── __init__.py
    │   ├── static.py                 # <- w8a8_static.py
    │   ├── dynamic.py                # <- w8a8_dynamic.py
    │   ├── fp8_dynamic.py            # <- w8a8fp8_dynamic.py
    │   ├── pdmix.py                  # <- w8a8_pdmix.py
    │   └── mxfp8.py                  # <- w8a8_mxfp8.py + ds_v4
    ├── w8a16/
    │   ├── __init__.py
    │   └── w8a16.py                  # <- w8a16.py
    ├── w4a8/
    │   ├── __init__.py
    │   ├── dynamic.py                # <- w4a8.py  
    │   └── mxfp4.py                  # <- w4a8_mxfp4.py + ds_v4
    ├── w4a4/
    │   ├── __init__.py
    │   ├── mxfp4.py                  # <- w4a4_mxfp4.py
    │   ├── mxfp4_flatquant.py        # <- w4a4_mxfp4_flatquant.py
    │   ├── flatquant.py              # <- w4a4_flatquant.py
    │   └── laos_dynamic.py           # <- w4a4_laos_dynamic.py
    ├── w4a16/
    │   ├── __init__.py
    │   ├── w4a16.py                  # <- w4a16.py
    │   └── mxfp4.py                  # <- w4a16_mxfp4.py
    ├── attention/
    │   ├── __init__.py
    │   ├── kv_c8.py                  # <- kv_c8.py
    │   └── fa_quant.py              # [可选] 从kv_c8.py拆分FA量化
```

**设计原则**:
- 同一精度族的Linear和MoE Scheme放在同一文件中（当前已是如此），因为它们的参数和权重处理高度相关
- 每个子目录有独立的 `__init__.py`，负责触发注册（import即注册）
- `methods/__init__.py` 只需 import 各子目录的 `__init__`，确保所有注册完成

#### 2.1.2 Config 目录独立

**现状**: 3种Config类（`modelslim_config.py`、`compressed_tensors_config.py`、`fp8_config.py`）散落在 `quantization/` 根目录，与工具类文件混杂。

**方案**:

```
quantization/
├── __init__.py
├── configs/                          # [新增] Config独立目录
│   ├── __init__.py                   # 统一导出3个Config
│   ├── modelslim.py                  # <- modelslim_config.py
│   ├── compressed_tensors.py         # <- compressed_tensors_config.py
│   └── fp8.py                        # <- fp8_config.py
├── utils.py
├── quant_type.py
├── quant_parser.py
├── method_adapters.py
└── methods/
```

**收益**: Config和Scheme（methods/）在目录层面清晰分离，职责一目了然。

#### 2.1.3 消除代码重复

**问题1: MoE `apply()` 模板代码重复（9个类）**

当前每个MoE Scheme的 `apply()` 都包含以下~30行重复代码：
```python
num_shared_experts = getattr(layer, "n_shared_experts", 0)
num_logical_experts = get_moe_num_logical_experts(...)
assert router_logits.shape[1] == num_logical_experts
topk_weights, topk_ids = select_experts(...)
if enable_force_load_balance: ...
moe_comm_method = _EXTRA_CTX.moe_comm_method
return moe_comm_method.fused_experts(build_fused_experts_input(...))
```

**方案**: 在 `methods/base.py` 的 `AscendMoEScheme` 中提供模板方法：

```python
class AscendMoEScheme(ABC):
    def apply(self, layer, x, router_logits, ...):
        # 1. 公共路由逻辑
        topk_weights, topk_ids = self._route_experts(layer, x, router_logits, ...)
        # 2. 子类提供权重收集策略
        weight_bundle = self._collect_weights(layer, ...)
        # 3. 公共调度逻辑
        return self._dispatch_and_compute(
            x, topk_weights, topk_ids, weight_bundle, ...)

    @abstractmethod
    def _collect_weights(self, layer, ...) -> MoEWeightBundle:
        """子类实现：收集并组织权重张量"""
        ...
```

**问题2: `_is_fused_moe_layer()` 定义3次**

**方案**: 提取到 `quantization/utils.py` 作为公共函数，3个Config统一引用。

**问题3: MoE `__init__()` 初始化模板重复**

**方案**: 在 `AscendMoEScheme.__init__()` 基类中统一处理 `group_size`、`use_aclgraph`、`dynamic_eplb` 等公共属性。

**问题4: `get_decompose_dim()` 重复且不一致**

**方案**: 提取到 `methods/common.py`，统一实现（保留 `MAX_SUPPORT_DIM` 保护逻辑）。

**问题5: MXFP scale reshape 模式重复**

**方案**: 在 `methods/common.py` 中提供 `reshape_mxfp_scale()` 工具函数。

**问题6: `packed_modules_model_mapping` 字典过大（~260行）**  --已有PR

**方案**: 按模型族分组定义，使用组合而非单一大字典：

```python
_DEEPSEEK_PACKED = {"gate_up_proj": [...], "experts": [...]}
_LLAMA_PACKED = {"gate_up_proj": [...]}
_PACKED_MODULES_MAPPING = {
    "deepseek_v2": _DEEPSEEK_PACKED,
    "deepseek_v3": _DEEPSEEK_PACKED,  # 复用
    "llama": _LLAMA_PACKED,
    ...
}
```



#### 2.1.4 FP8 config重构

继承vllm社区类，函数尽量复用

#### 2.1.5 量化方法日落计划

1.w4a8 linear量化，当前仅qwen3-32B跑通，性能不好，apply实际是w4a16，与权重格式不一致

2.w4a8 moe量化 per-group/per-group pack int8 性能不好，当前都是用per-channel，可以日落

---

### 2.2 ops/fused_moe 量化代码重构：策略模式解耦

#### 2.2.1 核心问题

`quant_apply_mlp`（`moe_mlp.py:88-410`）是一个322行的巨型函数，包含12+执行路径。分支条件混合了：
- 量化类型（W8A8/W4A8/W8A8MXFP/W4A16等）
- 通信方式（MC2/AllGather/AlltoAll）
- 算子融合策略（custom_gmm_swiglu/gmm_swiglu_quant_fusion）
- 激活函数类型（SwiGLU/GELU/SWIGLUSTEP）

阅读代码时无法将某个分支与具体的量化类型关联起来。

#### 2.2.2 重构方案：引入 MlpComputeStrategy 策略层

**核心思路**: 将 `quant_apply_mlp` 的分支逻辑从"按条件判断"重构为"按策略分发"。

```
quantization/
└── moe_compute/                      # [新增] MoE量化计算策略
    ├── __init__.py
    ├── base.py                       # MlpComputeStrategy 抽象基类
    ├── registry.py                   # 策略注册表
    ├── int_quant.py                  # W8A8/W4A8 整数量化策略
    ├── mxfp.py                       # W8A8MXFP/W4A8MXFP/W4A4MXFP MXFP策略
    ├── antiquant.py                  # W4A16 反量化策略
    └── shared_experts.py             # 共享专家量化计算
```

**策略基类设计**:

```python
class MlpComputeStrategy(ABC):
    """MoE MLP计算策略基类"""

    @abstractmethod
    def supports(self, quant_type: QuantType, comm_type: MoECommType) -> bool:
        """判断是否支持给定的量化类型+通信方式组合"""
        ...

    @abstractmethod
    def gate_up_proj(self, hidden_states, w1, w1_scale, group_list, ...) -> torch.Tensor:
        """第一阶段：gate_up投影 + 激活"""
        ...

    @abstractmethod
    def down_proj(self, hidden_states, w2, w2_scale, group_list, ...) -> torch.Tensor:
        """第二阶段：down投影"""
        ...
```

**策略实现示例**:

```python
class IntQuantMC2Strategy(MlpComputeStrategy):
    """W8A8/W4A8 整数量化 + MC2通信策略"""

    def supports(self, quant_type, comm_type):
        return quant_type in (QuantType.W8A8, QuantType.W4A8) and comm_type == MoECommType.MC2

    def gate_up_proj(self, hidden_states, w1, w1_scale, group_list, ...):
        # MC2路径: 优先使用融合算子 grouped_matmul_swiglu_quant
        if self._use_fusion:
            return DeviceOperator.npu_grouped_matmul_swiglu_quant(...)
        else:
            # 回退: npu_grouped_matmul -> npu_dequant_swiglu_quant
            ...
```

**分发机制**:

```python
# moe_mlp.py 中的 unified_apply_mlp 简化为:
def unified_apply_mlp(*, mlp_compute_input: MoEMlpComputeInput):
    if not mlp_compute_input.quant.is_quant:
        return unquant_apply_mlp(...)

    strategy = get_mlp_strategy(
        mlp_compute_input.quant.quant_type,
        mlp_compute_input.comm_type
    )
    hidden = strategy.gate_up_proj(...)
    return strategy.down_proj(hidden, ...)
```

#### 2.2.3 收益

| 方面 | 重构前 | 重构后 |
|------|--------|--------|
| 代码可读性 | 322行巨型函数，无法知道某分支对应哪种量化 | 每个策略类对应明确的量化类型族 |
| 可维护性 | 新增量化类型需修改巨型函数的if-else链 | 新增一个Strategy类，注册即可 |
| 可测试性 | 难以单独测试某个量化路径 | 每个Strategy可独立单元测试 |
| 与quantization/的关系 | 量化逻辑分散在ops/和quantization/两处 | MoE计算策略收编到quantization/moe_compute/ |

#### 2.2.4 渐进式迁移路径

考虑到 `quant_apply_mlp` 是核心热路径，建议分步迁移：

1. **Phase 1**: 在 `quantization/moe_compute/` 中创建策略框架和基类
2. **Phase 2**: 逐个提取分支到策略类（先从最简单的W4A16 antiquant路径开始）
3. **Phase 3**: `quant_apply_mlp` 退化为纯分发器，调用策略注册表
4. **Phase 4**: 将 `moe_compute/` 的导入关系从 `ops/fused_moe/` 反转（fused_moe依赖quantization，而非反过来）

---

### 2.3 KVCache 量化统一收编

#### 2.3.1 核心问题

当前存在5种KVCache量化机制，参数名和控制逻辑高度分散：

| 机制 | 主开关 | 定义位置 | 运行时逻辑位置 |
|------|--------|---------|--------------|
| FAKQuant (MLA KV量化) | `enable_fa_quant` | `modelslim_config.py` | `kv_c8.py`, `mla_v1.py` |
| C8 KVCache (Dense INT8) | `enable_c8_quant` | `modelslim_config.py` | `attention_v1.py` (~180行) |
| Sparse C8 (DSA稀疏+C8) | `enable_sparse_c8` | `ascend_config.py` | `sfa_v1.py`, `dsa_v1.py` |
| Indexer量化 (SFA索引器) | `enable_indexer_quant` | `modelslim_config.py` | `kv_c8.py`, `sfa_v1.py` |
| KV Transfer量化 (PD分离) | `enable_kv_quant` | `mooncake_connector.py` | `mooncake_connector.py` |



问题：

- 权重加载（静态量化） C8开关当前参数名称较多，分散在各文件内，quantization对外（MRV1/MRV2等）尽量归一
- 动态量化 通过启动命令开启：--kv-cache-dtype fp8 (社区归一)
- kvcache初始化  --绑定modelrunner ("\_allocate"/"_reshape")  -> 抽象到 backend.get_kv_cache_shape
- attention backend后端内，当前C8实现有无可优化的地方

#### 2.3.2 重构方案

**方案A: 引入 KVCacheQuantManager 统一管理**

```
quantization/
└── kvcache/                          # [新增] KVCache量化统一目录
    ├── __init__.py
    ├── manager.py                    # KVCacheQuantManager: 统一开关/查询接口
    ├── configs.py                    # KVCacheQuantConfig: 统一配置数据类
    ├── schemes/
    │   ├── __init__.py
    │   ├── fa_quant.py               # <- kv_c8.py 中 FAKQuant 部分
    │   ├── c8_dense.py               # <- kv_c8.py 中 C8 部分
    │   ├── c8_sparse.py              # <- 从 ascend_config.py + sfa_v1.py 提取
    │   └── indexer.py                # <- kv_c8.py 中 Indexer 部分
    └── attention_backend/
        ├── __init__.py
        └── c8_backend.py             # <- attention_v1.py 中 AscendC8AttentionBackendImpl
```

**KVCacheQuantManager 设计**:

```python
class KVCacheQuantManager:
    """统一管理所有KVCache量化机制的开关和查询"""

    def __init__(self, quant_description: dict):
        self._fa_quant_enabled = bool(quant_description.get("fa_quant_type"))
        self._c8_enabled = quant_description.get("kv_cache_type") == "C8"
        self._indexer_enabled = bool(quant_description.get("indexer_quant_type"))
        self._sparse_c8_enabled = False  # 由ascend_config设置
        self._fa_quant_layers: list[int] = []
        self._c8_quant_layers: list[int] = []
        self._indexer_quant_layers: list[int] = []

    @property
    def is_any_kv_quant_enabled(self) -> bool: ...

    def is_c8_layer(self, layer_idx: int) -> bool: ...
    def is_fa_quant_layer(self, layer_idx: int) -> bool: ...
    def is_indexer_quant_layer(self, layer_idx: int) -> bool: ...

    def get_kv_cache_dtype(self, layer_idx: int) -> torch.dtype: ...
    def get_kv_cache_split_factor(self, layer_idx: int) -> int: ...
```

**收益**:
- 所有KVCache量化的开关判断统一到一个入口
- 外部模块只需调用 `kv_manager.is_c8_layer(idx)` 而非直接读取分散的属性
- 新增KVCache量化方式只需在 `schemes/` 下新增文件 + 在Manager中注册

**方案B（轻量级替代）: 如果不想大动目录结构**

在现有 `quantization/` 下新增 `kvcache_config.py`，将分散在 `modelslim_config.py` 和 `ascend_config.py` 中的KVCache量化参数统一收拢到一个数据类中，对外提供统一查询接口。不做目录级别的拆分。

**推荐**: 方案A，因为KVCache量化已有5种机制且还在增长，值得独立管理。

---

### 2.4 离散量化代码收编

#### 2.4.1 需要收编的文件

| 当前位置 | 内容 | 建议去向 |
|---------|------|---------|
| `device/mxfp_compat.py` | MXFP dtype兼容层、ensure函数 | `quantization/compat/mxfp.py` |
| `attention/attention_v1.py` 中的 `AscendC8AttentionBackendImpl` | C8 KVCache注意力后端 (~180行) | `quantization/kvcache/attention_backend/c8_backend.py` |
| `attention/sfa_v1.py` 中的Sparse C8逻辑 | 稀疏C8 KVCache处理 | 保留在attention/（与注意力计算强耦合），但通过KVCacheQuantManager查询 |
| `attention/dsa_v1.py` 中的KV量化scatter | DSA KV量化 | 保留在attention/（与注意力计算强耦合），通过Manager查询 |
| `ascend_config.py` 中的 `enable_sparse_c8` 等 | Sparse C8配置 | 配置解析保留在ascend_config，运行时查询委托给KVCacheQuantManager |

#### 2.4.2 收编原则

**收编标准**: 如果代码的核心职责是"量化"，则收编到 `quantization/`；如果代码的核心职责是"注意力计算"但涉及量化分支，则保留原位但通过quantization模块的接口交互。

```
收编:
  device/mxfp_compat.py → quantization/compat/mxfp.py
  attention_v1.py::AscendC8AttentionBackendImpl → quantization/kvcache/attention_backend/

保留原位（通过接口解耦）:
  attention/sfa_v1.py 中的Sparse C8分支 → 通过 KVCacheQuantManager 查询
  attention/dsa_v1.py 中的KV量化 → 通过 KVCacheQuantManager 查询
  ascend_config.py 中的Sparse C8配置 → 委托给 KVCacheQuantManager
```

#### 2.4.3 兼容层设计

`mxfp_compat.py` 收编后，在 `quantization/compat/` 下提供统一的硬件兼容检查：

```
quantization/
└── compat/                           # [新增] 硬件兼容层
    ├── __init__.py
    └── mxfp.py                       # <- device/mxfp_compat.py
```

原 `device/mxfp_compat.py` 保留为兼容性re-export（避免破坏已有import路径），后续逐步迁移引用。

---

## 三、关键设计决策点

### 5.1 `_310p/quantization/` 是否同步重构

- **不同步**: 310P保持现有结构 —— 310P是硬件特化层，Scheme数量少（4个），当前结构已够用

### 5.2  部分量化方式日落

- **w4a8 linear **: 确认是否有现网局点，待日落
- **w4a8 moe**: per-group的两版量化格式日落

### 5.3  mxfp_compat.py 

- **待确定**：是否仅为了防止使用错误的PTA，删除或者收编到quantization目录



---

## 四、附录：当前代码问题详细清单

### A. 代码重复清单

| 编号 | 重复内容 | 涉及文件 | 重复行数 |
|------|---------|---------|---------|
| D1 | MoE apply() 路由+调度模板 | 9个MoE Scheme类 | ~30行/处 |
| D2 | `_is_fused_moe_layer()` | modelslim_config, fp8_config, compressed_tensors_config | 3行/处 |
| D3 | `create_scheme_for_layer()` | modelslim_config, fp8_config | ~20行/处 |
| D4 | MoE `__init__()` 公共属性初始化 | 5个MoE Scheme类 | ~8行/处 |
| D5 | `get_decompose_dim()` | w4a4_flatquant, w4a4_mxfp4_flatquant | ~15行/处 |
| D6 | MXFP scale reshape模式 | w8a8_mxfp8, w4a4_mxfp4, w4a8_mxfp4, fp8 | ~10行/处 |
| D7 | MC2 group init try/except | w8a8_dynamic, w4a8 | ~12行/处 |

### B. 代码质量问题清单

| 编号 | 问题 | 位置 | 类型 |
|------|------|------|------|
| Q1 | Attention Scheme未继承基类 | kv_c8.py (FAQuant, SFAQuant) | 类型契约违反 |
| Q2 | 类级别可变状态（input_size） | w4a4_flatquant.py:89 | 潜在Bug |
| Q3 | import-time全局副作用 | fp8_config.py:30, compressed_tensors_config.py:51 | 架构风险 |
| Q4 | 未使用的类变量 model_dtype | w8a8_mxfp8, w4a4_mxfp4, fp8 | 死代码 |
| Q5 | `AscendC8KVCacheAttentionMethod` 未注册 | kv_c8.py | 一致性问题 |
| Q6 | QuantType枚举与Registry字符串不匹配 | quant_type.py vs registry.py | 维护风险 |
| Q7 | `apply()` 签名20+参数 | methods/base.py:234-260 | 接口膨胀 |
| Q8 | `utils.py` 中 `isinstance(quant_cfg, dict)` 重复检查 | utils.py:126-133 | 冗余代码 |
| Q9 | `modelslim_config.py` get_quant_method() 70行嵌套条件 | modelslim_config.py:657-726 | 复杂度 |

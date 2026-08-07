# RFC：面向树外后端的可扩展 `--kv-cache-dtype`

- **状态：** 草案
- **讨论：** vLLM / vLLM-Ascend
- **最后更新：** 2026-08-07

## 1. 摘要

允许树外（OOT）平台后端（如 **vLLM-Ascend**）注册自定义 KV cache 量化类型，使其可以直接通过现有的 `--kv-cache-dtype` 参数选择，例如 `--kv-cache-dtype sfa-c8`，而不再依赖不透明的 `--additional-config` 旁路通道。

所提机制复刻了 vLLM 仓内已验证的 **量化方法注册器**（`register_quantization_config` / `QUANTIZATION_METHODS`）和 **attention backend 注册器**（`register_backend` / `AttentionBackendEnum.CUSTOM`）。

## 2. 动机

### 2.1 当前 vLLM-Ascend 的行为

vLLM-Ascend 提供了若干设备相关的 KV cache 量化方法，其中两个只能通过自由格式的 `--additional-config` 字典触达：

| 开关（位于 `additional_config`） | 含义 |
|---------------------------------|---------|
| `enable_sparse_sfa_c8`           | DSA 模型（DeepSeek-V3.2、GLM-5/5.2）上 Sparse Flash Attention 的打包 C8（INT8/FP8）KV cache |
| `enable_sparse_li_c8`            | Lightning Indexer 的 C8 key + scale cache（按层） |
| `c8_enable_reshape_optim`        | LI C8 cache 写入的 reshape 优化 |

这些在 `vllm_ascend/ascend_config.py:249-253` 读取，并由 `model_uses_sfa_sparse()` 门控。实际存储 dtype **与设备相关**（Ascend A2/A3 上为 `torch.int8`，A5 上为 `torch.float8_e4m3fn`）——见 `vllm_ascend/attention/sfa_v1.py:614-622` 和 `vllm_ascend/worker/model_runner_v1.py:378-386`（逻辑重复）。

还存在第二条独立的 C8 路径，面向稠密 GQA 模型，由 checkpoint 的 `quant_description["kv_cache_type"] == "C8"` 字符串驱动，而非任何 vLLM 参数（`vllm_ascend/quantization/modelslim_config.py:1002`）。

这些名称**没有集中枚举**——原始字符串字面量散布在 `sfa_v1.py`、`model_runner_v1.py`、`device_op.py`、`sfa_cp.py`、`sfa_kv_offload.py`、`kv_cache_interface.py` 中约 30 处。

### 2.2 为什么这是个问题

1. **UX 不一致。** 其他所有后端都通过 `--kv-cache-dtype` 暴露 KV cache 量化（fp8、fp8_e5m2 等）。Ascend 却把用户逼进一个 JSON `--additional-config` 大包里，仅在 vLLM-Ascend 文档（`docs/source/user_guide/configuration/additional_config.md:81-83`）中有说明。
2. **不透明 / 无类型。** `additional_config` 是个 `dict[str, Any]`，仅用于编译时参与 hash（`vllm/config/vllm.py:352-358`、`:473-481`）。`enable_sparse_sfa_c8` 与 `enable_sparse_sFa_c8` 这类拼写错误会静默回退为 `False`。既没有 `--help` 条目，也没有 Pydantic 校验。
3. **无法参与 vLLM 的 KV cache 分发。** vLLM 本已把 `--kv-cache-dtype` 路由进 attention backend（`kv_cache_dtype` 构造参数、`KVCacheSpec` 上的 `cache_dtype_str` 标签）。C8 布尔开关却短路了这一路由——实际使用的 dtype 是 `self.c8_k_cache_dtype`，由设备类型选择，**而非**由 `--kv-cache-dtype` 选择。
4. **后端扩散。** 随着更多树外后端（Ascend、TPU 等）引入各自的 KV cache 格式，把每一个都硬编码进上游 `CacheDType` `Literal` 既不可扩展，又迫使每次发版都要上游协调。

### 2.3 根因（上游侧）

后端当前无法扩展的唯一真相源是 `vllm/config/cache.py:19-36` 的 **`CacheDType` `Literal`**。它同时驱动：

| 层 | 位置 | 未知值的行为 |
|-------|----------|----------------------------|
| (a) Pydantic 字段类型 | `vllm/config/cache.py:76`（`cache_dtype: CacheDType`） | `ValidationError` |
| (b) CLI `choices` | `vllm/engine/arg_utils.py:185-199`、`:348-349`、`:1161`（`literal_to_kwargs` 从 `get_args(CacheDType)` 派生 `choices`） | `argparse` 拒绝 |
| (c) 运行期二次断言 | `vllm/v1/attention/selector.py:68-73`（`assert kv_cache_dtype in get_args(CacheDType)`） | `AssertionError` |
| (d) torch-dtype 表 | `vllm/utils/torch_utils.py:32-52`（`STR_DTYPE_TO_TORCH_DTYPE`）→ `get_kv_cache_torch_dtype`（`:299-302`） | `ValueError` |
| (e) quant-mode 分发 | `vllm/v1/kv_cache_interface.py:62-74`（`get_kv_quant_mode`） | 静默映射为 `KVQuantMode.NONE` |

后端要加 `sfa-c8`，当前必须上游改 (a)-(d)，最好再改 (e)。这五处全是闭合集合。

## 3. 目标与非目标

**目标**

- 后端插件可注册自定义 KV cache dtype 字符串（如 `"sfa-c8"`），**无需修改上游 vLLM 源码**。
- 注册值是一等的 `--kv-cache-dtype` 选项：出现在 `--help`、被 argparse 接受、通过 Pydantic 校验，并经由既有 `kv_cache_dtype` 管线流到 attention backend。
- 后端提供一个有类型的 **handler**，回答 vLLM 当前各写死表所回答的问题：「什么 torch dtype？」「是否量化？」「什么 quant 模式？」「哪些平台支持？」
- 向后兼容：既有 `CacheDType` 取值行为不变。

**非目标**

- 本 RFC **不规定** 各后端的量化*实现*；只提供注册/分发契约。vLLM-Ascend 的 C8 kernel、逐层选择、设备相关 dtype 等仍归后端所有。
- 本 RFC **不移除** `additional_config`——它仍用于真正不透明的选项。本 RFC 仅就 KV cache dtype 本身提供一等路径，让后端把*开关*迁移过来。
- 不改动 KV cache 落盘布局或传输协议。

## 4. 设计

### 4.1 指导原则：复刻量化方法注册器

vLLM 已为权重量化方法解决过一个同构问题。最贴近的模板是 `vllm/model_executor/layers/quantization/__init__.py`：

```python
QuantizationMethods = Literal[...]              # 上游已知默认值
QUANTIZATION_METHODS: list[str] = list(get_args(QuantizationMethods))  # 可变镜像
_CUSTOMIZED_METHOD_TO_QUANT_CONFIG = {}          # handler 注册表

def register_quantization_config(quantization: str):   # 装饰器
    def _wrapper(quant_config_cls):
        if quantization in QUANTIZATION_METHODS: ...
        else:
            QUANTIZATION_METHODS.append(quantization)
            if sq := current_platform.supported_quantization:   # 自动注册平台支持
                sq.append(quantization)
        _CUSTOMIZED_METHOD_TO_QUANT_CONFIG[quantization] = quant_config_cls
        return quant_config_cls
    return _wrapper
```

更关键的是，*config 字段* 被声明为 `QuantizationMethods | str`（`vllm/config/model.py:203`）——即**字段接受任意字符串**（自定义方法能通过 Pydantic），成员资格在运行期于 `_verify_quantization()`（`vllm/config/model.py:1016-1103`）中对**可变镜像列表**校验，再加上通过 `current_platform.verify_quantization()`（`vllm/platforms/interface.py:940-948`）的逐平台白名单。

本 RFC 把同一形态套用到 KV cache dtype，并因 vLLM 结构而做一处重要细化：`STR_DTYPE_TO_TORCH_DTYPE`（`vllm/utils/torch_utils.py:32-52`）被 **~15 处直接字典索引**读取（`interface.py:686`/`:780`、`model_runner.py:138`、`triton_attn.py:338`、`torch_utils.py:370`/`:401`……），多数*未加保护*（仅判 `!= "auto"`）。只包装 `get_kv_cache_torch_dtype` 仍会在这些点 `KeyError`。因此，复刻 `register_quantization_config` *就地 mutate* 共享 `QUANTIZATION_METHODS` 列表的做法，注册时**把 handler 解析出的 `torch_dtype()` 注入共享 `STR_DTYPE_TO_TORCH_DTYPE` 字典**。这使层 (d) 变成*零改动*层：所有既有 `STR_DTYPE_TO_TORCH_DTYPE[name]` / `name in STR_DTYPE_TO_TORCH_DTYPE` 调用点保持不变。因注入需解析后的 dtype 值，handler 以**及早**方式注册（装饰器，同 `register_quantization_config`），在平台激活时完成，而非惰性 qualname。见 4.3/4.7。

### 4.2 新增 `KVCacheDTypeHandler` 接口

新增一个抽象基类（小型 Protocol/ABC），把四张写死表 (c)-(e) 需要对某个 dtype 回答的问题全部收口，*而无需* 后端去 patch vLLM 内部：

```python
# vllm/config/kv_cache_dtype.py  (新模块)
from __future__ import annotations
from typing import Protocol, runtime_checkable
import torch
from vllm.v1.kv_cache_interface import KVQuantMode


@runtime_checkable
class KVCacheDTypeHandler(Protocol):
    """已注册 `--kv-cache-dtype` 取值的契约。

    后端实现此接口，以回答 vLLM 当前用闭合表
    (STR_DTYPE_TO_TORCH_DTYPE / get_kv_quant_mode /
    is_quantized_kv_cache) 回答的问题。单个 handler 可感知设备
    （例如按 Ascend 代际解析不同 torch dtype）。
    """

    # 公开名称，如 "sfa-c8"。必须与注册键一致。
    name: str

    def torch_dtype(self, *args, **kwargs) -> torch.dtype:
        """KV cache 张量的存储 dtype。**注册时调用一次**并注入共享
        STR_DTYPE_TO_TORCH_DTYPE 字典**，使所有既有直接字典访问点无需改动。
        可查询当前平台（届时设备须已解析——见 4.7）。"""
        ...

    def is_quantized(self) -> bool:
        """若为量化（非原生 dtype）cache 则为 True。替代
        is_quantized_kv_cache 的临时字符串匹配。"""
        ...

    def quant_mode(self) -> KVQuantMode:
        """映射到通用 kernel 使用的上游 KVQuantMode 枚举，或在后端
        完全自管分发时返回 KVQuantMode.NONE / 新的 BACKEND 值（见 4.6）。"""
        ...

    # 能力钩子，与 AttentionBackend.supports_kv_cache_dtype 类似
    # (vllm/v1/attention/backend.py:167)
    def supports_platform(self, platform) -> bool: ...
```

### 4.3 注册表

```python
# vllm/config/kv_cache_dtype.py  (续)
from typing import get_args
from vllm.config.cache import CacheDType            # 上游 Literal
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE   # 共享可变字典

# 1. 上游已知默认值（闭合 Literal 保留作文档 / IDE / 类型检查）
# 2. 可变镜像，自定义项追加其中
KV_CACHE_DTYPES: list[str] = list(get_args(CacheDType))

# 3. handler 注册表：name -> KVCacheDTypeHandler 实例（及早）
_KV_CACHE_DTYPE_HANDLERS: dict[str, KVCacheDTypeHandler] = {}


def register_kv_cache_dtype(name: str):
    """注册一个自定义 `--kv-cache-dtype` 取值（装饰器，及早）。

        @register_kv_cache_dtype("sfa-c8")
        class SfaC8Handler:
            def torch_dtype(self): ...
            def is_quantized(self): ...
            def quant_mode(self): ...

    装饰时 handler 会：(a) 追加到可变 KV_CACHE_DTYPES 列表（驱动 Pydantic
    + CLI 校验）；(b) 解析一次 torch_dtype() 并**注入 STR_DTYPE_TO_TORCH_DTYPE**，
    使所有既有 `dict[name]` / `name in dict` 点零改动；(c) 存储以供后续
    quant_mode()/is_quantized() 查询。
    """
    def _decorate(cls):
        if name in KV_CACHE_DTYPES:
            # 覆盖：告警但放行（复刻 register_quantization_config）
            ...
        else:
            KV_CACHE_DTYPES.append(name)
            if sk := current_platform.supported_kv_cache_dtypes:   # 见 4.5
                sk.append(name)
        inst = cls()
        # 关键步骤：就地 mutate 共享 dtype 字典——下游零改动
        STR_DTYPE_TO_TORCH_DTYPE[name] = inst.torch_dtype()
        _KV_CACHE_DTYPE_HANDLERS[name] = inst
        return cls
    return _decorate


def get_kv_cache_dtype_handler(name: str) -> KVCacheDTypeHandler | None:
    """返回 `name` 的 handler；对没有 handler 的上游 dtype 返回 None
    （dtype 表已覆盖这些）。"""
    return _KV_CACHE_DTYPE_HANDLERS.get(name)


def is_known_kv_cache_dtype(name: str) -> bool:
    return name in KV_CACHE_DTYPES
```

> **为何注入共享字典、而非只在 `get_kv_cache_torch_dtype` 加 handler 回退。**
> `STR_DTYPE_TO_TORCH_DTYPE` 经由直接 `[]` / `in` 在 ~15 处调用点读取
>（`platforms/interface.py:686,780`、`v1/worker/gpu/model_runner.py:138`、
> `v1/attention/backends/triton_attn.py:338`、`utils/torch_utils.py:370,401`，
> 加上 benchmarks/tests）。只把回退装进 `get_kv_cache_torch_dtype`，会让
> 这些点要么 `KeyError`，要么（`in` 守卫分支）静默跳过自定义路径。就地
> mutate 共享字典——与 `register_quantization_config` mutate 共享
> `QUANTIZATION_METHODS` 列表完全一致——用一行覆盖所有点，下游零编辑。
>
> **时序与 `--help` 廉价性。** 及早注入需设备已解析，故注册在平台激活时
>（首次访问 `current_platform`）执行，*晚于* `--help` 返回（CLI `--help`
> 在 engine-config 构造之前退出，从不触碰 `current_platform`）。CLI 用
> B1（`metavar`、无 `choices`），解析器构造完全不查注册表，故 `--help`
> 廉价、从不导入 `torch_npu`/CANN。首次真正的 `current_platform` 访问
>（在 `EngineArgs` config 构建期间）触发平台加载 → 后端
> `register_kv_cache_dtype` → 字典注入，全部早于任何
> `STR_DTYPE_TO_TORCH_DTYPE["sfa-c8"]` 访问。对 dtype 设备相关的后端
>（Ascend：A2/A3 用 int8、A5 用 fp8_e4m3fn），`torch_dtype()` 须在平台激活
> 时可调用；由于设备每进程固定、于平台 init 期间检测，成立。若未来某后端
> 只能更晚解析 dtype，可在那时再次注入（字典 mutate 幂等）。

### 4.4 对五处闭合层的改动

**层 (a) —— Pydantic 字段类型**（`vllm/config/cache.py:76`）：
```python
# 改前
cache_dtype: CacheDType = "auto"
# 改后 —— 接受任意字符串，对注册表校验
cache_dtype: CacheDType | str = "auto"
```
这复刻了 `quantization: QuantizationMethods | str | None`（`vllm/config/model.py:203`）。Pydantic 不再拒绝自定义字符串；字段校验器（当前 `_validate_cache_dtype`，位于 `cache.py:276-294`，仅打日志）新增成员检查：

```python
@field_validator("cache_dtype", mode="after")
@classmethod
def _validate_cache_dtype(cls, cache_dtype):
    if not is_known_kv_cache_dtype(cache_dtype):
        raise ValueError(
            f"Invalid kv_cache_dtype: {cache_dtype!r}. "
            f"Valid values are: {KV_CACHE_DTYPES}"
        )
    ...  # 既有日志
    return cache_dtype
```

**层 (b) —— CLI `choices`**（`vllm/engine/arg_utils.py`）：
`literal_to_kwargs`（`:185-199`）从 `get_args(CacheDType)` 派生 `choices`。两种选择（择一）：

- **方案 B1（最小、推荐）：** 不再为 `--kv-cache-dtype` 输出 `choices`，改为输出 `metavar="DTYPE"`，help 文本列出上游默认值 + "（后端可注册更多）"。这与 `literal_to_kwargs` 在类型 union 含 `str` 时已退化为 `metavar`（`:193-196`）的做法一致。由于 `cache_dtype` 现已为 `CacheDType | str`，`contains_type(type_hints, str)` 为真，`metavar` 会**自动**被选中——除了放宽注解外无需改 arg_utils。校验随之在 Pydantic 校验器（层 a）给出精确报错。
- **方案 B2（更丰富的 `--help`）：** 引入 `get_kv_cache_dtype_kwargs()`，在解析器构建时构建 `choices=list(KV_CACHE_DTYPES)`。要求（至少名称的）注册在 `make_arg_parser()` 之前完成（见 4.7）。UX 最佳，但新增了时序约束。

首版推荐 B1，因为它彻底移除了闭合 `choices`，使注册时序只需运行期即可；B2 是未来的增强。

**层 (c) —— selector 二次断言**（`vllm/v1/attention/selector.py:68-73`）：
把 `assert kv_cache_dtype in get_args(CacheDType)` 改为 `assert is_known_kv_cache_dtype(kv_cache_dtype)`。

**层 (d) —— torch-dtype 表**（`vllm/utils/torch_utils.py:32-52`）：
**下游零改动。** 注册时直接就地 mutate 共享 `STR_DTYPE_TO_TORCH_DTYPE` 字典（见 4.3），故所有既有消费者保持不变——包括否则会 `KeyError` 的*未守卫*直接索引：

| 位置 | 访问方式 |
|------|----------|
| `vllm/platforms/interface.py:686` | `STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]`（仅以 `!= "auto"` 守卫） |
| `vllm/platforms/interface.py:780` | `STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]` |
| `vllm/v1/worker/gpu/model_runner.py:138` | `STR_DTYPE_TO_TORCH_DTYPE[self.cache_config.cache_dtype]` |
| `vllm/v1/attention/backends/triton_attn.py:338` | `STR_DTYPE_TO_TORCH_DTYPE[cache_dtype_str]` |
| `vllm/utils/torch_utils.py:370` | `STR_DTYPE_TO_TORCH_DTYPE[kv_algo_str]`（`get_kv_cache_quant_algo_dtype`） |
| `vllm/utils/torch_utils.py:401` | `return STR_DTYPE_TO_TORCH_DTYPE[kv_cache_dtype]`（`kv_cache_dtype_str_to_dtype`） |
| `vllm/utils/torch_utils.py:293-300` | `in` + `[]`（`get_kv_cache_torch_dtype` 快路径） |

`get_kv_cache_torch_dtype` 不再新增回退分支；字典本就含该项。（`vllm/config/model.py:1987` 那个独立的 `_STR_DTYPE_TO_TORCH_DTYPE` 服务于*模型* dtype 而非 KV cache，不受影响。）

**层 (e) —— quant-mode 分发**（`vllm/v1/kv_cache_interface.py:62-74`、`get_kv_quant_mode`）：自定义名称委托给 handler，上游 dtype 回退到既有 `==`/`startswith` 分支：
```python
def get_kv_quant_mode(kv_cache_dtype: str) -> KVQuantMode:
    handler = get_kv_cache_dtype_handler(kv_cache_dtype)
    if handler is not None:
        return handler.quant_mode()
    ... # 既有分支
```
重复的 `is_quantized_kv_cache` 辅助函数（`vllm/v1/kv_cache_interface.py:77-83` **与** `vllm/utils/torch_utils.py:75-85`——当前两套并行实现）应合并为统一查询注册表的实现；见「开放问题」。

### 4.5 逐平台白名单（复刻 `supported_quantization`）

新增 `Platform.supported_kv_cache_dtypes: list[str] = []` 与 `Platform.verify_kv_cache_dtype(name)`，与 `supported_quantization` / `verify_quantization`（`vllm/platforms/interface.py:170, 940-948`）完全平行：

```python
# vllm/platforms/interface.py
supported_kv_cache_dtypes: list[str] = []   # [] 表示"不强制白名单"

@classmethod
def verify_kv_cache_dtype(cls, dtype: str) -> None:
    if cls.supported_kv_cache_dtypes and dtype not in cls.supported_kv_cache_dtypes:
        raise ValueError(
            f"{dtype} kv-cache-dtype is currently not supported in {cls.device_name}."
        )
```
`register_kv_cache_dtype` 在注册后端平台激活时把名称自动追加到 `current_platform.supported_kv_cache_dtypes`（复刻 `vllm/model_executor/layers/quantization/__init__.py:95-96` 对 `current_platform.supported_quantization` 的自动追加）。具体的 `NPUPlatform.supported_kv_cache_dtypes` 也可硬编码，以便与导入顺序无关地 fail-fast。

### 4.6 后端如何自管分发（无需上游字符串匹配）

一条关键设计约束：vLLM 的通用 kernel **不应**增加 `if kv_cache_dtype == "sfa-c8"` 分支。既有 `KVQuantMode` 枚举（`vllm/v1/kv_cache_interface.py:33-59`）本就意在充当这层间接（「*供 attention backend 与 kernel 分发量化逻辑，而不必对 `kv_cache_dtype` 做字符串匹配*」）。我们扩展该契约：

- 当后端完全自管 kernel 分发时，handler 返回 `KVQuantMode.NONE`（或新哨兵 `KVQuantMode.BACKEND`）——即通用 FP8/逐 token-head kernel 不得介入。后端的 attention backend（`AscendSFABackend` 等）直接读取 `cache_config.cache_dtype` 并自选 kernel。
- 或者，后端把其 dtype 映射到既有 `KVQuantMode` 以复用通用 kernel（如某 INT8 后端返回 `KVQuantMode.INT8_PER_TOKEN_HEAD`）。

这使得 `get_kv_quant_mode` 成为唯一分发点：它返回枚举成员，绝不返回后端专属字符串。

### 4.7 注册时序与入口点

平台插件入口点系统已存在（`vllm/plugins/__init__.py`，组 `vllm.platform_plugins`）。vLLM-Ascend 经 `setup.py:543-551` 注册：
```python
"vllm.platform_plugins": ["ascend = vllm_ascend:register"],
```
其中 `register()` 返回平台 qualname（`vllm_ascend/__init__.py:73-76`）。

插件 `register()` 回调在 `current_platform` 解析期间运行（`vllm/platforms/__init__.py:211-251`）。因层 (d) 注入需解析后的设备 dtype，**`register_kv_cache_dtype` 必须在平台激活时**（`NPUPlatform` 实例化、设备已检测）执行，而非在仅返回 qualname 的 `register()` 回调里。具体而言，vLLM-Ascend 在 `NPUPlatform.__init__`（或平台激活钩子）中调用它，届时 `get_ascend_device_type()` 可靠。该时序安全，因为：

- `--help` 在任何 `current_platform` 访问前退出 → 从不触发注册 → 保持廉价（不导入 `torch_npu`）。
- 首次 `current_platform` 访问发生在 `EngineArgs` config 构建期间（如 `CacheConfig._validate_cache_dtype` → `current_platform.verify_kv_cache_dtype`，或 `register_quantization_config` 式自动追加）→ 触发平台加载 → 后端 `register_kv_cache_dtype` → 字典注入，**全部早于** model runner / 平台 interface 中任何 `STR_DTYPE_TO_TORCH_DTYPE["sfa-c8"]` 访问。

CLI 选项：

- **B1（推荐）：** `metavar`（无 `choices`）。解析器从不查注册表，故注册时序仅运行期即可，`--help` 不受影响。
- **B2：** 专用及早加载入口点组 `vllm.kv_cache_dtype_plugins`（回调返回 `(name, …)` 列表），在 `vllm/__init__.py` 中于任何 CLI `--help` 前加载，复刻 `load_plugins_by_group`（`vllm/plugins/__init__.py:36-74`）。仅当需要准确 `--help` choices 时才需要；先发 B1。

### 4.8 向后兼容

- `CacheDType` `Literal` **保留**——它仍是上游默认值的权威清单，供类型检查器与文档使用。把字段放宽为 `CacheDType | str` 是严格加性的（每个既有取值仍能通过校验）。
- `STR_DTYPE_TO_TORCH_DTYPE` 既有表项全保留；自定义 dtype 经注册时注入新增表项，所有既有访问点零改动。成立的原因：每个 importer 绑定的是同一可变 dict 对象的**引用**（非拷贝）——见 §8 护栏。`get_kv_quant_mode` 保留既有 `==`/`startswith` 分支，仅对已注册名称回退到 handler。既有 FP8/TurboQuant/nvfp4 路径不受影响。
- `additional_config` 不变；后端可渐进迁移。

## 5. vLLM-Ascend 迁移草案

上游改动落地后，vLLM-Ascend 把*开关*从 `additional_config` 迁到 `--kv-cache-dtype`，同时保留其 kernel/层逻辑。

### 5.1 注册 dtype + handler

在 `vllm_ascend/__init__.py`（或新模块 `vllm_ascend/quantization/kv_cache_dtype.py`，于平台注册期间导入）：

```python
from vllm.config.kv_cache_dtype import register_kv_cache_dtype
from vllm.v1.kv_cache_interface import KVQuantMode

@register_kv_cache_dtype("sfa-c8")
class SfaC8Handler:
    """Ascend Sparse-Flash-Attention C8 KV cache（设备相关 dtype）。

    为 DSA 模型（DeepSeek-V3.2、GLM-5/5.2）激活打包 C8 主 cache
    + Lightning-Indexer C8 cache。替代原先的
    additional_config 开关 enable_sparse_sfa_c8 / enable_sparse_li_c8。
    """
    name = "sfa-c8"

    def torch_dtype(self):
        # 设备相关：复刻 sfa_v1.py:614-622
        from vllm_ascend.device.device_op import get_ascend_device_type, AscendDeviceType
        if get_ascend_device_type() == AscendDeviceType.A5:
            return torch.float8_e4m3fn
        return torch.int8

    def is_quantized(self):
        return True

    def quant_mode(self):
        return KVQuantMode.BACKEND     # 后端自管分发；见 4.6

    def supports_platform(self, platform):
        return platform.__class__.__name__ in ("NPUPlatform",)
```

（LI-C8 / reshape-opt 仍是逐层细化；见 5.2。）

### 5.2 读取开关

把 `vllm_ascend/ascend_config.py:249-253` 对 `additional_config` 的读取替换为由 `cache_config.cache_dtype` 驱动的单一派生标志：

```python
self.kv_cache_dtype = vllm_config.cache_config.cache_dtype
self.enable_sparse_sfa_c8 = self.kv_cache_dtype == "sfa-c8" and use_sparse
# enable_sparse_li_c8 / c8_enable_reshape_optim 变为逐层细化，
# 仍可通过 additional_config 配置（属真正逐层选项）。
```

~30 处 `enable_sparse_sfa_c8` 分支点维持原样——它们本就正确门控了设备相关 kernel 路径；只有其*真相源*迁移了。`sfa_v1.py:614-622` 与 `model_runner_v1.py:378-386`（重复）的设备相关 dtype 选择坍缩为 `get_kv_cache_dtype_handler("sfa-c8").torch_dtype()`，消除重复逻辑。

### 5.3 弃用路径

为避免硬中断，vLLM-Ascend 保留接受遗留 `enable_sparse_sfa_c8`/`enable_sparse_li_c8` 键，并给出指向 `--kv-cache-dtype sfa-c8` 的弃用告警，复刻 `enable_sparse_c8` -> 拆分的先例（见 `docs/source/user_guide/release_notes.md:80`）。

### 5.4 路径 B（GQA / ModelSlim C8）不受影响

由 checkpoint 驱动的 `"C8"` 路径（`vllm_ascend/quantization/modelslim_config.py:1002`、`enable_c8_quant`）仍由 `quant_config` 驱动，因其以模型 checkpoint 为键，而非用户参数。它可选地*也*注册一个 dtype，但不在本 RFC 范围内。

## 6. 考虑过的替代方案

1. **继续把取值硬编码进 `CacheDType`。** 对单个后端最简单；但随着多个树外后端各自需要设备相关表项，会污染上游 Literal 与 `STR_DTYPE_TO_TORCH_DTYPE`，塞入硬件名。否决。
2. **把 `additional_config` 泛化为有类型的逐平台 config 对象。** 能改善校验，但既不能提供一等 `--kv-cache-dtype` UX，也无法接入 attention backend 的 `kv_cache_dtype` 管线。本 RFC 是其严格子集，可与未来有类型 additional_config 共存。
3. **子命令 / 逐后端 CLI。** 否决——会割裂统一 CLI，并破坏预期单次 `vllm serve` 调用的既有编排工具。
4. **新增顶层 `--backend-kv-cache-dtype` 参数。** 增加并行旋钮，并使"哪个生效"模糊。否决；复用既有 `--kv-cache-dtype` 正是本 RFC 的目的。

## 7. 开放问题

1. **后端自管分发的 `KVQuantMode`。** 引入新哨兵 `KVQuantMode.BACKEND`（或 `BACKEND_CUSTOM`），还是让后端返回 `NONE` 并依赖其 attention backend 直接读 `cache_config.cache_dtype`？前者更显式，能让通用代码对后端模式断言"不分配逐张量 scale 张量"；后者无需改枚举。倾向于新增 `BACKEND`。
2. **合并重复的 `is_quantized_kv_cache`。** 当前有两套实现（`vllm/utils/torch_utils.py:75-85` 与 `vllm/v1/kv_cache_interface.py:77-83`）。本次迁移是将其坍缩为单一注册表感知函数的自然时机——但会触及若干导入点，应作为独立 PR。
3. **CLI `choices` vs `metavar`（B1 vs B2）。** 确认社区倾向保证准确的 `--help`（B2，需及早入口点加载）还是最小、无时序的 `metavar`（B1）。建议先发 B1。
4. **每后端多 dtype。** vLLM-Ascend 应把 `sfa-c8`、`li-c8` 等注册为各自顶层 dtype，还是单一 `sfa-c8` 加上经 `additional_config` 的逐层细化？后者对常见场景保持 CLI 整洁；逐层开关本就逐层，适合 `additional_config`。本 RFC 假定一个顶层 dtype 加细化。

## 8. 风险与缓解

- **字符串拼写错误类。** 从 `additional_config`（无校验）迁移*到*注册 dtype 引入了校验，故拼写错误现在会在 config 构造时 fail-fast，而非静默关闭 C8。这是改进，但属行为变化——由弃用窗口覆盖。
- **导入顺序陷阱。** 注册在平台激活时执行；若后端包未安装，`current_platform` 解析在 config 构建时 fail-fast（而非 `--help`，后者从不触碰 `current_platform`）。该 dtype 的字典表项未被注入，故任何对该 dtype 的 `STR_DTYPE_TO_TORCH_DTYPE[name]` 访问抛 `KeyError`——立即暴露，而非静默。
- **上游评审负担。** 改动触及 4 个热点文件（cache.py、arg_utils.py、selector.py、kv_cache_interface.py）外加一个新注册模块和 `Platform` 上的小白名单。层 (d) **无需**改动 `torch_utils.py` 及其 ~15 个调用方——注入共享字典即覆盖。其余编辑均为加性回退，故对既有 dtype 的波及面近乎为零。
- **护栏：禁止对共享 dtype 字典做快照。** 层 (d) 依赖 `STR_DTYPE_TO_TORCH_DTYPE` 始终是**被实时读取的可变共享对象**——全部 ~15 个调用点都用直接 `[]` / `in` / `.get()`（已验证：当前无 `.copy()`、`dict(...)`、`list(.keys())`、`tuple(...)`、comprehension 或重绑定）。`from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE` 绑定的是**同一 dict 对象的引用**，故注册期 mutate 对所有 importer 可见，与 import 顺序无关。此契约**必须维持**：未来任何 `dict(STR_DTYPE_TO_TORCH_DTYPE)`、`SUPPORTED = list(STR_DTYPE_TO_TORCH_DTYPE)`、`tuple(.keys())`，或在 import 期捕获 keys/values 的写法，都会产生陈旧快照、静默漏掉更晚注册的后端——这类 bug 难以察觉。`KV_CACHE_DTYPES` 可变列表同理。缓解：在字典定义处（`torch_utils.py:32`）加内联注释标注其"注册期可变"，并加一条 CI grep 测试断言不存在快照模式。（注：`vllm/config/model.py:1987` 的 `_STR_DTYPE_TO_TORCH_DTYPE` 是服务于*模型* dtype 的*独立*手写字典——与 KV cache 无关，不受影响。）

## 9. 推进计划

1. **阶段 1（上游 vLLM）：** 新增 `KVCacheDTypeHandler` + 注册表；把 `cache_dtype` 放宽为 `CacheDType | str`；将层 (a)、(b)、(c) 接成加性检查；**层 (d) 零改动**（在 `register_kv_cache_dtype` 中注入共享字典）；仅对单个 `get_kv_quant_mode`（层 e）加 handler 回退；新增 `Platform.supported_kv_cache_dtypes` + `verify_kv_cache_dtype`。不新增任何 dtype 取值；既有行为不变。
2. **阶段 2（vLLM-Ascend）：** 注册 `sfa-c8` + handler；对遗留 `additional_config` 键加弃用告警；更新 `docs/.../additional_config.md` 与 GLM-5.2 教程（`docs/source/tutorials/models/GLM5.2.md`）改用 `--kv-cache-dtype sfa-c8`。
3. **阶段 3（清理）：** 合并重复的 `is_quantized_kv_cache`；待弃用窗口关闭后移除遗留 `additional_config` 键。

## 附录 A —— 参考：仓内既有可扩展模式

| 模式 | 关键文件 | 机制 | 本 RFC 中的复用 |
|---------|-----------|-----------|-------------------|
| 量化方法注册器 | `vllm/model_executor/layers/quantization/__init__.py:47-105` | `Literal` + 可变镜像列表 `QUANTIZATION_METHODS` + `_CUSTOMIZED_METHOD_TO_QUANT_CONFIG` 字典 + `register_quantization_config` 装饰器；字段类型 `QuantizationMethods \| str`（`vllm/config/model.py:203`）；运行期于 `_verify_quantization` 校验（`:1016`）；平台白名单 `verify_quantization`（`vllm/platforms/interface.py:940`） | **主模板**——把同一形态套用到 `CacheDType` |
| attention backend 注册器 | `vllm/v1/attention/backends/registry.py:34-284` | `AttentionBackendEnum`（值 = 默认 qualpath）+ `register_backend` 装饰器 + `_ATTN_OVERRIDES` 字典 + `CUSTOM = None` 哨兵 + `resolve_obj_by_qualname` 惰性导入 | 曾考虑 qualname 惰性解析，但对 dtype 否决（注入需及早拿到解析值）——见 4.3。`CUSTOM` 式哨兵模式仍启发设计 |
| 平台插件入口点 | `vllm/plugins/__init__.py:18-74`、`vllm/platforms/__init__.py:202-251` | `vllm.platform_plugins` 入口点组、`load_plugins_by_group`、`resolve_obj_by_qualname`、`PlatformEnum.OOT` 哨兵 | 后端注册入口点；潜在 `vllm.kv_cache_dtype_plugins` 组 |
| 逐平台能力钩子 | `vllm/platforms/interface.py:170, 362-370, 940-948`；`AttentionBackend.supports_kv_cache_dtype`（`vllm/v1/attention/backend.py:167-173`） | `supported_quantization` ClassVar + `verify_*` classmethod + 逐平台 `get_attn_backend_cls` 分发 | `Platform.supported_kv_cache_dtypes` + `verify_kv_cache_dtype` |

## 附录 B —— 当前闭合层（需改的具体调用点）

| # | 层 | 文件:行 | 改动 |
|---|-------|-----------|--------|
| a | Pydantic 字段类型 | `vllm/config/cache.py:76` | `cache_dtype: CacheDType` -> `CacheDType \| str`；在校验器 `:276-294` 增加成员检查 |
| b | CLI `choices` | `vllm/engine/arg_utils.py:185-199`（`literal_to_kwargs`）、`:348-349`、`:1161` | 注解放宽为 `... \| str` 后自动选 `metavar`（B1）；或构建 `choices=KV_CACHE_DTYPES`（B2） |
| c | 运行期二次断言 | `vllm/v1/attention/selector.py:68-73` | `get_args(CacheDType)` -> `is_known_kv_cache_dtype()` |
| d | torch-dtype 表 | `vllm/utils/torch_utils.py:32-52`（被 `:293-300, 370, 401` 及 `platforms/interface.py:686,780`、`v1/worker/gpu/model_runner.py:138`、`v1/attention/backends/triton_attn.py:338` 等调用） | **零改动**：注册时 `STR_DTYPE_TO_TORCH_DTYPE[name] = handler.torch_dtype()`；所有直接 `[]`/`in` 调用点不变 |
| e | quant-mode 分发 | `vllm/v1/kv_cache_interface.py:62-74` | 对已注册名称回退到 `handler.quant_mode()` |
| f | 重复辅助函数 | `vllm/utils/torch_utils.py:75-85` 与 `vllm/v1/kv_cache_interface.py:77-83` | 合并为注册表感知的单实现（阶段 3） |
| g | 平台白名单 | `vllm/platforms/interface.py:170, 940-948` | 新增 `supported_kv_cache_dtypes` + `verify_kv_cache_dtype` |

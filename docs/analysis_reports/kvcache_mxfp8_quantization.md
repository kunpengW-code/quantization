# KVCache MXFP8 量化分析

## 1. 尾块

### 1.1 V 产生尾块的原因

#### Attention 中的两个矩阵乘法

```shell
# 1. Attention Score 计算
S = Q @ K^T    # [B*N, seq_len, head_dim] @ [B*N, head_dim, seq_len] → [B*N, seq_len, seq_len]
#                 收缩维度 = head_dim
P = softmax(S)
# 2. Output 计算
O = P @ V     # [B*N, seq_len, seq_len] @ [B*N, seq_len, head_dim] → [B*N, seq_len, head_dim]
#                 收缩维度 = seq_len
```

#### Q/K 无尾块，V 有尾块

| 张量 | MXFP8 分块轴（收缩维度） | 典型值 | 能否被 32 整除 |
|------|--------------------------|--------|----------------|
| Q | head_dim | 64, 128, 256 | **能** |
| K | head_dim | 64, 128, 256 | **能** |
| V | **seq_len** | 任意值（1, 2, ..., 1000, ...） | **不一定** |

- **Q 和 K**：MXFP8 沿 `head_dim` 分块，`head_dim` 通常是 32 的倍数（如 128 = 4×32），所以不会产生尾块。
- **V**：在 `P @ V` 中，MXFP8 需要沿 `seq_len` 分块。`seq_len` 是动态的，取决于输入序列长度，几乎不可能保证始终是 32 的倍数，因此产生**尾块（tail block）**。

---

### 1.2 为什么 MXFP8 必须在收缩维度进行量化

#### MX 格式的本质：Block-wise 共享 Scale

MXFP8 将 **32 个相邻元素** 分为一个 block，共享一个 scale factor：

```
block = [e₀, e₁, ..., e₃₁]  →  shared_scale × [q₀, q₁, ..., q₃₁]
```

#### 矩阵乘法中的关键问题

以 `C = A @ B` 为例，收缩维度为 K：

```
C[i,j] = Σₖ A[i,k] × B[k,j]
```

**沿收缩维度 K 分块**（block 边界对齐）：

```
C[i,j] = Σ_block (scale_A_block × scale_B_block) × Σₖ₌₁³² qA[i,k] × qB[k,j]
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                每个 block 只需 1 次 scale 乘法，硬件可高效处理
```

**沿非收缩维度分块**（block 跨越 K 轴）：

```
C[i,j] = Σₖ (scale_A[i, block(k)] × qA[i,k]) × (scale_B[block(k), j] × qB[k,j])
```

每个元素 k 可能对应**不同的 scale**，无法将 scale 提取到求和外面。

#### 对比

| 分块方式 | scale 应用次数 | 硬件友好度 |
|---------|---------------|-----------|
| 沿收缩维度 | K/32 次（per block） | Tensor Core 原生支持 |
| 沿非收缩维度 | K 次（per element） | 需要逐元素 dequant，丧失硬件加速 |

#### 根本原因：Tensor Core 的计算原语

硬件 Tensor Core 的 MX 矩阵乘法指令是这样设计的：

```
硬件一次处理: [32 elements, shared_scale_A] × [32 elements, shared_scale_B]
→ 要求收缩维度上 32 个元素共享同一个 scale
→ block 边界必须沿收缩维度对齐
```

如果 block 不是沿收缩维度划分，硬件无法在一次指令中正确配对元素和对应的 scale，要么无法执行，要么退化为低效的逐元素计算。

---

## 2. 尾块处理方案

在 `P @ V` 中，`seq_len` 随推理逐步增长，尾块（不足 32 个 token 的部分）会频繁变化。以下介绍两种处理方案。

### 2.1 方案一：Rescaling（重缩放）

#### 核心思路

尾块部分**始终量化存储**，当新 token 到来导致尾块变化时，先反量化、追加新数据、再重新量化。

#### 流程

```
初始状态（seq_len = 35，尾块 = 3 个 token）：

  ┌─────────────────────┬───────────┐
  │  block 0 (32 tokens)│ tail (3)  │  ← 全部以 MXFP8 + scale 存储
  └─────────────────────┴───────────┘

Step 1: 新 token 到来 → 反量化尾块
  ┌───────────┐     ┌───────────────────┐
  │ tail (3)  │ →   │ tail_fp16 (3)     │
  │  MXFP8    │     │ + new_token (1)   │
  └───────────┘     └───────────────────┘
                        tail_fp16 (4)

Step 2: 重新计算 scale，量化尾块
  ┌───────────────────┐     ┌───────────┐
  │ tail_fp16 (4)     │ →   │ tail (4)  │  ← 新的 shared_scale
  └───────────────────┘     │  MXFP8    │
                            └───────────┘

Step 3: 尾块满 32 时，升级为完整 block
  ┌─────────────────────┬───────────┐
  │  block 0 (32 tokens)│ block 1   │  ← scale 固定，不再变化
  └─────────────────────┴───────────┘
```

#### 计算开销分析

| 操作 | 频率 | 开销 |
|------|------|------|
| 反量化尾块 | 每个新 token | O(tail_size × head_dim)，tail_size < 32 |
| 追加新 token | 每个新 token | O(1 × head_dim) |
| 重新量化尾块 | 每个新 token | O(tail_size × head_dim)，需重新计算 scale |

- 尾块最多 31 个 token，反量化/量化开销较小
- 但**每个 decode step 都需要执行**，累积开销不可忽视
- 已形成的完整 block 不受影响，scale 无需更新

#### 优缺点

| 优点 | 缺点 |
|------|------|
| 存储统一，全部为 MXFP8 格式 | 每个 step 都有 dequant → append → quant 的额外计算 |
| 不引入混合精度管理逻辑 | 反复量化可能引入累积精度损失 |
| 硬件访存模式一致 | scale 频繁更新增加写带宽 |

---

### 2.2 方案二：高精窗口保护（FP16 Tail Window）

#### 核心思路

尾块部分**保留浮点精度（FP16/BF16）**，仅对 `seq_len` 中能被 32 整除的前缀部分进行 MXFP8 量化。

#### 流程

```
seq_len = 35 时：

  ┌─────────────────────┬───────────┐
  │  block 0 (32 tokens)│ tail (3)  │
  │       MXFP8         │   FP16    │  ← 尾块保持浮点
  └─────────────────────┴───────────┘

新 token 到来 → 直接追加到 FP16 尾块，无需反量化/重新量化：

  ┌─────────────────────┬───────────┐
  │  block 0 (32 tokens)│ tail (4)  │
  │       MXFP8         │   FP16    │
  └─────────────────────┴───────────┘

尾块积满 32 个 token → 量化为完整 block，尾块清零：

  ┌─────────────────────┬───────────┬────────┐
  │  block 0 (32 tokens)│ block 1   │ tail(0)│
  │       MXFP8         │   MXFP8   │  FP16  │
  └─────────────────────┴───────────┴────────┘
```

#### 矩阵乘法处理

`P @ V` 拆分为两部分计算：

```
O = P[:, :aligned] @ V_quantized[:aligned, :]  +  P[:, aligned:] @ V_tail[aligned:, :]
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    MXFP8 矩阵乘法（Tensor Core 加速）              FP16 矩阵乘法（常规计算单元）
```

- 前缀部分：走 Tensor Core MXFP8 通路，硬件原生加速
- 尾块部分：走 FP16 通路，元素少（< 32），开销极低
- 以上内容放在FIA算子内部完成，框架侧不感知

#### 计算开销分析

| 操作 | 频率 | 开销 |
|------|------|------|
| 追加新 token 到 FP16 尾块 | 每个新 token | O(1 × head_dim)，纯写入 |
| FP16 尾块矩阵乘 | 每个新 token | O(tail_size × head_dim)，tail_size < 32 |
| 尾块量化（满 32 时） | 每 32 个 token | O(32 × head_dim)，摊还后极低 |

- 无 dequant/requant 开销
- 尾块矩阵乘规模极小（< 32 行），对整体性能影响微乎其微，且在FIA算子内部完成

#### 优缺点

| 优点 | 缺点 |
|------|------|
| 无反复量化开销，每个 step 只做追加 | 需要管理两种精度的存储和计算路径 |
| 尾块部分零精度损失 | 增加软件复杂度（双缓冲区管理） |
| 摊还量化开销（每 32 个 token 一次） | 增加软件复杂度（双缓冲区管理） |
| 尾块计算量极小，不影响整体吞吐 | 尾块占用 FP16 存储（最多 31 × head_dim × 2B，可忽略） |

---

### 2.3 方案对比

| 维度 | Rescaling | 高精窗口保护 |
|------|-----------|-------------|
| 存储格式 | 统一 MXFP8 | MXFP8 + FP16 混合 |
| 每 step 额外计算 | dequant + quant（~31 × head_dim） | 框架侧不涉及，FIA内部处理 |
| 量化频率 | 每个 token | 每 32 个 token |
| 精度 | 反复量化有累积误差风险 | 尾块部分无损 |

------

## 3. 调研

### 3.1 vLLM NVFP4 KVCache 量化的尾块处理

#### NVFP4 格式概述

NVFP4 是 NVIDIA Blackwell 架构引入的 FP4 量化格式，block size 为 **16**（每个 block 16 个元素共享一个 FP8 scale）。与 MXFP8 的 block size 32 类似，但粒度更细。

#### 量化维度：沿 head_dim 量化，天然无尾块

vLLM 的 NVFP4 KVCache 量化是**沿 head_dim（最后一个维度）** 进行的，而非沿 seq_len：

```python
# vllm/_custom_ops.py
block_size = 16
assert n % block_size == 0, f"last dim has to be multiple of 16, but got {n}."
```

标准 Transformer 的 head_size 通常为 64、128、256，均为 16 的倍数，因此**在量化维度上不存在尾块**。

#### 物理存储布局

每个 token 的 KV 数据被打包为 `[FP4 data | FP8 scales]` 的连续布局：

```python
# vllm/utils/torch_utils.py
def nvfp4_kv_cache_full_dim(head_size: int) -> int:
    return head_size // 2 + head_size // 16
    # head_size=128 → 64 + 8 = 72 bytes per head per token
```

每个 token 独立量化，token 之间无共享 scale。

#### 序列维度的"尾块"：由 Paged Attention 机制处理

对于序列长度维度的不完整页（最后一个 page 未填满），vLLM 使用 FlashInfer 的 **paged attention** 机制：

```python
# vllm/v1/attention/backends/flashinfer.py
paged_kv_last_page_len_np = seq_lens_np % page_size
self.paged_kv_last_page_len.np[:num_reqs] = np.where(
    (paged_kv_last_page_len_np == 0) & (seq_lens_np != 0),
    page_size,
    paged_kv_last_page_len_np,
)
```

`paged_kv_last_page_len` 记录最后一页中有效 token 数，attention kernel 自动 mask 掉无效位置。由于每个 token 的量化是独立的（沿 head_dim），**不完整页中的每个 token 仍然是完整量化的**，无需特殊处理。

#### Scale 缓冲区的 Padding 处理

对于 swizzled scale 布局中的 padding，vLLM 使用**零初始化**防止未初始化的 scale 值污染反量化：

```python
# vllm/_custom_ops.py
def create_fp4_scale_tensor(m, n, device, is_sf_swizzled_layout):
    if is_sf_swizzled_layout:
        rounded_m = round_up(m, 128)
        scale_n = n // block_size
        rounded_n = round_up(scale_n, 4)
        # 必须零初始化：swizzled 布局会 pad M 到 128 的倍数，scale-N 到 4 的倍数
        # 量化 kernel 不会写入所有 padded 位置，但 GEMM 会读取它们
        return torch.zeros(
            (rounded_m, rounded_n // 4), device=device, dtype=torch.int32
        )
```

#### 已知问题：SM120 上 K 维度非 32 对齐

Issue [#48898](https://github.com/vllm-project/vllm/issues/48898) 记录了在 SM120（RTX 6000D Blackwell）上，当 Tensor Parallelism 分片后 K 维度不是 32 的倍数时，CUTLASS NVFP4 GEMM kernel 会读取 padded scale 列的陈旧内存。临时修复为回退到 Marlin backend（PR [#48965](https://github.com/vllm-project/vllm/pull/48965)）。

#### 为什么 NVFP4 可以沿 head_dim 量化而不损失硬件亲和性

在前面的分析中，MXFP8 必须沿收缩维度（seq_len）量化才能保证 Tensor Core 硬件亲和。但 NVFP4 选择了沿 head_dim 量化，这看似矛盾，原因在于两者的计算路径不同：

##### 两个 MatMul 的量化轴分析

| MatMul | 收缩维度 | head_dim 的角色 | 沿 head_dim 量化 |
|--------|---------|----------------|-----------------|
| `S = Q @ K^T` | **head_dim** | 收缩维度 | **硬件亲和** ✓ |
| `O = P @ V` | **seq_len** | 输出维度 | **理论上非亲和** ✗ |

对于 `Q @ K^T`，head_dim 正好是收缩维度，沿 head_dim 量化完全符合硬件要求。

对于 `P @ V`，head_dim 是输出维度而非收缩维度，理论上会导致硬件不亲和。但 NVFP4 能 work 的原因如下：

##### 1. Fused Attention Kernel 不是标准 GEMM

FlashAttention 类 kernel 采用 online softmax 的逐 token 迭代累加模式，而非传统 GEMM 的 tile-and-accumulate 模式：

```
# 标准 GEMM 模式（需要收缩维度对齐）
O[i,j] = Σ_k P[i,k] × V[k,j]  → 需要沿 k（seq_len）分 block

# Fused Attention 模式（逐 token 迭代）
for each token k:
    load V[k, :]  → dequant（沿 head_dim 的 16 个 block）
    O += P[i,k] × V_dequant[k, :]
```

每次加载一个 V token 时，可以直接 dequant 后参与计算。scale 的应用发生在输出维度（head_dim）上，不影响 seq_len 方向的累加。

##### 2. head_dim 上的 scale 数量极少

head_dim=128 时，只有 128/16=8 个 scale。kernel 可以将这 8 个 scale 全部放入寄存器，逐 block 应用 scale 的开销极低：

```
V_dequant[k, :] = [scale_0 × q[0:16], scale_1 × q[16:32], ..., scale_7 × q[112:128]]
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                   8 个 scale 在寄存器中，无额外访存
```

##### 3. 专用 trtllm-gen 后端

vLLM 强制 NVFP4 KVCache 走 `trtllm-gen` attention path：

```python
# vllm/v1/attention/backends/flashinfer.py
if self.is_kvcache_nvfp4:
    backend = "trtllm-gen"  # NVIDIA 专门为这种场景优化的 fused kernel
```

这是 NVIDIA 专门为 NVFP4 KVCache + fused attention 优化的 kernel，内部高效处理了 per-token dequant。

##### 与 MXFP8 的本质区别

| 维度 | NVFP4（沿 head_dim） | MXFP8（沿 seq_len） |
|------|---------------------|---------------------|
| 量化轴 | head_dim（输出维度） | seq_len（收缩维度） |
| 计算路径 | Fused Attention kernel | 标准 Tensor Core GEMM |
| 硬件亲和性 | 依赖专用 kernel 内部处理 | 原生 Tensor Core 支持 |
| 尾块问题 | **无**（每个 token 独立量化） | **有**（seq_len 动态变化） |
| 代价 | 必须使用特定 attention kernel | 需要处理尾块（rescaling/高精窗口） |

**核心结论**：NVFP4 沿 head_dim 量化是一种**设计权衡**——放弃了 `P @ V` 的标准 GEMM 亲和性，换取了每个 token 独立量化（无尾块问题），代价是必须依赖专门的 fused attention kernel（如 trtllm-gen）而非通用 GEMM。这与 MXFP8 沿 seq_len 量化以支持标准 GEMM 路径是两种不同的技术路线。

#### 总结

| 维度 | vLLM NVFP4 的处理方式 |
|------|----------------------|
| 量化轴 | head_dim（非 seq_len） |
| block size | 16 |
| 量化维度尾块 | **不存在**：head_size 必为 16 的倍数，通过 assert 强制保证 |
| 序列维度不完整页 | Paged Attention 的 `last_page_len` 机制，每个 token 独立量化，无需特殊处理 |
| scale padding | 零初始化（`torch.zeros`） |
| 核心思路 | **从设计上规避**：量化轴选择 head_dim 使得尾块问题不存在 |

**关键洞察**：vLLM 的 NVFP4 KVCache 不存在我们讨论的"尾块"问题，因为它沿 head_dim 量化（每个 token 独立），而非沿 seq_len 量化。seq_len 维度的不完整由 paged attention 的 mask 机制处理，与量化无关。这与 MXFP8 沿 seq_len（收缩维度）量化的场景有本质区别。

---

### 3.2 vllm-ascend DeepSeek V4 压缩 Token 的尾块处理

#### DeepSeek V4 压缩机制

DeepSeek V4 在部分注意力层引入了 token 压缩，使用两种压缩比：

| 压缩类型 | 压缩比 | 说明 |
|---------|--------|------|
| C4 | 4:1 | 每 4 个 token 压缩为 1 个 |
| C128 | 128:1 | 每 128 个 token 压缩为 1 个 |

压缩后，KVCache 的有效序列长度变为 `seq_len / compress_ratio`，这个值几乎不可能被量化 block size（如 32）整除，从而产生尾块。

#### 解决方案：压缩感知的 Block Size 设计

vllm-ascend 的核心策略是**为不同压缩层定义不同的 block size**，使 block 大小与压缩后的序列长度自然对齐：

```python
# vllm_ascend/models/layer/attention/layer.py
def get_dsv4_block_sizes():
    # block_size: [mla, swa, c4_state, c128_state], [page_size_padded_t1, page_size_padded_t2]
    _DSV4_BLOCK_SIZES = {
        128: [[128, 128, 8, 32], [16640, 131072]],
        64:  [[64, 64, 4, 16], [8320, 65536]],
        32:  [[32, 32, 2, 8], [4160, 32768]],
    }
```

#### Page Size Padding 保证硬件对齐

对于压缩后的 KVCache 页，vllm-ascend 使用 `page_size_padded` 参数**超量分配**页面大小以满足硬件对齐要求：

```python
# vllm_ascend/models/deepseek_v4.py
class AscendCompressorStateCache:
    def get_kv_cache_spec(self, vllm_config):
        pads = _dsv4_block_sizes()[block_size][1]
        page_size_padded = pads[0] if (state_dim == 512 and compress_ratio == 4) else pads[1]
        return AscendSlidingWindowMLASpec(
            block_size=self.block_size,
            page_size_padded=page_size_padded,  # 16640 或 131072
        )
```

#### 内存分配：Ceiling Division 处理尾部

内存计算使用向上取整除法，自然为尾部不完整 block 分配完整空间：

```python
# vllm_ascend/core/kv_cache_interface.py
def max_memory_usage_bytes(self, vllm_config):
    return cdiv(max_model_len, self.block_size * self.compress_ratio) * self.page_size_bytes
```

#### 专用 KVCache 管理器

vllm-ascend 实现了 `CompressAttentionManager`，专门处理压缩 token 的 KVCache 管理：

```python
# vllm_ascend/core/kv_cache_interface.py
KVCacheSpecRegistry.register(
    kvcache_spec_cls=AscendMLAAttentionSpec,
    manager_class=CompressAttentionManager,
    uniform_type_base_spec=FullAttentionSpec,
)
```

该管理器将物理 block 数量计算为 `ceil(num_tokens / (block_size × compress_ratio))`，而非普通的 `ceil(num_tokens / block_size)`。

#### 多组分层级 KVCache（6 Groups）

DeepSeek V4 Flash 使用 **6 个 KVCache 组**，每组有不同的 block size 和压缩比。vllm-ascend 通过 multi-group layerwise KV cache 支持：

- 每组独立的 GVA（Global Virtual Address）分配
- 每组独立的 block ID 管理（`block_ids_by_group_np`）
- `discard_partial_chunks` 默认为 `true`，丢弃不完整的尾部 chunk

#### 总结

| 维度 | vllm-ascend DeepSeek V4 的处理方式 |
|------|-----------------------------------|
| 核心策略 | 压缩感知的 block size：为 C4/C128 层定义缩小的 block size |
| 硬件对齐 | `page_size_padded` 超量分配页面大小 |
| 内存分配 | `cdiv(max_model_len, block_size × compress_ratio)` 向上取整 |
| 不完整 chunk | `discard_partial_chunks=true`，默认丢弃 |
| KVCache 管理 | 专用 `CompressAttentionManager` 处理压缩域的 block 计数 |
| 核心思路 | **从 block 结构设计上适配**：调整 block size 使压缩后序列自然对齐 |

**关键洞察**：vllm-ascend 没有采用"尾块特殊处理"的思路，而是通过**为不同压缩层定制 block size**，从结构设计层面让压缩后的序列长度与 block 边界对齐。同时通过 `page_size_padded` 和 `discard_partial_chunks` 处理残留的对齐问题。

#### 尾块的实际处理：SWA 滑动窗口兜底

上述 block size 设计解决了压缩域的对齐问题，但还有一个更根本的问题：**未凑满压缩组的 token 怎么办？**

例如 C128 压缩，decode 阶段每次只来 1 个 token，需要累积 128 个才能压缩。这 127 个"等待中"的 token 如何参与注意力计算？

##### 双路径注意力架构

DeepSeek V4 采用 **SWA + 压缩 KV 双路径**注意力，每个 token 同时写入两条路径：

```
新 token 到达（decode step）
    │
    ├──→ SWA cache：始终写入原始 KV，滑动窗口立即可见
    │
    └──→ State cache：累积压缩器中间状态
              │
              ├── 未凑满（如 C128 只有 50/128 个）
              │     └── 无压缩 KV 输出，该 token 仅通过 SWA 路径可见
              │
              └── 凑满（128/128 个）
                    └── 压缩 KV 输出，该 token 同时通过压缩路径 + SWA 路径可见
```

##### SWA 覆盖所有 token

每个 token **无条件写入 SWA cache**，不受压缩状态影响：

```python
# vllm_ascend/attention/dsa_v1.py — 压缩之前先写 SWA
DeviceOperator.dsa_kv_compress_scatter(swa_kv_cache, kv, swa_decode_metadata.slot_mapping)

# 然后才执行压缩
compressed_kv = torch.ops._C_ascend.compressor(...)
if compressed_kv.shape[0] > 0:  # 仅凑满时才有输出
    DeviceOperator.dsa_kv_compress_scatter(compress_kv_cache, compressed_kv, ...)
```

##### 注意力计算：两条路径合并

NPU 融合 attention kernel 同时消费两条路径：

```python
attn_output = attn_op(
    q,
    ori_kv=swa_kv_cache,          # SWA：覆盖窗口内所有 token（含未压缩尾部）
    cmp_kv=compress_kv_cache,      # 压缩：仅已完成的压缩组
    ori_mask_mode=4,               # 滑动窗口 mask
    ori_win_left=window_size - 1,  # 窗口左边界
    ...
)
```

##### 各区域 Token 的可见性

```
序列: [... 窗口外 ...|--- 滑窗 window_size ---|]
                      ↑                       ↑
                 窗口左边界                 当前位置

压缩 KV: [C0][C1]...[C30][C31]  ← 已完成的压缩组
SWA:     [........全部窗口内原始 token........]
```

| Token 状态 | SWA 可见 | 压缩 KV 可见 |
|-----------|---------|-------------|
| 窗口内 + 已压缩 | ✓ | ✓（重叠，双路径） |
| 窗口内 + 未凑满压缩组 | ✓ | ✗（仅 SWA 兜底） |
| 窗口外 + 已压缩 | ✗（已驱逐） | ✓（长距离上下文） |
| 窗口外 + 未凑满 | ✗ | ✗（丢失） |

##### State Cache：增量计算累加器

未凑满的 token 不会被丢弃，其压缩器中间状态保存在 **state cache** 中：

```
Token 1 → 计算 partial state → 存入 state cache
Token 2 → 计算 partial state → 累加到 state cache
...
Token 128 → 计算 partial state → 累加 → 凑满！→ 输出 compressed KV，清空 state cache
```

State cache 存的是压缩器的投影中间状态（`wkv`/`wgate` 投影 + APE 位置编码），与 SWA cache 中的原始 KV **完全不同**，两者各司其职。

##### 总结

DeepSeek V4 的尾块处理本质上是**架构层面的设计**，而非工程 hack：

- **SWA 路径**：保证每个 token 立即可见，零延迟
- **压缩路径**：提供长距离上下文（窗口外历史），但有延迟（需凑满压缩组）
- **重叠区域**：双路径互补，kernel 内部正确合并
- **State cache**：增量累加，避免回溯重算

---

## 4. 框架侧实现

kvcache MXFP8 量化最终方案为高精窗口保护，以下实现基于此方案展开。

### 4.1 vLLM 上游直接实现 MXFP8 KVCache 量化

#### 1. KVCache dtype 注册

在 `vllm/v1/kv_cache_interface.py` 中新增 MXFP8 枚举：

```python
class KVQuantMode(Enum):
    # ...existing...
    MXFP8 = 6  # MX FP8 block-wise quantization, block_size=32
```

#### 2. KVCache spec 与 page size

K 和 V 的存储需求不同：

```
K cache（沿 head_dim 量化，无尾块）：
  page_size = block_size × num_kv_heads × (head_size × 1B + head_size/32 × 1B)
  # MXFP8 数据(1B/elem) + FP8 scale(1B per 32 elems)

V cache（沿 seq_len 量化，有尾块）：
  量化部分：block_size × num_kv_heads × (head_size × 1B + 1 × 1B)
  # 每个 token 的 head_dim 方向共享 1 个 scale（沿 seq_len 分 block）
  FP16 尾块：最多 31 tokens × num_kv_heads × head_size × 2B
```

#### 3. 写入路径：量化时机

在 KVCache 写入时（`vllm/v1/worker/gpu_model_runner.py` 或 attention backend 内部）：

```
新 token 写入 KVCache：
  K：直接 MXFP8 量化写入（head_dim 对齐，无尾块）
  V：
    ├── seq_len 能被 32 整除的部分 → MXFP8 量化写入
    └── 不足 32 的部分 → FP16 原样写入（高精窗口）
```

量化 kernel 需要支持：
- 沿指定轴（head_dim 或 seq_len）的 block-wise MXFP8 量化
- 输出：FP8 数据 + FP8 shared scale

#### 4. Attention backend 适配（核心难点）

需要修改 attention kernel 以支持混合精度 KVCache：

```
S = Q @ K^T：
  K 全部为 MXFP8 → 标准 MX GEMM，无特殊处理

P = softmax(S)

O = P @ V：
  V 混合精度 → 拆分计算
  O = P[:, :aligned] @ V_mxfp8[:aligned, :]   ← MX Tensor Core 通路
    + P[:, aligned:] @ V_fp16[aligned:, :]       ← FP16 通路
```

对于 vLLM 上游的 attention backend（FlashInfer / FlashAttention）：

| Backend | 现状 | 需要的改动 |
|---------|------|-----------|
| FlashInfer | 支持 FP8 KVCache（per-token scale） | 需扩展为 block-wise scale（每 32 token 共享） |
| FlashAttention | 支持 FP8 KVCache（per-token scale） | 同上 |
| 自定义 Triton kernel | 灵活 | 可实现混合精度路径，但性能需验证 |

#### 关键风险

1. **Attention kernel 改动量大**：上游 FlashInfer/FlashAttention 的 MXFP8 block-wise scale 支持需要上游社区配合，或自行 fork 修改
2. **vllm ascend 适配工作量仍存在**：vllm ascend使用自己独立的attention后端，kvcache量化主要影响和修改点都在这里，上游支持后，这部分适配仍需要搞

### 4.2 vllm-ascend 实现设计

#### 设计选择：独立 Tail Buffer vs KVCache Block Pool 管理

| 维度 | 独立 Tail Buffer | KVCache Block Pool（参考 DSV4 SWA） |
|------|-----------------|-------------------------------------|
| 存储粒度 | 每请求最多 31 token，精确分配 | block_size=128 token 为最小单位 |
| 空间浪费 | 无（按需使用） | 每请求浪费 97+ token 位（128-31） |
| 生命周期 | 临时累加，满 32 即量化清空 | 持久存储，随滑窗驱逐 |
| 管理复杂度 | 低（slot 表 + 长度数组） | 高（需 sub-block 偏移管理） |
| 与 block pool 关系 | 独立，不影响 packed layout | 需修改 block_stride，影响所有 cache 类型 |

**结论：采用独立 Tail Buffer**。尾块本质是量化块的临时累加缓冲区，不是持久缓存，与 DeepSeek V4 的 SWA（持久滑动窗口缓存）有本质区别，不适合纳入 block pool 管理。

#### 整体架构

```
                    ┌──────────────────────────────────┐
                    │         Scheduler                 │
                    │  slot_table: req_id → slot_id     │
                    │  tail_len[slot_id]: int           │
                    └──────────┬───────────────────────┘
                               │ slot_id, tail_len
                               ▼
┌──────────────────────────────────────────────────────────┐
│                    Per Layer                              │
│                                                          │
│  ┌─────────────────────┐    ┌──────────────────────────┐ │
│  │  KVCache Block Pool │    │  Tail Buffer (独立分配)    │ │
│  │  (packed layout)    │    │  shape: (max_reqs, 32,   │ │
│  │                     │    │         num_kv_heads,     │ │
│  │  K: MXFP8 + scale  │    │         head_dim)         │ │
│  │  V: MXFP8 + scale  │    │  dtype: FP16              │ │
│  │  (32对齐的部分)     │    │  (仅 V 的尾部 FP16 数据)  │ │
│  └─────────────────────┘    └──────────────────────────┘ │
│           │                          │                    │
│           └──────────┬───────────────┘                    │
│                      ▼                                    │
│              FIA Attention 算子                           │
│         O = P[:,:aligned] @ V_mxfp8                      │
│           + P[:,aligned:] @ V_tail_fp16                   │
└──────────────────────────────────────────────────────────┘
```

#### Tail Buffer 定义

```python
class MXFP8TailBuffer:
    """每层一个，存储 V cache 的 FP16 尾部数据"""
    def __init__(self, max_num_reqs, num_kv_heads, head_dim, device):
        # 预分配：max_num_reqs 个 slot，每个 slot 最多 32 个 token
        self.buffer = torch.zeros(
            (max_num_reqs, 32, num_kv_heads, head_dim),
            dtype=torch.float16, device=device
        )
        self.free_slots = list(range(max_num_reqs))

    def allocate(self) -> int:
        return self.free_slots.pop()

    def free(self, slot_id: int):
        self.buffer[slot_id].zero_()
        self.free_slots.append(slot_id)
```

仅 V cache 需要 tail buffer，K cache 沿 head_dim 量化无尾块。

#### 请求与 Tail Buffer 的映射

##### Slot 分配

```
Scheduler 准入请求时：
  slot_id = tail_buffer.allocate()          # 从 free_slots 取一个
  slot_table[req_id] = slot_id              # 记录映射关系

请求完成/抢占时：
  slot_id = slot_table.pop(req_id)
  tail_buffer.free(slot_id)                 # 归还 slot
```

##### Slot Table 机制详解

`slot_table` 是新增的轻量级映射表，本质是 `dict[req_id, int]` 或固定大小数组，用于解耦逻辑请求与物理存储位置。

**与现有机制的对比**：

| 机制 | 粒度 | 来源 | 用途 |
|------|------|------|------|
| `block_table` | per-block | 已有 | 逻辑 block → 物理 block 映射 |
| `slot_mapping` | per-token | 已有 | token 在 block pool 中的物理位置 |
| `tail_slot_table` | per-request | **新增** | 请求 → tail buffer slot 映射 |

**工作原理**：

```
Tail Buffer（预分配 max_reqs=24 个 slot）：

slot_id:  [0][1][2][3][4][5]...[23]
           ↑  ↑      ↑
           │  │      │
           │  │      └── req_C 的 tail 数据
           │  └───────── req_B 的 tail 数据
           └──────────── req_A 的 tail 数据

slot_table（字典，per-request）：
  req_A → slot_id=0
  req_B → slot_id=1
  req_C → slot_id=3
```

**请求执行时**（所有层共享同一个 slot_id）：
```python
# Layer 0: tail_buffer_layer0[slot_id=0][tail_len] = new_v_token
# Layer 1: tail_buffer_layer1[slot_id=0][tail_len] = new_v_token
# ...
# Layer 60: tail_buffer_layer60[slot_id=0][tail_len] = new_v_token
```

**Batch 执行时**（请求顺序与 slot 顺序无关）：
```python
# 当前 batch: [req_A, req_C, req_B]
slot_ids = [slot_table[r] for r in batch_req_ids]  # [0, 3, 1]
tail_lens = [tail_len[r] for r in batch_req_ids]   # [14, 7, 22]

# FIA 算子收到：
# tail_buffer[slot_ids[0]=0][:14]  → req_A 的尾块
# tail_buffer[slot_ids[1]=3][:7]   → req_C 的尾块
# tail_buffer[slot_ids[2]=1][:22]  → req_B 的尾块
```

**设计要点**：
- slot_id 是随机分配的，不要求连续，请求顺序与物理存储顺序解耦
- 同一请求在所有层使用相同的 slot_id，保证跨层一致性
- 维护成本极低，可挂在 scheduler state 或 attention metadata builder 中

##### Metadata 传递

```python
# 调度器维护（per-request 粒度，所有层共享）
tail_len_per_request: dict[req_id, int]  # 当前尾块中的 token 数

# 构建 attention metadata 时（per-layer）
class MXFP8AttentionMetadata:
    tail_buffer: torch.Tensor       # 当前层的 tail buffer 引用
    tail_lens: torch.Tensor         # [num_reqs]，每个请求的 tail_len
    aligned_seq_lens: torch.Tensor  # [num_reqs]，seq_len - tail_len
    slot_ids: torch.Tensor          # [num_reqs]，请求对应的 slot_id
```

`tail_len` 是 per-request 状态，所有层共享同一个值（同一请求在所有层的尾块长度相同）。

#### Chunked Prefill 流程

```
输入：chunk_size 个新 token，当前 tail_len = old_tail

Step 1: 拼接
  total = old_tail + chunk_size

Step 2: 拆分
  new_aligned_tokens = (total // 32) * 32
  new_tail_len = total % 32

Step 3: 量化写入
  将 new_aligned_tokens 个 V token 做 MXFP8 量化 → 写入 KVCache block pool
  （K 的对应 token 同时量化写入，K 无尾块）

Step 4: 更新 tail buffer
  将剩余 new_tail_len 个 V token 写入 tail_buffer[slot_id][:new_tail_len]

Step 5: 更新状态
  tail_len[req_id] = new_tail_len
```

示例：`old_tail=10, chunk_size=100`

```
total = 110
aligned = 96 (3 blocks × 32)  → 量化写入 block pool
new_tail = 14                  → 写入 tail_buffer[slot_id][:14]
```

#### Decode 流程

```
每个 decode step，1 个新 token：

Step 1: 追加
  tail_buffer[slot_id][tail_len] = new_v_token  (FP16)
  tail_len += 1

Step 2: 检查是否满 32
  if tail_len == 32:
      量化 tail_buffer[slot_id][:32] → MXFP8
      写入 KVCache block pool（V 的一个完整 block）
      K 的对应 32 个 token 同时量化写入（K 的一个完整 block）
      tail_len = 0

Step 3: 更新状态
  tail_len[req_id] = tail_len
```

#### 尾块满 32 时的 Block 分配

K 和 V 的 block 始终同步填充（都按 seq_len 推进），区别仅在于 V 有 tail 延迟写入：

```
seq_len:  [0--------32][32-------64][64-------96][96---110]
K block:  [  block 0  ][  block 1  ][  block 2  ][ tail  ]  ← K 无尾块，直接量化
V block:  [  block 0  ][  block 1  ][  block 2  ][ tail  ]  ← V 的 block 0/1/2 在 tail 满 32 时写入
```

当 tail 满 32 时，从 block pool 分配一个新 block，同时存放 K 和 V 的量化数据。K 和 V 的 block 数量始终一致。

#### FIA 算子集成

框架侧负责管理 tail buffer，FIA 算子内部处理混合精度计算：

```python
# 框架侧构建 attention 调用
attn_output = fia_attention(
    q,                              # query
    kv_cache=kv_cache_block_pool,   # MXFP8 KVCache（32 对齐部分）
    tail_buffer=tail_buffer.buffer, # FP16 V 尾块 buffer
    tail_lens=tail_lens,            # 每个请求的 tail_len
    block_table=block_table,        # block 映射表
    aligned_seq_lens=aligned_seq_lens,  # 每个请求的 aligned seq_len
    ...
)

# FIA 算子内部：
# S = Q @ K^T                        ← K 全量 MXFP8，标准计算
# P = softmax(S)
# O_mxfp8 = P[:, :aligned] @ V_mxfp8 ← MX Tensor Core 通路
# O_tail  = P[:, aligned:] @ V_tail   ← FP16 通路（tail_len < 32，开销极低）
# O = O_mxfp8 + O_tail
```

框架侧不感知 FIA 内部的拆分逻辑，只需传递 tail buffer 和相关 metadata。

#### 内存开销

```
Tail Buffer 大小 = max_num_reqs × 32 × num_kv_heads × head_dim × sizeof(FP16)

典型配置（max_reqs=24, num_kv_heads=8, head_dim=128）：
  = 24 × 32 × 8 × 128 × 2B = 3,145,728 B ≈ 3 MB / layer

61 层模型：3 MB × 61 = 183 MB
```

相比 KVCache block pool（通常数 GB），tail buffer 开销可忽略。

#### 抢占与 Swap

| 场景 | 处理 |
|------|------|
| 请求抢占 | slot 保持分配，tail 数据保留在 buffer 中，恢复时直接使用 |
| 请求 Swap 到 CPU | 将 `tail_buffer[slot_id][:tail_len]` 拷贝到 CPU，slot 可释放 |
| 请求 Swap 回 GPU | 从 CPU 恢复 tail 数据到 slot，恢复 tail_len |
| 请求完成 | 释放 slot，清零 buffer |

#### PD 分离

Prefill worker 完成 prefill 后，需要将 tail buffer 数据传输给 Decode worker：

```
传输内容：
  1. KVCache blocks（MXFP8 量化数据 + scale）
  2. tail_buffer[slot_id][:tail_len]（FP16，最多 31 token）
  3. tail_len 值

传输量：tail 部分最多 31 × num_kv_heads × head_dim × 2B ≈ 126 KB / layer / request
```

K 和 V 的 block 数量一致（tail 满 32 时同步写入），但 V 可能有一个未满的 tail 需要额外传输。

---

## 5.特性叠加

### 5.1 prefix cache

block_size 默认128/FIA算子要求512，可以整除32（64：dynamic-mx-quant的粒度），prefix cache按block粒度做前缀匹配，初步分析尾块不涉及，需叠加验证功能是否正常

### 5.2 PD分离

新增k_scale cache, v_scale cache，v cahce 高精窗口（尾块）的传输

k v传输的block num数量会出现差异，v=k-1

TP不对等场景需要单独适配（存储格式改为BNBD，适配起来会好一些）

### 5.3 池化

新增k_scale cache, v_scale cache，v cahce 高精窗口（尾块）的池化

池化按block粒度存储，v cahce 高精窗口不再此范围内，初步分析不需要额外做池化

### 5.4 PCP/DCP

c8叠加PCP/DCP的必要性

DCP切kvcache，感知尾块，需要做额外适配，方案需和特性owner一起分析



## 6.其他方案考虑

MXFP动态+FP静态混合量化：Q/K走MXFP8动态量化，V走FP8静态量化

优势：
根源消除尾块问题，框架亲和，框架侧改动量小很多

劣势：

精度是否可以保证

需单独量化C8权重，v scale需在权重量化过程中直接生成，保存在权重中

FIA算子性能，这种方案相比MXFP8全量化会有多少差异
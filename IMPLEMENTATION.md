# Nano-vLLM-Pro 实现过程文档

本文档详细记录了 Roadmap.md 中 5 个 Phase 的实现细节、设计决策和验证方法。

---

## Phase 1: Triton 算子适配

### 实现概述

将 `layers/` 下依赖 `@torch.compile` 和 PyTorch 原生算子的模块替换为手写 Triton kernel，减少 kernel launch 开销和显存读写次数。

### 1.1 Fused RMSNorm（P0）

**文件**: `nanovllm/layers/triton_rmsnorm.py`, `nanovllm/layers/layernorm.py`

**原始实现**:
```python
@torch.compile
def rms_forward(self, x):
    orig_dtype = x.dtype
    x = x.float()
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x.mul_(torch.rsqrt(var + self.eps))
    x = x.to(orig_dtype).mul_(self.weight)
    return x
```

**Triton 实现**:
- 单个 Triton kernel `_rms_norm_kernel` 处理纯 norm 和 fused add+norm 两种情况
- 通过 `HAS_RESIDUAL` constexpr 控制是否执行 residual add
- 每个 program 处理一行（一个 token），并行度 = num_tokens
- 支持 float32 精度计算，自动 cast 回原始 dtype
- BLOCK_SIZE 根据 hidden_size 自适应调整，num_warps 随之调整

**接口**:
- `triton_rms_norm(x, weight, eps)` → 纯 RMSNorm
- `triton_add_rms_norm(x, residual, weight, eps)` → Fused residual add + RMSNorm

**双路径切换**: `RMSNorm.forward()` 根据 `use_triton` 和 `x.is_cuda` 自动选择 Triton 或 torch.compile 路径。

### 1.2 Fused SiLU+Mul（P0）

**文件**: `nanovllm/layers/triton_activation.py`, `nanovllm/layers/activation.py`

**原始实现**:
```python
@torch.compile
def forward(self, x):
    x, y = x.chunk(2, -1)
    return F.silu(x) * y
```

**Triton 实现**:
- 单个 kernel `_silu_and_mul_kernel`
- 每个 program 处理一行，从输入中加载前半（gate）和后半（up）
- 在 kernel 内计算 `sigmoid(gate) * gate * up`，避免中间 tensor 写回显存
- 输出维度 = 输入维度 / 2

**关键设计**: 使用 `tl.exp(-gate)` 计算 sigmoid，避免 `tl.sigmoid`（某些 Triton 版本不支持）。

### 1.3 Fused RoPE（P1）

**文件**: `nanovllm/layers/triton_rotary_embedding.py`, `nanovllm/layers/rotary_embedding.py`

**原始实现**:
```python
@torch.compile
def forward(self, positions, query, key):
    cos_sin = self.cos_sin_cache[positions]
    cos, sin = cos_sin.chunk(2, dim=-1)
    query = apply_rotary_emb(query, cos, sin)
    key = apply_rotary_emb(key, cos, sin)
    return query, key
```

**Triton 实现**:
- 2D grid: `(num_tokens, num_heads)`，每个 program 处理一个 token 的一个 head
- 原地旋转：直接在 Q/K buffer 上修改，不需要额外分配
- 加载 cos/sin 从预计算缓存，按 positions 索引
- 将 head_dim 分为两半，执行旋转操作：`y1 = x1*cos - x2*sin, y2 = x2*cos + x1*sin`
- `rotary_dim_half` 作为 constexpr 传入，优化编译

**关键设计**: `positions` 作为输入张量传入，而非在 kernel 内生成，与现有调度逻辑一致。

### 1.4 Fused Sampler（P1）

**文件**: `nanovllm/layers/triton_sampler.py`, `nanovllm/layers/sampler.py`

**原始实现**: 仅支持 Gumbel-max trick 的 temperature 采样。

**Triton 实现**:
- `_argmax_kernel`: Greedy 采样（argmax）
- `_temperature_sampling_kernel`: Temperature 采样（Gumbel-max trick）
- `triton_top_k_sampling`: Top-K 采样（PyTorch 预过滤 + Triton Gumbel-max）
- `triton_top_p_sampling`: Top-P 采样（PyTorch 累积概率过滤 + Triton Gumbel-max）

**采样策略**:
- `temperature=0.0` → Greedy（argmax）
- `top_k > 0` → Top-K 过滤后采样
- `top_p < 1.0` → Top-P (nucleus) 过滤后采样
- 默认 → Temperature 采样

**Gumbel-max trick**: 使用简单的 LCG 伪随机数生成器，避免 Triton 中缺乏内置 RNG 的限制。种子每次采样时随机生成，确保不同 step 的结果不同。

**SamplingParams 扩展**:
- 新增 `top_k: int = -1` 和 `top_p: float = 1.0`
- `temperature=0.0` 现在表示 greedy 模式（原始实现禁止 greedy）

---

## Phase 2: CPU Offload

### 实现概述

支持将 KV-Cache 页块卸载到 CPU 内存，在 GPU 显存不足时扩展可服务序列数。

### 2.1 BlockManager 扩展

**文件**: `nanovllm/engine/block_manager.py`

新增字段:
- `cpu_pool`: CPU pinned memory tensor，形状 `[2, num_layers, num_cpu_blocks, block_size * num_kv_heads * head_dim]`
- `cpu_block_table`: `dict[int, int]`，GPU block_id → CPU block_id 映射
- `offloaded_blocks`: `set[int]`，已卸载到 CPU 的 GPU block ID 集合
- `offload_stream`: CUDA Stream，用于异步 H2D/D2H 传输

新增方法:
- `should_offload(threshold)`: GPU 空闲块比例低于阈值时返回 True
- `offload_blocks(seq, kv_cache, num_blocks)`: 将序列最早的 blocks 异步复制到 CPU
- `reload_blocks(seq, kv_cache)`: 将序列已卸载的 blocks 从 CPU 异步搬回 GPU
- `gpu_free_ratio`: GPU 空闲块比例

### 2.2 卸载策略

- 当 `enable_cpu_offload=True` 且 `gpu_free_ratio < threshold` 时触发
- 优先卸载最早进入 running 队列的序列的非活跃 blocks（排除最后一个 block）
- 仅卸载 `ref_count <= 1` 的 blocks（避免影响共享缓存）
- 使用 `torch.cuda.Stream` 进行异步复制，与主计算流不冲突
- reload 时调用 `current_stream().wait_stream(offload_stream)` 确保 attention 前数据就绪

### 2.3 Config 扩展

- `enable_cpu_offload: bool = False`
- `cpu_offload_threshold: float = 0.1`

---

## Phase 3: 标准化 Benchmark

### 实现概述

建立可复现的性能评测框架，覆盖关键推理指标。

### 3.1 Sequence 时间戳字段

**文件**: `nanovllm/engine/sequence.py`

新增:
- `arrival_time`: 请求到达时间（`Sequence.__init__` 中记录）
- `first_token_time`: 首 token 生成时间（`LLMEngine.step()` 中记录）
- `finish_time`: 序列完成时间
- `num_decode_tokens`: decode token 计数
- `total_decode_time`: decode 总耗时

### 3.2 LLMEngine 埋点

**文件**: `nanovllm/engine/llm_engine.py`

- 在 `step()` 中记录：
  - Prefill 完成后的 `first_token_time`
  - 序列完成时的 `finish_time`
  - 每步的 prefill/decode TPS
  - 抢占计数
- 新增 `get_benchmark_stats()` 返回统计信息

### 3.3 Benchmark 框架

**文件**: `benchmark/metrics.py`, `benchmark/bench_runner.py`

**指标体系**:
| 指标 | 含义 | 采集方式 |
|:---|:---|:---|
| TTFT | 首 token 延迟 | first_token_time - arrival_time |
| TPS | 总 decode 吞吐 | 总输出 token / 总 decode 时间 |
| TPS per request | 单请求吞吐 | 输出 token / 该请求 decode 时间 |
| E2E Latency | 端到端延迟 | finish_time - arrival_time |
| Latency Percentiles | P50/P90/P99 | `statistics.quantiles()` |

**BenchmarkCollector**: 收集 per-request 指标，聚合为 `BenchmarkResult`。

**输出格式**: JSON + Markdown 表格。

---

## Phase 4: 调度层重构 — 消除 Prefill/Decode 震荡

### 实现概述

将 Prefill 和 Decode 从互斥调度改为混合调度，每个 step 同时调度 prefill 和 decode 序列。

### 4.1 问题分析

原始 `Scheduler.schedule()` 的逻辑：
```text
while waiting:  # 优先 prefill
    ...schedule prefill...
if scheduled: return prefill
# then decode
while running:
    ...schedule decode...
```

导致震荡：
```text
Prefill 填满显存 → Decode 发现显存不足 → 抢占释放 block →
释放的显存立即被新一轮 Prefill 占满 → Decode 再次抢占 → 循环
```

### 4.2 混合调度方案

**文件**: `nanovllm/engine/scheduler.py` - `_schedule_mixed()`

**核心逻辑**:
1. **Decode 优先**：先为当前 running 序列预留 decode 所需的 token budget 和 block 空间
2. **Prefill 使用剩余预算**：在确保 decode 不会因显存不足而被抢占的前提下，调度新的 prefill 请求
3. **返回结构不变**：仍然返回 `(seqs, is_prefill)`，但 seqs 中同时包含 prefill 和 decode 序列

**关键变化**:
- decode 序列先调度，确保不被抢占
- prefill 使用剩余 token budget，不会侵占 decode 空间
- 每个 step 的 `is_prefill` 标志仅在有 prefill 序列且无 decode 序列时为 True

### 4.3 ModelRunner 适配

**文件**: `nanovllm/engine/model_runner.py` - `_run_mixed()`

对于混合 step：
1. 分离 prefill_seqs 和 decode_seqs
2. 分别执行 prefill forward 和 decode forward
3. 分别采样
4. 按 all_seqs 的原始顺序合并 token_ids

**设计决策**: 使用两次 forward pass 而非尝试将 prefill/decode 合并到同一个 batch。原因：
- Prefill 使用 `flash_attn_varlen_func`，Decode 使用 `flash_attn_with_kvcache`，注意力模式不同
- 两次 forward 的额外开销远小于震荡造成的吞吐下降
- 实现简单、正确性容易保证

### 4.4 震荡复现脚本

**文件**: `benchmark/oscillation_repro.py`

构造场景：256 个请求，prompt 长度 512，max_tokens 256，单 GPU 运行。

记录每步的：
- `is_prefill` 标志
- 抢占次数
- running/waiting 序列数
- 空闲 block 数

输出震荡分析：总抢占次数、模式切换次数、震荡评分。

### 4.5 Config 扩展

- `enable_mixed_scheduling: bool = True`
- `prefill_budget_ratio: float = 0.5`

---

## Phase 5: Chunked Prefill 优化

### 实现概述

将长 prompt 的 prefill 拆分为多个 chunk 分步执行，避免单个长 prompt 独占整个 step 的 token budget。

### 5.1 实现设计

**文件**: `nanovllm/engine/scheduler.py` - `_schedule_chunked_prefill()`

**Chunk 粒度**: 以 `block_size`（256 tokens）为单位。

**调度逻辑**:
1. 计算 prefill_budget（= 总 budget - decode 已用 budget）
2. 遍历 waiting 队列，为每个序列分配 `min(remaining_budget, chunk_size, num_tokens)` 个 token
3. 已完全 prefilled 的序列从 waiting 移到 running
4. 未完成的序列保留在 waiting 队列，下一步继续调度

**公平调度**: 多个 waiting 序列轮流分配 chunk，而非单个序列独占。短 prompt 可以在一个 step 内完成，长 prompt 被拆分到多个 step。

**与 Phase 4 联动**: chunked prefill 在混合调度框架下运行，每个 step 的 prefill_budget 同时服务多个序列的 chunk。

### 5.2 前缀缓存复用

已经缓存的 block 不重复计算——`num_cached_blocks` 在 `can_allocate()` 和 `allocate()` 中正确计算，仅 prefill 新增部分。

### 5.3 Config 扩展

- `enable_chunked_prefill: bool = True`

---

## 验证方法

### Phase 1 验证

```python
# 验证 Triton RMSNorm 数值正确性
import torch
from nanovllm.layers.triton_rmsnorm import triton_rms_norm, triton_add_rms_norm

x = torch.randn(4, 512, device='cuda')
weight = torch.ones(512, device='cuda')
y = triton_rms_norm(x, weight)  # 应与 torch.compile 版本结果一致
```

### Phase 3 验证

```bash
python -m benchmark.bench_runner --model ~/huggingface/Qwen3-0.6B/ --num-seqs 32
# 输出包含 TTFT/TPS/E2E/Latency 的 JSON 和 Markdown 报告
```

### Phase 4 验证

```bash
# 原始调度（震荡）
python -m benchmark.oscillation_repro --model ~/huggingface/Qwen3-0.6B/
# 混合调度（无震荡）
# 设置 enable_mixed_scheduling=True 后运行同一脚本
```

验证标准：
- 震荡场景下抢占次数降为 0 或接近 0
- Decode TPS 不再出现周期性抖动

### Phase 5 验证

构造长短 prompt 混合场景：
- 短 prompt (128 tokens) 和长 prompt (2048 tokens) 同时请求
- 验证短 prompt 的 TTFT 不被长 prompt 阻塞
- 同一 step 内可同时为多个序列推进 prefill

---

## 文件变更清单

| 文件 | 操作 | Phase |
|:---|:---|:---|
| `nanovllm/layers/triton_rmsnorm.py` | 新增 | Phase 1 |
| `nanovllm/layers/triton_activation.py` | 新增 | Phase 1 |
| `nanovllm/layers/triton_rotary_embedding.py` | 新增 | Phase 1 |
| `nanovllm/layers/triton_sampler.py` | 新增 | Phase 1 |
| `nanovllm/layers/layernorm.py` | 修改（增加 Triton 路径） | Phase 1 |
| `nanovllm/layers/activation.py` | 修改（增加 Triton 路径） | Phase 1 |
| `nanovllm/layers/rotary_embedding.py` | 修改（增加 Triton 路径） | Phase 1 |
| `nanovllm/layers/sampler.py` | 修改（多策略 + Triton 路径） | Phase 1 |
| `nanovllm/sampling_params.py` | 修改（增加 top_k/top_p，允许 greedy） | Phase 1 |
| `nanovllm/engine/block_manager.py` | 修改（CPU Offload） | Phase 2 |
| `nanovllm/engine/sequence.py` | 修改（时间戳字段） | Phase 3 |
| `nanovllm/engine/llm_engine.py` | 修改（Benchmark 采集） | Phase 3 |
| `benchmark/__init__.py` | 新增 | Phase 3 |
| `benchmark/metrics.py` | 新增 | Phase 3 |
| `benchmark/bench_runner.py` | 新增 | Phase 3 |
| `nanovllm/engine/scheduler.py` | 修改（混合调度 + Chunked Prefill） | Phase 4/5 |
| `nanovllm/engine/model_runner.py` | 修改（混合调度支持） | Phase 4 |
| `nanovllm/config.py` | 修改（新增配置项） | Phase 2/4/5 |
| `benchmark/oscillation_repro.py` | 新增 | Phase 4 |
| `nanovllm/models/qwen3.py` | 修改（use_triton 参数） | Phase 1 |
| `nanovllm/__init__.py` | 修改（导出 Config） | 全局 |
| `README.md` | 重写 | 文档 |
| `IMPLEMENTATION.md` | 新增 | 文档 |

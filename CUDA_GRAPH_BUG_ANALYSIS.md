# CUDA Graph 报错分析：Triton Kernel 非法内存访问

## 报错现象

运行 `python bench.py --use_triton` 时，在 `capture_cudagraph()` 的 warmup 阶段即崩溃：
```
RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered
```

精确报错位置（`CUDA_LAUNCH_BLOCKING=1` 确认）：
```
nanovllm/layers/triton_rotary_embedding.py:107 → _rotary_embedding_kernel[grid]
```

调用链：
```
ModelRunner.capture_cudagraph()  →  warmup  →  Qwen3Model.forward()
  → Qwen3DecoderLayer.forward()
    → Qwen3Attention.forward()
      → RotaryEmbedding.__call__()  [走 Triton 路径]
        → triton_apply_rotary_emb()
          → _rotary_embedding_kernel[grid]  ← 💥 非法内存访问
```

**注意**：即使不传 `--use_triton`，`Config` 中 `use_triton` 默认值也是 `True`，所以同样会崩溃。

---

## 根因分析

### 问题 1（致命）：Triton RoPE kernel 的 cos/sin 缓存索引越界

**`triton_rotary_embedding.py` 第 34-36 行**：
```python
cos_ptrs = COS_PTR + pos * cos_stride + half_offsets
sin_ptrs = SIN_PTR + pos * sin_stride + half_offsets
cos = tl.load(cos_ptrs, mask=mask, other=0.0).to(tl.float32)
sin = tl.load(sin_ptrs, mask=mask, other=0.0).to(tl.float32)
```

而 `triton_apply_rotary_emb()` 第 91-93 行对 `cos_sin_cache` 做了切分：
```python
cos_sin = cos_sin_cache.squeeze(1)  # [max_pos, head_dim]
cos = cos_sin[:, :rotary_dim_half]  # [max_pos, rotary_dim_half]
sin = cos_sin[:, rotary_dim_half:]  # [max_pos, rotary_dim_half]
```

**关键问题**：`cos` 和 `sin` 是从 `cos_sin_cache` 切出来的**非连续视图（non-contiguous view）**。

`cos_sin_cache` 的 shape 是 `[max_position_embeddings, 1, head_dim]`，squeeze 后变成 `[max_pos, head_dim]`，内存布局是 **行优先连续**（每行 head_dim 个元素连续存储）。

但 `cos = cos_sin[:, :rotary_dim_half]` 取的是前半列，`sin = cos_sin[:, rotary_dim_half:]` 取的是后半列——这两个子张量在内存中**不是连续的**，它们是 `cos_sin` 的列切片视图。

**在 kernel 中**，代码传递了 `cos_f.stride(0)` 作为行步长，但 `cos_f` 是 float32 cast 后的**非连续张量**——`.float()` 会对非连续输入做一次 copy，使其变成连续的，但此时**行步长 `cos_f.stride(0)` 变成了 `rotary_dim_half`，而不是原始的 `head_dim`**。

然而 kernel 里 `COS_PTR + pos * cos_stride + half_offsets` 使用 `cos_stride` 作为行步长，如果 `cos_f` 经过 `.float()` 后变连续，那 `cos_stride = rotary_dim_half` 是正确的。

**等等——让我重新审视**。实际上 `.float()` 会创建新张量，对于非连续视图会做 copy 使其连续。所以 cos 的 stride(0) = rotary_dim_half，这和 kernel 中的寻址逻辑是匹配的。

**那真正的越界在哪？** 看 `positions` 的值！

`capture_cudagraph()` 中第 229 行：
```python
positions = torch.zeros(max_bs, dtype=torch.int64)
```

**所有 position 都是 0**——这不会越界。

但是 warmup 阶段（`warmup_model()`）中：
```python
seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
for seq in seqs:
    seq.num_scheduled_tokens = seq_len
self.run(seqs, True)  # is_prefill=True
```

Prefill 阶段 position 范围是 `0 ~ seq_len-1`，`seq_len = min(max_num_batched_tokens, max_model_len)`，最大为 16384。而 `cos_sin_cache` 的行数是 `max_position_embeddings`，对 Qwen3-8B 通常是 32768 或 131072，所以 prefill warmup 也不越界。

**让我再仔细看**——问题可能不在 cos/sin 越界，而在 **Q/K 的寻址**。

### 问题 2（致命）：Triton RoPE kernel 中 Q/K 的 stride 传递错误

`triton_apply_rotary_emb()` 第 107-115 行：
```python
_rotary_embedding_kernel[grid](
    q,
    q.stride(0), q.stride(1), q.stride(2),
    k,
    k.stride(0), k.stride(1), k.stride(2),
    ...
)
```

**Q 和 K 经过了 `.float()` 转换**（第 98-99 行）：
```python
q = query.float()    # query shape: [num_tokens, num_heads, head_dim]
k = key.float()
```

`query` 和 `key` 的原始 shape 是 `[num_tokens, num_heads, head_dim]`，来自 `qkv.split()` + `.view()` 操作。**在 CUDA Graph 的 warmup 阶段**，`positions = torch.zeros(max_bs)`，所以 `num_tokens = bs`（batch size）。

**关键问题**：`q = query.float()` 这个操作会创建一个**新的连续张量**（因为 `.float()` 对已经是连续的张量做 dtype cast，保持连续性和 stride 不变）。所以 `q.stride(0) = num_heads * head_dim`, `q.stride(1) = head_dim`, `q.stride(2) = 1`，这和 kernel 的寻址逻辑匹配。

**但是！** kernel 中的寻址是：
```python
q_base = Q_ptr + token_idx * Q_stride_0 + head_idx * Q_stride_1
q1_ptrs = q_base + half_offsets           # 前半
q2_ptrs = q_base + ROTARY_DIM_HALF + half_offsets  # 后半
```

这里 `ROTARY_DIM_HALF = head_dim // 2`，`q1_ptrs` 取的是 `[token, head, 0:half]`，`q2_ptrs` 取的是 `[token, head, half:head_dim]`——对于完整 RoPE（rotary_dim == head_dim），这是正确的。

**但等等——Qwen3-8B 的 head_dim 是多少？** 对 Qwen3-8B：`hidden_size=4096, num_attention_heads=32, head_dim=128`。所以 `rotary_dim_half = 64`，`BLOCK_SIZE = 64`（next_power_of_2(64) = 64）。

**那问题到底在哪？** 让我检查 `k` 的 num_heads——Qwen3 使用 GQA，`num_kv_heads` 可能 < `num_heads`。

### 问题 3（致命）：Grid 维度与 Q/K 的 head 数量不匹配

`triton_apply_rotary_emb()` 第 105 行：
```python
grid = (num_tokens, num_heads)
```

这里 `num_heads = query.shape[1]`——取的是 **Q 的 head 数**。

但 **K 的 head 数 = num_kv_heads**，通常比 Q 的 num_heads 小（GQA）。Qwen3-8B 中：
- `num_attention_heads = 32`（Q heads）
- `num_key_value_heads = 8`（K/V heads）

**kernel 用同一个 `head_idx = tl.program_id(1)` 去索引 Q 和 K**：
```python
q_base = Q_ptr + token_idx * Q_stride_0 + head_idx * Q_stride_1   # head_idx 0..31
k_base = K_ptr + token_idx * K_stride_0 + head_idx * K_stride_1   # head_idx 0..31
```

**当 `head_idx >= 8`（num_kv_heads）时，`k_base` 就会越界！** 这就是非法内存访问的根因。

K 的 shape 是 `[num_tokens, num_kv_heads, head_dim]` = `[bs, 8, 128]`，总元素数只有 `bs * 8 * 128`。但 kernel 用 `head_idx=8..31` 去寻址，直接越界。

**这是确定的根因。** Qwen3 使用 GQA（Grouped Query Attention），Q 有 32 个头，K/V 只有 8 个头，Triton kernel 用同一个 grid 遍历 Q 的头数去索引 K，导致 K 的越界访问。

---

## 其他次要问题

### 问题 4（中等）：RMSNorm Triton kernel 的 `__call__` 覆盖

`RMSNorm` 中重写了 `__call__`，在 Triton 路径直接返回结果而不经过 `nn.Module.__call__`，**跳过了 hooks 和 `torch.compile` 的追踪**。在 CUDA Graph capture 时，这可能导致：
- Triton kernel 调用不被 graph 追踪（但 Triton kernel 本身是 CUDA kernel，graph 可以捕获）
- 但如果 Triton kernel 中间创建了新张量（如 `torch.empty_like`），这些动态分配在 graph replay 时会有问题

**实际上**，`triton_rms_norm()` 每次调用都会创建 `torch.empty_like(x_2d)` 作为输出——在 CUDA Graph capture 时，这个新张量的创建会被记录到 graph 中，但 **graph replay 时不会重新分配内存**，因为 graph 捕获的是整个执行流程的快照。但前提是每次 graph replay 的输入 tensor 地址不变——这在 `capture_cudagraph` 的设计中是通过固定 `graph_vars` 的内存来保证的。

**但问题是**：Triton kernel 内部创建的中间张量（`y = torch.empty_like(...)`）在 graph capture 阶段会被记录，**replay 时这些中间张量的地址不变**，所以 RMSNorm 和 Activation 的 Triton 路径在 CUDA Graph 下理论上是可行的。

### 问题 5（中等）：SiluAndMul 重写 `__call__` 的风险

与 RMSNorm 类似，`SiluAndMul` 重写了 `__call__` 来走 Triton 路径。`triton_silu_and_mul()` 内部也创建了 `torch.empty(...)` 输出张量。同上分析，CUDA Graph capture 下可行，但需要确保中间张量的生命周期和地址在 replay 时一致。

### 问题 6（低风险）：`use_triton` 默认值为 `True`

`Config.use_triton` 默认是 `True`，这意味着**即使不传 `--use_triton` 参数，也会走 Triton 路径**。bench.py 中 `--use_triton` 是 `action="store_true"`，默认 `False`，但这个 `False` 只传给了 `LLM(path, use_triton=use_triton)`——然而 `Config` 的 `use_triton` 字段默认值是 `True`，而 `kwargs` 中 `use_triton=False` 应该能覆盖默认值。

**等等，让我再看**：`LLM.__init__` → `LLMEngine.__init__` 中：
```python
config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
config = Config(model, **config_kwargs)
```
所以 `use_triton=False` 会被传入 Config。**但 warmup 阶段的错误已经先发生了**——因为 warmup 使用的是 `use_triton=True` 时的 Triton 路径。

不对——`use_triton=False` 时，`hf_config.use_triton = False`，所有层都不会走 Triton 路径，应该不会报错。但上面的测试输出显示**不传 `--use_triton` 时也崩溃了**，报错位置在 `capture_cudagraph` 的 `with torch.cuda.graph(graph, self.graph_pool):` 行——**这是 warmup 阶段 RoPE kernel 崩溃后 CUDA 状态已损坏导致的连锁反应**。

### 问题 7（低风险）：`positions` 的数据类型

`positions` 在 `capture_cudagraph` 中是 `torch.int64`（`torch.zeros(max_bs, dtype=torch.int64)`）。Triton RoPE kernel 中 `positions_ptr` 没有指定类型，但 `tl.load(positions_ptr + token_idx)` 会按默认 int32 读取。**int64 和 int32 的 stride 不同**，但这里 token_idx 只是一个偏移量，所以只影响值的解释。如果 position 值 > 2^31，会溢出——但 position 0 不会有问题。

---

## 问题总结

| # | 严重度 | 位置 | 问题 | 影响 |
|---|--------|------|------|------|
| 1 | **致命** | `triton_rotary_embedding.py:105` | grid 用 Q 的 head 数索引 K，GQA 下 K 越界 | CUDA illegal memory access → 进程崩溃 |
| 2 | 中等 | `layernorm.py:53` / `activation.py:19` | `__call__` 覆盖跳过 `nn.Module` 的 hook 机制 | CUDA Graph capture 时行为不确定 |
| 3 | 低 | `config.py:19` | `use_triton` 默认 `True`，与 bench.py 参数语义不一致 | 用户可能误以为不传 `--use_triton` 就不用 Triton |

---

## 修复建议

### 修复 1：RoPE Triton kernel 适配 GQA（必做）

**方案 A**：让 kernel 分别对 Q 和 K 使用不同的 grid 遍历——需要拆成两个 kernel 调用。

**方案 B（推荐）**：保持单 kernel，但增加 `NUM_Q_HEADS` / `NUM_KV_HEADS` 参数，在 kernel 内部对 K 的 head_idx 做映射：

```python
# 在 kernel 中
q_head_idx = head_idx
k_head_idx = head_idx * NUM_KV_HEADS // NUM_Q_HEADS  # GQA 映射
```

**方案 C（最简单）**：Q 和 K 分别调用 kernel，每个 kernel 只处理自己 head 数对应的维度：

```python
# Q: grid = (num_tokens, num_q_heads)
_rotary_embedding_kernel_q[(num_tokens, num_q_heads)](q, ..., NUM_HEADS=num_q_heads)

# K: grid = (num_tokens, num_kv_heads)
_rotary_embedding_kernel_k[(num_tokens, num_kv_heads)](k, ..., NUM_HEADS=num_kv_heads)
```

**推荐方案 C**——拆成两次独立调用，逻辑最清晰，不需要 GQA 映射，且与原始 PyTorch 实现的行为完全一致。

### 修复 2：移除 `__call__` 覆盖，改用 `forward` 分支（建议做）

将 `RMSNorm`、`SiluAndMul`、`RotaryEmbedding` 的 Triton 分支逻辑移入 `forward()` 方法内部，而不是重写 `__call__`。这样 `nn.Module.__call__` 的 hook 机制正常工作，CUDA Graph capture 时行为一致。

```python
# 修改前
def __call__(self, x):
    if self.use_triton and x.is_cuda:
        return self._triton_forward(x)
    return self.forward(x)

# 修改后
def forward(self, x):
    if self.use_triton and x.is_cuda:
        return self._triton_forward(x)
    return self._eager_forward(x)  # 原 forward 改名为 _eager_forward
```

### 修复 3：`Config.use_triton` 默认值改为 `False`（可选）

将默认值设为 `False`，仅在用户显式启用时走 Triton 路径，避免意外行为。

---

## 验证方案

1. 修复 RoPE kernel 后，用 `CUDA_LAUNCH_BLOCKING=1 python bench.py --use_triton` 验证 warmup + CUDA Graph capture 通过
2. 对比 `--use_triton` 开/关的输出结果是否一致
3. 确认 CUDA Graph replay 阶段 Triton kernel 的中间张量不会导致地址冲突

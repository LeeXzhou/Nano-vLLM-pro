# Nano-vLLM-Pro

一个基于 [Nano vLLM](https://github.com/GeeeekExplorer/nano-vllm) 的轻量级 LLM 推理引擎，提供结构清晰、易于理解和扩展的 vLLM 替代方案。

## 特性

### 核心能力（继承自 Nano vLLM）
- **Paged KV-Cache**：基于 block 的分页 KV-Cache 管理，支持动态分配/回收
- **Prefix Caching**：xxhash 增量哈希实现前缀缓存去重，减少重复计算
- **Tensor Parallelism**：多 GPU 张量并行（1-8卡），spawn 多进程 + NCCL 通信
- **CUDA Graph Capture**：decode 阶段 CUDA Graph 捕获，减少 kernel launch 开销
- **Flash Attention**：prefill 用 `flash_attn_varlen_func`，decode 用 `flash_attn_with_kvcache`
- **Qwen3 模型支持**：完整实现 Qwen3ForCausalLM，包含 Q/K norm

### 增强特性（Nano-vLLM-Pro 新增）
- **Triton 算子适配**（Phase 1）：手写 Triton kernel 替换 `@torch.compile`，减少 kernel launch 开销和显存读写
  - Fused RMSNorm（residual add + norm + scale + shift）
  - Fused SiLU+Mul（消除中间 tensor 显存写入）
  - Fused RoPE（原地旋转，避免额外分配）
- **CPU Offload**（Phase 2）：KV-Cache 页块卸载到 CPU pinned memory，扩展可服务序列数
- **标准化 Benchmark**（Phase 3）：结构化性能评测框架，覆盖 TTFT / TPS / E2E Latency / Percentiles
- **混合调度**（Phase 4）：Prefill/Decode 混合调度，消除显存震荡问题
- **Chunked Prefill**（Phase 5）：长 prompt 拆分为 chunk 分步执行，提升系统响应性和并发度
- **多策略采样**：支持 greedy / temperature / top-k / top-p 采样策略

## 项目结构

```
nanovllm/
├── __init__.py              # 导出 LLM, SamplingParams, Config
├── config.py                # 全局配置数据类
├── llm.py                   # 顶层 LLM 类
├── sampling_params.py       # 采样参数（支持 temperature/top-k/top-p/greedy）
├── engine/
│   ├── llm_engine.py        # 核心推理引擎（含 benchmark 采集）
│   ├── sequence.py          # 序列状态管理（含时间戳字段）
│   ├── block_manager.py     # Paged KV-Cache + 前缀缓存 + CPU Offload
│   ├── model_runner.py      # 模型执行器（支持混合调度）
│   └── scheduler.py         # 调度器（混合调度 + Chunked Prefill）
├── layers/
│   ├── attention.py         # Flash Attention + Triton store_kvcache
│   ├── linear.py            # 张量并行线性层族
│   ├── activation.py        # SiluAndMul（Triton + torch.compile 双路径）
│   ├── layernorm.py         # RMSNorm（Triton + torch.compile 双路径）
│   ├── rotary_embedding.py  # RoPE（Triton + torch.compile 双路径）
│   ├── sampler.py           # 采样器（PyTorch 实现）
│   ├── embed_head.py        # 词嵌入 + LM Head
│   ├── triton_rmsnorm.py    # Triton Fused RMSNorm kernel
│   ├── triton_activation.py # Triton Fused SiLU+Mul kernel
│   └── triton_rotary_embedding.py  # Triton Fused RoPE kernel
├── models/
│   └── qwen3.py             # Qwen3 模型实现
└── utils/
    ├── loader.py             # SafeTensors 权重加载
    └── context.py            # 全局上下文
benchmark/
├── __init__.py
├── metrics.py               # 指标采集与报告
├── bench_runner.py           # Benchmark 运行脚本
└── oscillation_repro.py      # Phase 4 震荡复现脚本
```

## 快速开始

### 安装

```bash
# 使用 conda 环境
conda activate leezhou_vllm

# 安装依赖
pip install -e .
```

### 基本使用

```python
from nanovllm import LLM, SamplingParams

llm = LLM("~/huggingface/Qwen3-0.6B/", enforce_eager=True)

# 基本生成
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
outputs = llm.generate(["Hello, world!"], sampling_params)

# Top-K 采样
sampling_params = SamplingParams(temperature=0.8, top_k=50, max_tokens=256)

# Top-P 采样
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=256)

# Greedy 采样
sampling_params = SamplingParams(temperature=0.0, max_tokens=256)
```

### 配置选项

```python
llm = LLM(
    model_path,
    max_model_len=4096,           # 最大模型长度
    enforce_eager=False,           # True: 禁用 CUDA Graph
    tensor_parallel_size=1,        # 张量并行大小
    enable_mixed_scheduling=True,  # Phase 4: 混合调度
    enable_chunked_prefill=True,   # Phase 5: Chunked Prefill
    enable_cpu_offload=False,      # Phase 2: CPU Offload
    prefill_budget_ratio=0.5,      # Prefill 占 token budget 的比例
)
```

### 运行 Benchmark

```bash
# 基本吞吐量测试
python bench.py

# 标准化 Benchmark（多维度对比）
python -m benchmark.bench_runner --model ~/huggingface/Qwen3-0.6B/ --num-seqs 32

# 震荡复现实验（Phase 4 baseline）
python -m benchmark.oscillation_repro --model ~/huggingface/Qwen3-0.6B/
```

## 架构设计

### 数据流

```
用户 API
  └── LLM.generate() → LLMEngine.step() 循环
       ├── Scheduler.schedule()  → 决定 prefill/decode 序列
       ├── ModelRunner.run()     → 准备数据 → 前向 → 采样
       └── Scheduler.postprocess() → 更新序列状态
```

### 调度策略对比

| 特性 | 原始调度 | 混合调度（Phase 4） |
|:---|:---|:---|
| Prefill/Decode | 互斥，先 prefill 后 decode | 同时调度，decode 优先 |
| 显存震荡 | 有（prefill 填满 → decode 抢占 → 循环） | 无（decode 预留空间） |
| 抢占次数 | 频繁 | 接近 0 |
| Chunked Prefill | 仅首个序列 | 多序列并行 chunk（Phase 5） |

### Triton 算子优化

| 算子 | 原始实现 | Triton 优化 | 收益 |
|:---|:---|:---|:---|
| RMSNorm | 2个 `@torch.compile` 子图 | 1个 Triton kernel | 减少 kernel launch + 中间 tensor |
| SiluAndMul | `@torch.compile` SwiGLU | Fused SiLU+Mul | 消除中间 tensor 显存写入 |
| RoPE | `@torch.compile` apply_rotary_emb | 原地旋转 kernel | 避免额外分配 |

## 依赖

- Python >= 3.10, < 3.13
- PyTorch >= 2.4.0
- Triton >= 3.0.0
- Transformers >= 4.51.0
- Flash-Attention
- xxhash
- SafeTensors

## License

MIT License - 基于 [Nano vLLM](https://github.com/GeeeekExplorer/nano-vllm) by Xingkai Yu，增强扩展 by LeeXzhou。

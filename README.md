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
- **Triton 算子加速**：手写 Triton kernel 替换 `@torch.compile`，减少 kernel launch 开销和显存读写
  - Fused RMSNorm（residual add + norm + scale + shift）
  - Fused SiLU+Mul（消除中间 tensor 显存写入）
  - Fused RoPE（原地旋转，避免额外分配）
- **多策略采样**：支持 greedy / temperature / top-k / top-p 采样策略
- **Benchmark 工具**：基础性能评测脚本（吞吐量统计）

## 项目结构

```
nanovllm/
├── __init__.py              # 导出 LLM, SamplingParams, Config
├── config.py                # 全局配置数据类
├── llm.py                   # 顶层 LLM 类
├── sampling_params.py       # 采样参数（支持 temperature/top-k/top-p/greedy）
├── engine/
│   ├── llm_engine.py        # 核心推理引擎
│   ├── sequence.py          # 序列状态管理
│   ├── block_manager.py     # Paged KV-Cache + 前缀缓存
│   ├── model_runner.py      # 模型执行器（含 CUDA Graph 捕获）
│   └── scheduler.py         # 调度器（Prefill/Decode 分阶段调度）
├── layers/
│   ├── attention.py         # Flash Attention + KV-Cache 写入
│   ├── linear.py            # 张量并行线性层族
│   ├── activation.py        # SiluAndMul（Triton + PyTorch 双路径）
│   ├── layernorm.py         # RMSNorm（Triton + torch.compile 双路径）
│   ├── rotary_embedding.py  # RoPE（Triton + PyTorch 双路径）
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
bench.py                     # 基准测试脚本
example.py                   # 使用示例（含 chat template）
```

## 快速开始

### 配置

项目需要 `config.yaml` 来指定模型路径（已从仓库移除，避免泄露本地路径）：

```yaml
# config.yaml
model_path: "~/huggingface/Qwen3-0.6B/"
```

### 基本使用

```python
import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
tokenizer = AutoTokenizer.from_pretrained(path)
llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

prompts = [
    "introduce yourself",
    "list all prime numbers within 100",
]
# 使用 chat template 格式化 prompt
prompts = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    for prompt in prompts
]

# 基本生成
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
outputs = llm.generate(prompts, sampling_params)

# Top-K 采样
SamplingParams(temperature=0.8, top_k=50, max_tokens=256)

# Top-P 采样
SamplingParams(temperature=0.7, top_p=0.9, max_tokens=256)

# Greedy 采样
SamplingParams(temperature=0.0, max_tokens=256)
```

### LLM 配置选项

```python
llm = LLM(
    path,
    enforce_eager=False,           # True: 禁用 CUDA Graph（调试用）
    tensor_parallel_size=1,        # 张量并行大小
    use_triton=False,              # True: 启用 Triton kernel 加速
    max_model_len=4096,            # 最大模型长度
)
```

### 运行 Benchmark

```bash
# 基础 Benchmark（默认 256 并发序列）
python bench.py

# 启用 Triton 加速对比
python bench.py --use_triton

# 自定义并发数和序列长度
python bench.py --use_triton --num_seqs 128 --max_input_len 512 --max_output_len 512
```

bench.py 输出示例：
```
===== 测试报告 (BENCHMARK REPORT) =====
 模型路径 (Model Path)       : ~/huggingface/Qwen3-0.6B/
 Triton 加速 (use_triton)    : True
 并发序列数 (Num Sequences)   : 256
-----------------------------------------
 输入 Token 总数 (Input)     : 123456 tok
 输出 Token 总数 (Output)    : 67890 tok
 吞吐总数 (Total Tokens)     : 191346 tok
-----------------------------------------
 总耗时 (Total Time)         : 45.67 s
 生成吞吐量 (Generation)     : 1486.54 tok/s (仅计算输出)
 整体吞吐量 (Total)          : 4190.01 tok/s (输入+输出)
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

### 调度策略

| 阶段 | 行为 |
|:---|:---|
| Prefill | 优先调度 waiting 队列，首个序列支持 chunked prefill |
| Decode | Prefill 无新序列后调度 running 队列，逐 token 解码 |
| 抢占 | 显存不足时将 running 序列回退到 waiting 队列 |

### Triton 算子优化

| 算子 | 原始实现 | Triton 优化 | 收益 |
|:---|:---|:---|:---|
| RMSNorm | 2个 `@torch.compile` 子图 | 1个 Triton kernel | 减少 kernel launch + 中间 tensor |
| SiluAndMul | 纯 PyTorch（chunk + F.silu + mul） | Fused SiLU+Mul | 消除中间 tensor 显存写入 |
| RoPE | 纯 PyTorch（chunk + cat） | 原地旋转 kernel | 避免额外分配 |

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

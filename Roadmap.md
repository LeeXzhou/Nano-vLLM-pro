### Nano-vLLM-pro Roadmap

**项目简介**：
Nano-vLLM-pro 是一个基于Nano vLLM的轻量级 LLM 推理引擎，目标是提供一个结构清晰、易于理解和扩展的 vLLM 替代方案。当前已实现 Paged KV-Cache、Prefix Caching、Tensor Parallelism、CUDA Graph Capture 等核心能力，以下是我们规划的下一步开发方向。

---

#### Phase 1: Triton 算子适配
**目标**：将 `layers/` 下当前依赖 `torch.compile` 和 PyTorch 原生算子的模块替换为手写 Triton kernel，减少 kernel launch 开销和显存读写次数，提升 decode 阶段的 token 生成吞吐。

| 模块 | 当前实现 | Triton 优化方向 | 优先级 |
| :--- | :--- | :--- | :--- |
| **RMSNorm** | `@torch.compile` 装饰的两个子图（norm / fused add+norm） | Fused RMSNorm kernel：单 kernel 完成 residual add + norm + scale + shift | P0 |
| **SiluAndMul** | `@torch.compile` 的 SwiGLU（silu + elementwise mul） | Fused SiLU+Mul kernel：消除中间 tensor 的显存写入 | P0 |
| **RotaryEmbedding** | `@torch.compile` 的 apply_rotary_emb | Fused RoPE kernel：直接在 Q/K buffer 上原地旋转，避免额外分配 | P1 |
| **Sampler** | `@torch.compile` 的 Gumbel-max trick | Fused Top-K / Top-P + 采样 kernel：支持 greedy / top-k / top-p 多种策略 | P1 |
| **Linear** | `F.linear`（cuBLAS） | 低优先级，cuBLAS 已高度优化；可选做 fused QKV+RoPE | P2 |
| **store_kvcache** | 已有 Triton kernel | 已完成，无需改动 | — |

**验收标准**：每个算子替换后，单层 forward latency 不退化，整体 decode TPS 有可测量提升（见 Phase 3 benchmark）。

---

#### Phase 2: CPU Offload
**目标**：支持将 KV-Cache 页块卸载到 CPU 内存，在 GPU 显存不足时扩展可服务序列数，降低 OOM 和抢占频率。

**2.1 核心设计**
*   **BlockManager 扩展**：在现有 GPU block 池的基础上，增加 CPU block 池（pinned memory）
*   **Offload 策略**：当 GPU block 池耗尽时，将最早进入 running 队列的序列的 inactive block（已生成完毕且不会被回溯的部分）offload 到 CPU
*   **Reload 时机**：被 offload 的序列在 decode 时需要这些 block，则提前一轮将 block 搬回 GPU（async H2D overlap）

**2.2 实现步骤**
1.  在 `BlockManager` 中新增 `cpu_pool` 和 `block_cpu_table`，管理 CPU 端的物理块映射
2.  实现 `offload_blocks(seq_id, block_ids)` 和 `reload_blocks(seq_id, block_ids)`，使用 `cudaMemcpyAsync` 进行 GPU↔CPU 搬运
3.  在 `Scheduler` 的 decode 调度路径中增加 offload 决策逻辑：GPU 池低于阈值时触发
4.  确保 CUDA Stream 正确同步：reload 必须在 attention 计算前完成

**2.3 验收标准**
*   在 `max_model_len` 超出纯 GPU 可容纳序列数时，系统能继续服务而不 OOM
*   Offload/Reload 引入的额外延迟可接受（目标：< 5% TPS 下降，对比纯 GPU 场景）

---

#### Phase 3: 标准化 Benchmark
**目标**：建立可复现的性能评测框架，覆盖关键推理指标，并为后续优化提供量化依据。

**3.1 指标体系**

| 指标 | 含义 | 采集方式 |
| :--- | :--- | :--- |
| **TTFT** (Time To First Token) | 从请求进入到首 token 生成完成的时间 | 记录 `add_request` 时间戳到 `postprocess` 中序列状态变为 running 的时刻 |
| **TPS** (Tokens Per Second) | 每秒生成的 decode token 数（总输出 token / 总 decode 时间） | 累计每个序列的 decode token 数，除以 decode 阶段总耗时 |
| **TPS per request** | 单请求视角的 TPS | 单序列输出 token 数 / 该序列 decode 耗时 |
| **E2E Latency** | 端到端延迟（从请求输入到输出完成） | 记录 `add_request` 到序列 `finished` 的时间 |
| **Latency Percentiles** | P50 / P90 / P99 延迟 | 对所有请求的 E2E Latency / TTFT 做分位数统计 |

**3.2 对比维度**
*   **Triton vs Non-Triton**：同一模型、同一 workload，分别跑 `torch.compile` 路径和 Triton kernel 路径，对比上述指标
*   **不同 batch Size**：1 / 8 / 32 / 128 / 256，观察 TPS 和延迟曲线
*   **不同序列长度**：短 prompt (128) / 中 prompt (512) / 长 prompt (2048)，观察 TTFT 变化
*   **CPU Offload vs 纯 GPU**：在显存压力场景下对比 offload 开销

**3.3 实现步骤**
1.  在 `Sequence` 中增加时间戳字段：`arrival_time`、`first_token_time`、`finish_time`
2.  在 `LLMEngine.step()` 中埋点记录关键时间
3.  新增 `benchmark/` 目录，包含结构化的 benchmark 脚本和结果输出（JSON / Markdown 表格）
4.  保留现有 `bench.py` 作为快速吞吐测试入口

**3.4 验收标准**
*   `benchmark/` 目录可独立运行，输出包含上述所有指标的结构化报告
*   Triton vs Non-Triton 的对比结果可复现

---

#### Phase 4: 调度层重构 — 消除 Prefill/Decode 震荡
**目标**：解决当前"先 Prefill 后 Decode"调度策略导致的显存震荡问题。

**4.1 问题分析**
当前 `Scheduler.schedule()` 的逻辑是：**只要有 waiting 序列，就优先做 prefill，直到 waiting 队列清空或显存不足，再进入 decode**。这会导致以下震荡循环：
```text
Prefill 填满显存 → Decode 发现显存不足 → 抢占 (preempt) 释放 block →
释放的显存立即被新一轮 Prefill 占满 → Decode 再次抢占 → 循环
```
**实验验证**：构造一个可复现的场景（例如：256 个请求，prompt 长度 512，max_tokens 256，在单 GPU 上运行），观察 scheduler 每一步的 `is_prefill` 标志和抢占次数，截取震荡过程的日志截图，作为后续修复的 baseline。

**4.2 修复方案**
将 Prefill 和 Decode 混合调度：**每个 step 同时调度 prefill 和 decode 序列**。
*   每个 step 的 token budget 分成两部分：`prefill_budget` 和 `decode_budget`
*   `decode_budget` 优先保障：先为当前 running 序列预留 decode 所需的 token budget 和 block 空间
*   `prefill_budget` 使用剩余预算：在确保 decode 不会因显存不足而被抢占的前提下，调度新的 prefill 请求
*   这样 decode 始终有保障地执行，不会因 prefill 而被抢占

**4.3 实现步骤**
1.  撰写震荡复现实验脚本，截取 baseline 截图
2.  重构 `Scheduler.schedule()`：移除 `is_prefill` 的互斥返回逻辑，改为返回 `(prefill_seqs, decode_seqs)` 双队列
3.  修改 `ModelRunner`：支持同一 step 内同时执行 prefill 和 decode（需要两次 forward 或混合 batch）
4.  调整 `postprocess()` 逻辑以适配新的调度返回结构

**4.4 验收标准**
*   震荡场景下抢占次数降为 0 或接近 0
*   Decode TPS 不再出现周期性抖动
*   对比截图：修复前后的 scheduler 每步状态日志

---

#### Phase 5: Chunked Prefill 优化
**目标**：将长 prompt 的 prefill 拆分为多个 chunk 分步执行，避免单个长 prompt 独占整个 step 的 token budget，提升系统响应性和并发度。

**5.1 当前状态**
当前 Scheduler 已支持对 waiting 队列中**第一个序列**做部分 prefill（如果其 token 数超出剩余 budget），但后续序列如果放不下则直接跳过。这意味着一个超长 prompt 可以阻塞整个 batch。

**5.2 优化设计**
*   **Chunk 粒度**：以 `block_size`（当前 256 tokens）为单位拆分 prefill 请求
*   **公平调度**：每个 step 的 prefill budget 按 round-robin 或 FCFS 分配给多个 waiting 序列，而非让单个序列独占
*   **与 Phase 4 联动**：在混合调度框架下，chunked prefill 天然适配——每个 step 的 prefill_budget 可同时服务多个序列的 chunk
*   **前缀缓存复用**：已经缓存的 chunk 不重复计算，只 prefill 新增部分

**5.3 实现步骤**
1.  在 `Sequence` 中增加 `num_scheduled_tokens` 字段，追踪已调度但尚未完成的 prefill token 数
2.  修改 `Scheduler.schedule()` 的 prefill 路径：对每个 waiting 序列分配 `min(remaining_budget, chunk_size)` 个 token
3.  确保 `ModelRunner.prepare_prefill()` 支持部分 token 的 prefill（当前已部分支持）
4.  在 `postprocess()` 中正确处理部分 prefill 完成的序列：未完成的序列保留在 waiting 队列，下一次 step 继续调度

**5.4 验收标准**
*   长短 prompt 混合场景下，短 prompt 的 TTFT 不再被长 prompt 阻塞
*   同一 step 内可同时为多个序列推进 prefill
*   整体 TPS 和 TTFT P90 有可测量改善

---

#### 开发优先级总览

| 优先级 | Phase | 预估工期 | 依赖关系 |
| :--- | :--- | :--- | :--- |
| **P0** | Phase 1: Triton 算子适配 | 2-3 周 | 无 |
| **P0** | Phase 3: 标准化 Benchmark | 1-2 周 | 无（与 Phase 1 可并行） |
| **P1** | Phase 4: 调度层重构 | 2-3 周 | Phase 3（需要 benchmark 做 baseline） |
| **P1** | Phase 5: Chunked Prefill | 1-2 周 | Phase 4（依赖混合调度框架） |
| **P2** | Phase 2: CPU Offload | 2-3 周 | Phase 4（减少抢占后 offload 场景更清晰） |

**路线图说明**：
Phase 1 和 Phase 3 可以并行推进，完成后为后续 Phase 提供性能度量和优化工具。Phase 4 是架构层面的关键改动，Phase 5 在其基础上实现。Phase 2 放在较后位置，因为调度优化本身就能减少显存压力，降低 offload 的必要性。
import os
import time
from random import randint, seed
import argparse
from nanovllm import LLM, SamplingParams
import yaml

with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

MODEL_PATH = config["model_path"]


def parse_args():
    parser = argparse.ArgumentParser(description="nanoVLLM 基准测试脚本")

    parser.add_argument("--use_triton", action="store_true", help="启用 Triton")
    parser.add_argument(
        "--num_seqs", type=int, default=256, help="并发序列数 (默认: 256)"
    )
    parser.add_argument(
        "--max_input_len", type=int, default=1024, help="最大输入长度 (默认: 1024)"
    )
    parser.add_argument(
        "--max_output_len", type=int, default=1024, help="最大输出长度 (默认: 1024)"
    )
    parser.add_argument("--seed", type=int, default=0, help="随机种子 (默认: 0)")

    return parser.parse_args()


def main():
    args = parse_args()
    use_triton = args.use_triton
    num_seqs = args.num_seqs
    max_input_len = args.max_input_len
    max_ouput_len = args.max_output_len

    path = os.path.expanduser(MODEL_PATH)
    print(f"正在初始化模型，当前配置 [use_triton={args.use_triton}] ...")
    llm = LLM(path, enforce_eager=False, max_model_len=max_input_len + max_ouput_len, use_triton=use_triton) # 默认使用CUDA Graph

    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, max_input_len))]
        for _ in range(num_seqs)
    ]
    sampling_params = [
        SamplingParams(
            temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)
        )
        for _ in range(num_seqs)
    ]

    print("正在Warmup...")
    llm.generate(["Benchmark: "], SamplingParams())

    print("开始执行性能测试...")
    t_start = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t_end = time.time() - t_start

    # total_tokens = sum(sp.max_tokens for sp in sampling_params)
    # throughput = total_tokens / t
    # print(
    #     f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s"
    # )

    total_input_tokens = sum(len(ids) for ids in prompt_token_ids)
    total_output_tokens = sum(sp.max_tokens for sp in sampling_params)
    total_tokens = total_input_tokens + total_output_tokens
    
    generation_throughput = total_output_tokens / t_end
    total_throughput = total_tokens / t_end

    # 打印测试报告
    print("\n" + "="*20 + " 测试报告 (BENCHMARK REPORT) " + "="*20)
    print(f" 模型路径 (Model Path)       : {path}")
    print(f" Triton 加速 (use_triton)    : {args.use_triton}")
    print(f" 并发序列数 (Num Sequences)   : {num_seqs}")
    print("-" * 69)
    print(f" 输入 Token 总数 (Input)     : {total_input_tokens} tok")
    print(f" 输出 Token 总数 (Output)    : {total_output_tokens} tok")
    print(f" 吞吐总数 (Total Tokens)     : {total_tokens} tok")
    print("-" * 69)
    print(f" 总耗时 (Total Time)         : {t_end:.2f} s")
    print(f" 生成吞吐量 (Generation)     : {generation_throughput:.2f} tok/s (仅计算输出)")
    print(f" 整体吞吐量 (Total)          : {total_throughput:.2f} tok/s (输入+输出)")


if __name__ == "__main__":
    main()

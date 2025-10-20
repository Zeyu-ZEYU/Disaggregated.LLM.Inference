#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepEP High-Throughput all-to-all micro-benchmark
GPU-only, cross-node, NO vLLM parallel_state (no DP/EP init), no MoE GEMM.

Launch (multi-node example):
  MASTER_ADDR=<master_ip> MASTER_PORT=29500 NNODES=2 NODE_RANK=0 \
  torchrun --nproc_per_node=8 bench_deepep_ht_a2a_nodp.py

  MASTER_ADDR=<master_ip> MASTER_PORT=29500 NNODES=2 NODE_RANK=1 \
  torchrun --nproc_per_node=8 bench_deepep_ht_a2a_nodp.py

Optional envs:
  TOKENS=8192
  HIDDEN=4096
  ITERS=50
  VLLM_DEEPEP_BUFFER_SIZE_MB=256
  CUDA_DEVICE_MAX_CONNECTIONS=1
"""

import os
import time

import torch
import torch.distributed as dist


def init_nccl_world():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank % max(1, torch.cuda.device_count()))
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ.setdefault("NCCL_P2P_LEVEL", "NVL")  # prefer NVLink intra-node


def make_deepep_ht_buffer(world_size: int, rank: int):
    """
    Create deep_ep.Buffer for High Throughput mode directly, avoiding vLLM managers.
    """
    import deep_ep

    # Size knobs (bytes); HT uses both NVLink buffers and (if internode) RDMA buffers.
    buf_mb = int(os.getenv("VLLM_DEEPEP_BUFFER_SIZE_MB", "256"))
    num_nvl_bytes = buf_mb * 1024 * 1024

    # Detect internode if RANKs span multiple hosts. A simple heuristic is fine here.
    # You can force internode by setting DEEPEP_INTERNODE=1
    internode = int(os.getenv("DEEPEP_INTERNODE", "0"))
    if internode == 0:
        # Heuristic: if NCCL_SOCKET_IFNAME is set (multi-node typical) assume internode
        internode = 1 if "NCCL_SOCKET_IFNAME" in os.environ else 0

    if internode:
        num_rdma_bytes = buf_mb * 1024 * 1024
        # Empirically HT 推荐每 rank 多些 QPs；你也可以用 16、32 试
        num_qps_per_rank = 8
    else:
        num_rdma_bytes = 0
        num_qps_per_rank = 1

    # Construct Buffer. `group` 直接用 WORLD（GPU group）。
    buffer = deep_ep.Buffer(
        group=dist.group.WORLD,
        num_nvl_bytes=num_nvl_bytes,
        num_rdma_bytes=num_rdma_bytes,
        low_latency_mode=False,  # High Throughput mode
        num_qps_per_rank=num_qps_per_rank,
    )

    # 控制通信占用 SM 数（HT 内核用 SM 做 pack/copy）。可以收紧到 20，或按你 GPU 调整。
    try:
        deep_ep.Buffer.set_num_sms(20)
    except Exception:
        pass

    return buffer


def main():
    # Tunables
    tokens_per_rank = int(os.getenv("TOKENS", "8192"))
    hidden = int(os.getenv("HIDDEN", "4096"))
    iters = int(os.getenv("ITERS", "50"))
    os.environ.setdefault("VLLM_DEEPEP_BUFFER_SIZE_MB", "256")

    init_nccl_world()
    rank = dist.get_rank()
    world = dist.get_world_size()

    # --- Prepare deep_ep.Buffer (HT) and PF driver ---
    buffer = make_deepep_ht_buffer(world, rank)

    from vllm.model_executor.layers.fused_moe.config import (
        FUSED_MOE_UNQUANTIZED_CONFIG,
    )
    from vllm.model_executor.layers.fused_moe.deepep_ht_prepare_finalize import (
        DeepEPHTPrepareAndFinalize,
    )
    from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
        TopKWeightAndReduceContiguous,
    )

    # One local expert per rank in this micro-bench
    num_experts = world
    n_local_experts = 1
    rank_expert_offset = rank * n_local_experts

    pf = DeepEPHTPrepareAndFinalize(
        buffer=buffer,
        num_dispatchers=world,  # EP world size
        dp_size=1,  # no data-parallelism here
        rank_expert_offset=rank_expert_offset,
    )

    # Inputs on GPU (BF16 recommended for HT path)
    x = torch.randn(tokens_per_rank, hidden, device="cuda", dtype=torch.bfloat16)

    # Trivial routing: send everything to this rank (top-1); you can randomize to stress a2a
    topk_ids = torch.full((tokens_per_rank, 1), rank, device="cuda", dtype=torch.int64)
    topk_weights = torch.ones((tokens_per_rank, 1), device="cuda", dtype=torch.bfloat16)

    quant_config = FUSED_MOE_UNQUANTIZED_CONFIG  # simplest path
    apply_router_weight_on_input = False
    expert_map = None

    # Warmup
    for _ in range(5):
        receiver = pf.prepare_async(
            a1=x,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            num_experts=num_experts,
            expert_map=expert_map,
            apply_router_weight_on_input=apply_router_weight_on_input,
            quant_config=quant_config,
        )
        (
            expert_x,
            expert_x_scale,
            expert_tokens_meta,
            expert_topk_ids,
            expert_topk_weights,
        ) = receiver()
        fused_expert_output = expert_x.to(torch.bfloat16)  # HT combine expects BF16
        out = torch.empty(tokens_per_rank, hidden, device="cuda", dtype=torch.bfloat16)
        pf.finalize(
            output=out,
            fused_expert_output=fused_expert_output,
            topk_weights=expert_topk_weights
            if expert_topk_weights is not None
            else topk_weights,
            topk_ids=expert_topk_ids,
            apply_router_weight_on_input=apply_router_weight_on_input,
            weight_and_reduce_impl=TopKWeightAndReduceContiguous(),
        )
        torch.cuda.synchronize()

    # Benchmark
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        receiver = pf.prepare_async(
            a1=x,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            num_experts=num_experts,
            expert_map=expert_map,
            apply_router_weight_on_input=apply_router_weight_on_input,
            quant_config=quant_config,
        )
        (
            expert_x,
            expert_x_scale,
            expert_tokens_meta,
            expert_topk_ids,
            expert_topk_weights,
        ) = receiver()
        fused_expert_output = expert_x.to(torch.bfloat16)
        out = torch.empty(tokens_per_rank, hidden, device="cuda", dtype=torch.bfloat16)
        pf.finalize(
            output=out,
            fused_expert_output=fused_expert_output,
            topk_weights=expert_topk_weights
            if expert_topk_weights is not None
            else topk_weights,
            topk_ids=expert_topk_ids,
            apply_router_weight_on_input=apply_router_weight_on_input,
            weight_and_reduce_impl=TopKWeightAndReduceContiguous(),
        )
    torch.cuda.synchronize()
    t1 = time.time()

    total_tokens = tokens_per_rank * world * iters
    elapsed = max(t1 - t0, 1e-9)
    mtok_s = total_tokens / elapsed / 1e6
    if rank == 0:
        print(
            f"[DeepEP-HT][No-DP] world={world} hidden={hidden} tokens/rank={tokens_per_rank} iters={iters}"
        )
        print(
            f"[DeepEP-HT][No-DP] time={elapsed:.3f}s  tokens={total_tokens}  throughput={mtok_s:.3f} MTok/s"
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

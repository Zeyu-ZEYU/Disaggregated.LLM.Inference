#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Minimal DeepEP High-Throughput A2A benchmark for inference-side routing.
- GPU-only process group (no vLLM groups, no DP init)
- Cross-node friendly: launch with torchrun across nodes
- Uses DeepEP Buffer.get_dispatch_layout / dispatch / combine as in official example
- No model compute: we only emulate MoE dispatch+combine to measure A2A performance

USAGE (single node, 4 GPUs):
  torchrun --nproc_per_node=4 bench_deepep_ht_infer_a2a.py

USAGE (two nodes, 4 GPUs each):
  # On ALL nodes, set the same rendezvous envs (or pass as torchrun args)
  # e.g. MASTER_ADDR, MASTER_PORT, NODE_RANK, WORLD_SIZE=8
  torchrun --nnodes=2 --nproc_per_node=4 --node_rank=${NODE_RANK} \
           --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
           bench_deepep_ht_infer_a2a.py \
           --num-experts 64 --tokens-per-rank 32768 --hidden-size 4096

Tuning tips:
  - For A100, set: TORCH_CUDA_ARCH_LIST="8.0"
  - Ensure NCCL/IB envs are properly set for cross-node (NCCL_IB=1, NCCL_NET_GDR_LEVEL=5, etc. if applicable)
"""

import argparse
import os
import time
from typing import Optional, Tuple, List, Union

import torch
import torch.distributed as dist

from deep_ep import Buffer, EventOverlap


# -----------------------------
# Utilities
# -----------------------------

def log(rank: int, *a, **kw):
    if rank == 0:
        print(*a, **kw, flush=True)


def get_hidden_bytes(x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> int:
    """
    DeepEP HT uses at least 2 bytes granularity for the hidden dimension,
    so we follow the official example: bytes = hidden * max(elem_size, 2).
    """
    t = x[0] if isinstance(x, tuple) else x
    return t.size(1) * max(t.element_size(), 2)


def build_inputs(
    world_size: int,
    rank: int,
    tokens_per_rank: int,
    hidden_size: int,
    dtype: torch.dtype,
    topk: int,
    num_experts: int,
    seed: int = 1234,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create synthetic activations and routing for inference-side A2A.
    - a1      : [T_r, H] tokens on this rank
    - topk_id : [T_r, topk] with global expert ids in [0, num_experts)
    - topk_w  : [T_r, topk] softmax-like weights (not used by HT combine but kept for completeness)
    NOTE: For HT, experts are assumed evenly sharded across ranks:
          local_experts = num_experts / world_size, contiguous mapping.
    """
    assert num_experts % world_size == 0, "num_experts must be divisible by world_size"
    torch.manual_seed(seed + rank)

    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    a1 = torch.randn(tokens_per_rank, hidden_size, dtype=dtype, device=device)

    # Create random global expert ids; distribute roughly uniformly
    # to stress cross-rank traffic.
    topk_ids = torch.randint(
        low=0, high=num_experts, size=(tokens_per_rank, topk), device=device, dtype=torch.int32
    )
    # Weights could be normalized; here we just make them positive.
    topk_w = torch.rand(tokens_per_rank, topk, dtype=torch.float32, device=device)

    return a1, topk_ids, topk_w


# -----------------------------
# Core A2A helpers (DeepEP HT)
# -----------------------------

_buffer: Optional[Buffer] = None  # module-global buffer like the official sample


def ensure_buffer(group: dist.ProcessGroup, hidden_bytes: int) -> Buffer:
    """
    Allocate/resize DeepEP buffer according to dispatch/combine config hints.
    We mirror the official example logic.
    """
    global _buffer

    # Query size hints from DeepEP configs (pick the max of dispatch/combine)
    num_nvl_bytes, num_rdma_bytes = 0, 0
    for cfg in (Buffer.get_dispatch_config(group.size()), Buffer.get_combine_config(group.size())):
        num_nvl_bytes = max(cfg.get_nvl_buffer_size_hint(hidden_bytes, group.size()), num_nvl_bytes)
        num_rdma_bytes = max(cfg.get_rdma_buffer_size_hint(hidden_bytes, group.size()), num_rdma_bytes)

    if (
        _buffer is None
        or _buffer.group != group
        or _buffer.num_nvl_bytes < num_nvl_bytes
        or _buffer.num_rdma_bytes < num_rdma_bytes
    ):
        _buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes)

    return _buffer


def a2a_dispatch_forward(
    x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    previous_event: Optional[EventOverlap] = None,
):
    """
    HT dispatch (forward). Returns the tensors living on *expert owners* after routing,
    plus a communication handle and event for chaining.
    """
    global _buffer
    # Precompute the token layout (how many tokens go to each rank/expert, etc.)
    num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, previous_event = \
        _buffer.get_dispatch_layout(
            topk_idx, num_experts,
            previous_event=previous_event,
            async_finish=True,
            allocate_on_comm_stream=(previous_event is not None),
        )

    recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, event = \
        _buffer.dispatch(
            x,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=previous_event,
            async_finish=True,
            allocate_on_comm_stream=(previous_event is not None),
        )

    return recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, event


def a2a_combine_forward(
    x: torch.Tensor,
    handle,
    previous_event: Optional[EventOverlap] = None,
):
    """
    HT combine (forward): gather results back to the original token owners in order.
    """
    global _buffer
    combined_x, _, event = _buffer.combine(
        x,
        handle,
        async_finish=True,
        previous_event=previous_event,
        allocate_on_comm_stream=(previous_event is not None),
    )
    return combined_x, event


# -----------------------------
# Benchmark
# -----------------------------

def benchmark_once(
    a1: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_w: torch.Tensor,
    num_experts: int,
    do_fake_expert_compute: bool = True,
):
    """
    One end-to-end A2A pass:
      1) dispatch to expert owners
      2) (optional) fake expert compute
      3) combine back to token owners
    Returns elapsed time in seconds.
    """
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    # (1) dispatch
    recv_x, recv_topk_idx, recv_topk_w, num_recv_per_expert, handle, ev_d = a2a_dispatch_forward(
        a1, topk_ids, topk_w, num_experts
    )

    # (2) fake expert compute (e.g., scaling); ensure it runs on compute stream
    if do_fake_expert_compute and recv_x.numel() > 0:
        # A tiny fused op to simulate some work; keep it memory-bound.
        recv_x.mul_(1.0001)

    # (3) combine
    combined_x, ev_c = a2a_combine_forward(recv_x, handle, previous_event=ev_d)

    # Wait for combine to finish
    if ev_c.event is not None:
        ev_c.current_stream_wait()

    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return t1 - t0, combined_x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens-per-rank", type=int, default=32768,
                        help="Number of input tokens on each rank.")
    parser.add_argument("--hidden-size", type=int, default=4096,
                        help="Hidden size per token.")
    parser.add_argument("--dtype", type=str, choices=["bf16", "fp16"], default="bf16",
                        help="Activation dtype for dispatch/combine. HT combine expects bf16.")
    parser.add_argument("--num-experts", type=int, default=64,
                        help="Global number of experts (must be divisible by world size).")
    parser.add_argument("--topk", type=int, default=1,
                        help="Experts per token (HT example typically uses topk=1).")
    parser.add_argument("--iters", type=int, default=20,
                        help="Number of measured iterations.")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Warmup iterations (not timed).")
    parser.add_argument("--num-sms", type=int, default=24,
                        help="DeepEP Buffer.set_num_sms value.")
    parser.add_argument("--no-fake-compute", action="store_true",
                        help="Disable fake expert compute between dispatch and combine.")
    args = parser.parse_args()

    # --- init distributed ---
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # pin GPU
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    # dtype selection
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    if args.dtype == "fp16":
        # DeepEP HT combine currently assumes bf16; we'll still run,
        # but it's recommended to benchmark with bf16 for realistic paths.
        pass

    # DeepEP static knob: how many SMs comm kernels may use
    Buffer.set_num_sms(args.num_sms)

    if args.num_experts % world_size != 0:
        if rank == 0:
            print(f"[WARN] num_experts {args.num_experts} is not divisible by world_size {world_size}. "
                  f"DeepEP HT assumes even shard; rounding up to a multiple.")
        # round up to multiple for safety
        new_ne = ((args.num_experts + world_size - 1) // world_size) * world_size
        args.num_experts = new_ne

    # --- build inputs ---
    a1, topk_ids, topk_w = build_inputs(
        world_size=world_size,
        rank=rank,
        tokens_per_rank=args.tokens_per_rank,
        hidden_size=args.hidden_size,
        dtype=dtype,
        topk=args.topk,
        num_experts=args.num_experts,
    )

    # --- allocate / resize DeepEP buffer ---
    hidden_bytes = get_hidden_bytes(a1)
    # use the default global process group as the GPU group
    buffer = ensure_buffer(dist.group.WORLD, hidden_bytes)  # noqa: F841 (held by global _buffer)

    # --- barrier before timing ---
    dist.barrier()

    # --- warmup ---
    for _ in range(args.warmup):
        dt, _ = benchmark_once(
            a1, topk_ids, topk_w,
            num_experts=args.num_experts,
            do_fake_expert_compute=(not args.no_fake_compute),
        )

    # --- measure ---
    times = []
    for _ in range(args.iters):
        dt, _ = benchmark_once(
            a1, topk_ids, topk_w,
            num_experts=args.num_experts,
            do_fake_expert_compute=(not args.no_fake_compute),
        )
        times.append(dt)

    # --- aggregate ---
    t = torch.tensor([sum(times) / len(times)], device=a1.device)
    dist.all_reduce(t, op=dist.ReduceOp.AVG)
    avg_s = float(t.item())

    # Effective routed tokens per second = (tokens_per_rank * world_size) / avg_time
    eff_tps = (args.tokens_per_rank * world_size) / max(avg_s, 1e-9)
    log(rank, f"[DeepEP-HT A2A] world_size={world_size}, experts={args.num_experts}, topk={args.topk}, "
              f"hidden={args.hidden_size}, dtype={args.dtype}, num_sms={args.num_sms}")
    log(rank, f"  tokens_per_rank={args.tokens_per_rank}, warmup={args.warmup}, iters={args.iters}")
    log(rank, f"  avg latency per pass = {avg_s*1000:.3f} ms")
    log(rank, f"  effective routed tokens/s (global) ≈ {eff_tps:,.0f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

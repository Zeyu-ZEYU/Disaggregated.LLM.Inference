#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Minimal DeepEP High-Throughput A2A benchmark for inference-side routing.
- GPU-only process group (no vLLM groups, no DP init)
- Cross-node friendly: torchrun (env://) or explicit TCPStore rendezvous
- Uses DeepEP Buffer.get_dispatch_layout / dispatch / combine as in official example
- No model compute: only emulate MoE dispatch+combine to measure A2A performance

Examples
--------
# Single node, 4 GPUs:
  TORCH_CUDA_ARCH_LIST="8.0" \
  torchrun --nproc_per_node=4 bench_deepep_ht_a2a.py

# Two nodes, 4 GPUs each (env rendezvous, still RDMA for NCCL):
  export MASTER_ADDR=<master_ip_or_hostname>
  export MASTER_PORT=29501
  torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 bench_deepep_ht_a2a.py
  # on node 1:
  torchrun --nnodes=2 --nproc_per_node=4 --node_rank=1 bench_deepep_ht_a2a.py

# Explicit TCPStore rendezvous (no torchrun env); you run this on every node/rank:
  python bench_deepep_ht_a2a.py \
    --master-addr <master_ip_or_hostname> --master-port 29501 \
    --rank <global_rank> --world-size <global_world_size> --gpu-id <local_gpu_id>

Notes
-----
- Rendezvous (env:// or TCPStore) 只负责进程发现；实际通信由 NCCL 完成。
  只要不要禁用 IB（不要设置 NCCL_IB_DISABLE=1），NCCL 仍会走 RDMA/IB。
- 你可以设置例如：
  export NCCL_IB=1
  export NCCL_NET_GDR_LEVEL=5
  export NCCL_DEBUG=INFO            # 如需排查
  export NCCL_SOCKET_IFNAME=eth0    # 指定网卡（按需）
  export NCCL_IB_HCA=mlx5_0,mlx5_1  # 指定 HCA（按需）
"""

import argparse
import datetime
import os
import time
from typing import Optional, Tuple, Union

import torch
import torch.distributed as dist
from deep_ep import Buffer, EventOverlap
from torch.distributed import TCPStore

# -----------------------------
# Rendezvous / init helpers
# -----------------------------


def init_dist_with_tcp_store(
    backend: str = "nccl",
    master_addr: str | None = None,
    master_port: int | None = None,
    rank: int | None = None,
    world_size: int | None = None,
    timeout_s: int = 1800,
    prefer_env: bool = True,
) -> tuple[int, int, int]:
    """
    用 TCP rendezvous 初始化 NCCL 进程组，但保持 NCCL 用 RDMA/IB 传输。
    返回 (rank, world_size, local_rank)。

    prefer_env=True 时，如果检测到 torchrun 的环境变量，就直接用 env://。
    否则走显式 TCPStore（需手动提供 master_addr/port/rank/world_size）。
    """

    # torchrun 常见环境
    env_rank = os.environ.get("RANK")
    env_world_size = os.environ.get("WORLD_SIZE")
    env_local_rank = os.environ.get("LOCAL_RANK")
    env_master_addr = os.environ.get("MASTER_ADDR")
    env_master_port = os.environ.get("MASTER_PORT")

    if prefer_env and all(
        [env_rank, env_world_size, env_local_rank, env_master_addr, env_master_port]
    ):
        r = int(env_rank)
        ws = int(env_world_size)
        lr = int(env_local_rank)
        torch.cuda.set_device(lr)
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=datetime.timedelta(seconds=timeout_s),
            rank=r,
            world_size=ws,
        )
        return r, ws, lr

    # 显式 TCPStore
    assert master_addr is not None and master_port is not None, (
        "When not using torchrun env, you must pass --master-addr/--master-port"
    )
    assert rank is not None and world_size is not None, (
        "When not using torchrun env, you must pass --rank/--world-size"
    )

    local_rank_env = os.environ.get("LOCAL_RANK")
    lr = int(local_rank_env) if local_rank_env is not None else 0
    torch.cuda.set_device(lr)

    is_master = rank == 0
    store = TCPStore(
        master_addr,
        int(master_port),
        int(world_size),
        is_master,
        timeout=datetime.timedelta(seconds=timeout_s),
    )

    dist.init_process_group(
        backend=backend,
        store=store,
        rank=int(rank),
        world_size=int(world_size),
        timeout=datetime.timedelta(seconds=timeout_s),
    )
    return int(rank), int(world_size), lr


# -----------------------------
# Utilities
# -----------------------------


def log(rank: int, *a, **kw):
    if rank == 0:
        print(*a, **kw, flush=True)


def get_hidden_bytes(x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> int:
    """
    DeepEP HT 使用至少 2 字节粒度对齐 hidden 维度；参考官方示例：
    bytes = hidden * max(elem_size, 2)
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
    构造推理侧 A2A 的合成输入与路由：
      a1      : [T_r, H]
      topk_id : [T_r, topk] 全局 expert id
      topk_w  : [T_r, topk] 权重（HT combine 不用，但保持一致性）
    这里假设 experts 按连续切分均匀分布到各 rank（DeepEP HT 假设）。
    """
    assert num_experts % world_size == 0, "num_experts must be divisible by world_size"
    torch.manual_seed(seed + rank)
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    a1 = torch.randn(tokens_per_rank, hidden_size, dtype=dtype, device=device)

    topk_ids = torch.randint(
        low=0,
        high=num_experts,
        size=(tokens_per_rank, topk),
        device=device,
        dtype=torch.int32,
    )
    topk_w = torch.rand(tokens_per_rank, topk, dtype=torch.float32, device=device)
    return a1, topk_ids, topk_w


# -----------------------------
# Core A2A helpers (DeepEP HT)
# -----------------------------

_buffer: Optional[Buffer] = None  # 与官方示例一致：模块级缓存


def ensure_buffer(group: dist.ProcessGroup, hidden_bytes: int) -> Buffer:
    """
    按 DeepEP 配置建议申请/扩容通信缓冲区（参考官方示例）。
    """
    global _buffer
    num_nvl_bytes, num_rdma_bytes = 0, 0
    for cfg in (
        Buffer.get_dispatch_config(group.size()),
        Buffer.get_combine_config(group.size()),
    ):
        num_nvl_bytes = max(
            cfg.get_nvl_buffer_size_hint(hidden_bytes, group.size()), num_nvl_bytes
        )
        num_rdma_bytes = max(
            cfg.get_rdma_buffer_size_hint(hidden_bytes, group.size()), num_rdma_bytes
        )

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
    High-Throughput dispatch：
    返回路由后 *expert 拥有者* 侧的张量，以及 handle/event。
    """
    global _buffer
    (
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        previous_event,
    ) = _buffer.get_dispatch_layout(
        topk_idx,
        num_experts,
        previous_event=previous_event,
        async_finish=True,
        allocate_on_comm_stream=(previous_event is not None),
    )

    (
        recv_x,
        recv_topk_idx,
        recv_topk_weights,
        num_recv_tokens_per_expert_list,
        handle,
        event,
    ) = _buffer.dispatch(
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
    return (
        recv_x,
        recv_topk_idx,
        recv_topk_weights,
        num_recv_tokens_per_expert_list,
        handle,
        event,
    )


def a2a_combine_forward(
    x: torch.Tensor,
    handle,
    previous_event: Optional[EventOverlap] = None,
):
    """
    High-Throughput combine：按原 token 顺序回收结果到 *token 拥有者*。
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
    一次端到端 A2A：
      1) dispatch 到 expert 拥有者
      2) （可选）模拟 expert 计算
      3) combine 回 token 拥有者
    返回 (耗时秒, combined_x)
    """
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    # (1) dispatch
    recv_x, recv_topk_idx, recv_topk_w, num_recv_per_expert, handle, ev_d = (
        a2a_dispatch_forward(a1, topk_ids, topk_w, num_experts)
    )

    # (2) 模拟专家计算：用一个就地乘法保持简单（尽量走计算流）
    if do_fake_expert_compute and recv_x.numel() > 0:
        recv_x.mul_(1.0001)

    # (3) combine
    combined_x, ev_c = a2a_combine_forward(recv_x, handle, previous_event=ev_d)

    # 等 combine 完成
    if ev_c.event is not None:
        ev_c.current_stream_wait()

    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return t1 - t0, combined_x


def main():
    parser = argparse.ArgumentParser()
    # rendezvous 相关（可选；用 torchrun 时可不传）
    parser.add_argument("--master-addr", type=str, default=None)
    parser.add_argument("--master-port", type=int, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--world-size", type=int, default=None)
    parser.add_argument(
        "--no-env",
        action="store_true",
        help="不要使用 torchrun/env://，强制用 TCPStore + --master-addr/--master-port/--rank/--world-size",
    )

    # 绑核（无 torchrun 时需要显式给）
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=None,
        help="Local GPU id；若用 torchrun 则忽略，走 LOCAL_RANK",
    )

    # DeepEP & workload
    parser.add_argument("--tokens-per-rank", type=int, default=32768)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bf16", "fp16"],
        default="bf16",
        help="HT combine 推荐 bf16；fp16 仅用于对比",
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        default=64,
        help="必须是 world_size 的倍数（HT 均匀切分）",
    )
    parser.add_argument(
        "--topk", type=int, default=1, help="每 token 的 experts 数，HT 示例通常用 1"
    )
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--num-sms", type=int, default=24, help="DeepEP Buffer.set_num_sms"
    )
    parser.add_argument(
        "--no-fake-compute",
        action="store_true",
        help="关闭 dispatch 与 combine 之间的模拟专家计算",
    )
    args = parser.parse_args()

    # --- init distributed (env:// 优先；否则 TCPStore) ---
    prefer_env = not args.no_env
    r, ws, lr = init_dist_with_tcp_store(
        backend="nccl",
        master_addr=args.master_addr,
        master_port=args.master_port,
        rank=args.rank,
        world_size=args.world_size,
        prefer_env=prefer_env,
    )

    # 若用户手动传了 --gpu-id，则覆盖 LOCAL_RANK
    if args.gpu - id is not None:
        torch.cuda.set_device(int(args.gpu_id))

    # dtype
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    # DeepEP 内部静态并发参数
    Buffer.set_num_sms(args.num_sms)

    # 校验/修正 num_experts
    if args.num_experts % ws != 0:
        if r == 0:
            print(
                f"[WARN] num_experts {args.num_experts} 不是 world_size={ws} 的倍数，向上取整以满足 HT 均匀切分。",
                flush=True,
            )
        args.num_experts = ((args.num_experts + ws - 1) // ws) * ws

    # --- 构造输入 ---
    a1, topk_ids, topk_w = build_inputs(
        world_size=ws,
        rank=r,
        tokens_per_rank=args.tokens_per_rank,
        hidden_size=args.hidden_size,
        dtype=dtype,
        topk=args.topk,
        num_experts=args.num_experts,
    )

    # --- 申请/扩容 DeepEP buffer ---
    hidden_bytes = get_hidden_bytes(a1)
    _ = ensure_buffer(dist.group.WORLD, hidden_bytes)

    dist.barrier()

    # --- warmup ---
    for _ in range(args.warmup):
        _dt, _ = benchmark_once(
            a1,
            topk_ids,
            topk_w,
            num_experts=args.num_experts,
            do_fake_expert_compute=(not args.no_fake_compute),
        )

    # --- measure ---
    times = []
    for _ in range(args.iters):
        dt, _ = benchmark_once(
            a1,
            topk_ids,
            topk_w,
            num_experts=args.num_experts,
            do_fake_expert_compute=(not args.no_fake_compute),
        )
        times.append(dt)

    # --- aggregate ---
    avg_local = sum(times) / max(len(times), 1)
    t = torch.tensor([avg_local], device=a1.device)
    dist.all_reduce(t, op=dist.ReduceOp.AVG)
    avg_s = float(t.item())

    eff_tps = (args.tokens_per_rank * ws) / max(avg_s, 1e-9)

    if r == 0:
        print(
            f"[DeepEP-HT A2A] world_size={ws}, experts={args.num_experts}, topk={args.topk}, "
            f"hidden={args.hidden_size}, dtype={args.dtype}, num_sms={args.num_sms}",
            flush=True,
        )
        print(
            f"  tokens_per_rank={args.tokens_per_rank}, warmup={args.warmup}, iters={args.iters}",
            flush=True,
        )
        print(f"  avg latency per pass = {avg_s * 1000:.3f} ms", flush=True)
        print(f"  effective routed tokens/s (global) ≈ {eff_tps:,.0f}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

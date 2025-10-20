import argparse
import os
import time

import torch
import torch.distributed as dist

# DeepEP HT prepare/finalize and modular kernel wrapper
from vllm.model_executor.layers.fused_moe.deepep_ht_prepare_finalize import (
    DeepEPHTPrepareAndFinalize,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEModularKernel,
)

# EP group / All2All manager for DeepEP HT
from vllm.distributed import get_ep_group
from vllm.distributed.device_communicators.all2all import (
    DeepEPHTAll2AllManager,
)


def ddp_barrier():
    dist.barrier(device_ids=[torch.cuda.current_device()])


def main():
    # ---------------------- CLI ----------------------
    ap = argparse.ArgumentParser("DeepEP-HT all-to-all micro-benchmark (no model)")
    ap.add_argument("--tokens", type=int, default=32768, help="N: number of tokens")
    ap.add_argument("--hidden", type=int, default=4096, help="H: token hidden size")
    ap.add_argument("--experts", type=int, default=32, help="E: global number of experts")
    ap.add_argument("--topk", type=int, default=2, help="K: experts per token (routing)")
    ap.add_argument("--warmup", type=int, default=5, help="warmup iterations")
    ap.add_argument("--iters", type=int, default=20, help="timed iterations")
    args = ap.parse_args()

    # ---------------- DDP init (torchrun) ----------------
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    # Ensure HT backend is selected (optional: you can also set this in shell)
    os.environ.setdefault("VLLM_ALL2ALL_BACKEND", "deepep_high_throughput")

    # ----------------  Build DeepEP HT buffer via manager ----------------
    # Manager needs a CPU group; we use the EP group's CPU view.
    # In this micro-benchmark the whole world forms one EP group.
    ep_cpu_group = get_ep_group().cpu_group
    ht_mgr = DeepEPHTAll2AllManager(ep_cpu_group)

    # HT manager takes no kwargs; it computes defaults internally and returns deep_ep.Buffer
    deepep_buffer = ht_mgr.get_handle({})

    # ----------------  Construct HT Prepare/Finalize  ----------------
    ep_size = world_size
    dp_size = 1  # we only benchmark EP A2A here
    assert (
        args.experts % ep_size == 0
    ), f"--experts ({args.experts}) must be divisible by world_size ({ep_size})"
    num_local_experts = args.experts // ep_size
    rank_expert_offset = rank * num_local_experts

    pf = DeepEPHTPrepareAndFinalize(
        buffer=deepep_buffer,        # REQUIRED: deep_ep.Buffer from the manager
        num_dispatchers=ep_size,     # number of EP ranks
        dp_size=dp_size,             # DP size in this benchmark
        rank_expert_offset=rank_expert_offset,
    )

    # -------------- Dummy expert compute (no GEMM) --------------
    class DummyExperts:
        # Modular kernel will call this between prepare() and finalize().
        # We return the input so the run measures almost only A2A traffic.
        def __call__(
            self,
            *,
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            inplace,
            activation,
            apply_router_weight_on_input,
            global_num_experts,
            expert_map,
        ):
            return hidden_states

    # Modular kernel = (prepare/finalize) + (experts), no shared experts here.
    kernel = FusedMoEModularKernel(pf, experts=DummyExperts(), shared_experts=None)

    # ----------------- Synthetic routing & data ------------------
    N, H, E, K = args.tokens, args.hidden, args.experts, args.topk
    x = torch.randn(N, H, device="cuda", dtype=torch.float16)
    topk_ids = torch.randint(0, E, (N, K), device="cuda", dtype=torch.int32)
    topk_wts = torch.rand(N, K, device="cuda", dtype=torch.float16)

    # ----------------------- Warmup -----------------------
    ddp_barrier()
    for _ in range(args.warmup):
        _ = kernel(
            hidden_states=x,
            w1=None,
            w2=None,
            topk_weights=topk_wts,
            topk_ids=topk_ids,
            inplace=True,
            activation="silu",
            apply_router_weight_on_input=False,
            global_num_experts=E,
            expert_map=None,
        )
    ddp_barrier()

    # ----------------------- Timing -----------------------
    torch.cuda.synchronize()
    times = []
    for _ in range(args.iters):
        t0 = time.time()
        _ = kernel(
            hidden_states=x,
            w1=None,
            w2=None,
            topk_weights=topk_wts,
            topk_ids=topk_ids,
            inplace=True,
            activation="silu",
            apply_router_weight_on_input=False,
            global_num_experts=E,
            expert_map=None,
        )
        torch.cuda.synchronize()
        times.append(time.time() - t0)
    ddp_barrier()

    # ----------------------- Report -----------------------
    import numpy as np

    t = np.array(times, dtype=float)
    avg = t.mean()
    p50 = np.percentile(t, 50)
    p95 = np.percentile(t, 95)

    # Rough one-pass bandwidth estimate (per rank, single direction ×2):
    # bytes ≈ N * H * 2 (fp16) * 2 (dispatch+combine)
    bytes_moved = N * H * 2 * 2
    gbps = bytes_moved / avg / 1e9

    if rank == 0:
        print(f"[DeepEP-HT A2A] world_size={world_size}  E={E}  K={K}  N={N}  H={H}")
        print(f"  avg={avg:.4f}s  p50={p50:.4f}s  p95={p95:.4f}s")
        print(f"  approx one-pass bandwidth ≈ {gbps:.2f} GB/s (per rank, rough)")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

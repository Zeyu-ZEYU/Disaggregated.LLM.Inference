# vLLM All-to-All with DeepEP

- Build vLLM with "pip install -e .". So, we can edit the code directly.
- Install DeepEP dependencies.
    - In /vllm/tools/ep_kernels, run the following cmd according to what your GPU devices are.
        ```bash
        # for hopper
        TORCH_CUDA_ARCH_LIST="9.0" bash install_python_libraries.sh
        # for blackwell
        TORCH_CUDA_ARCH_LIST="10.0" bash install_python_libraries.sh
        ```
    - Additional step for multi-node deployment:
        ```bash
        sudo bash configure_system_drivers.sh
        sudo reboot # Reboot is required to load the new driver
        ```
- Test DeepEP/bench_deepep_ht_a2a.py:
    - Case 1: single node with 4 GPUs
        ```bash
        export TORCH_CUDA_ARCH_LIST="8.0"        # A100
        export NCCL_IB=1                         # 确保使用 IB/RDMA
        # 可选调优：
        # export NCCL_NET_GDR_LEVEL=5
        # export NCCL_DEBUG=INFO
        torchrun --nproc_per_node=4 bench_deepep_ht_a2a.py.py \
        --num-experts 64 --tokens-per-rank 32768 --hidden-size 4096 --dtype bf16
        ```
    - Case 2: two nodes each with 4 GPUs
        ```bash
        # 在两台机器上都设置同一地址与端口
        export MASTER_ADDR=<master_ip_or_hostname>
        export MASTER_PORT=29501
        export NCCL_IB=1
        # 可选：NCCL_SOCKET_IFNAME/NCCL_IB_HCA/NCCL_NET_GDR_LEVEL...

        # 节点 0：
        torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 bench_deepep_ht_a2a.py.py \
        --num-experts 64 --tokens-per-rank 32768 --hidden-size 4096

        # 节点 1：
        torchrun --nnodes=2 --nproc_per_node=4 --node_rank=1 bench_deepep_ht_a2a.py.py \
        --num-experts 64 --tokens-per-rank 32768 --hidden-size 4096
        ```

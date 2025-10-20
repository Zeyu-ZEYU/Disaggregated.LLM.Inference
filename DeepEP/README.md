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
- Test DeepEP/bench_deepep_ht_a2a.py within a node first:
    ```bash
    export VLLM_ALL2ALL_BACKEND=deepep_high_throughput
    torchrun --nproc_per_node=4 bench_deepep_ht_a2a.py \
    --tokens 32768 --hidden 4096 --experts 32 --topk 2
    ```

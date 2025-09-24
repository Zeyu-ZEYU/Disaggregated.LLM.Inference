# Instructions for Setting Up Disaggregated LLM Inference

## Install vLLM
git clone https://github.com/vllm-project/vllm.git

- Install vLLM following https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html#requirements
- Run 1 prefill vLLM instance on server1_ip port 30000 by "vllm serve meta-llama/Llama-3.1-8B --host server1_ip --port 30000"
- Run 1 decode vLLM instance on server2_ip port 30000 by "vllm serve meta-llama/Llama-3.1-8B --host server2_ip --port 30000"

## Run disaggregated LLM inference using proxy
Launch PD disaggregation by configuring 1 prefill instance and 1 decode instance.

In https://docs.vllm.ai/en/latest/examples/online_serving/disaggregated_serving.html, just use disagg_proxy_demo.py.
launch this proxy demo through:
```bash
python3 examples/online_serving/disaggregated_serving/disagg_proxy_demo.py  \
    --model $model_name \
    --prefill server1_ip:30000 \
    --decode server2_ip:30000 \
    --port 8000
```

Once the proxy starts, it works as if it is a vLLM server running on port 8000.
Then start a client to give prompts to the server via port 8000. To run a client, see https://docs.vllm.ai/en/latest/examples/online_serving/api_client.html and https://docs.vllm.ai/en/latest/examples/online_serving/openai_completion_client.html.

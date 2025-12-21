# vLLM + LMCache（PD 分离）按 layer 推送 KV 到 decode 本机 CPU 的实现说明

这个文档说明在 **vLLM + LMCache 的 PD 分离（prefill / decode split）场景**下，如何通过一组**尽量简单、可解释、工程侵入性低**的修改，实现：

- prefill 侧 **每一层（layer）完成后立即推送 KV**
- KV 直接落到 **对应 decode instance 的本机 CPU pinned 内存（LMCache 管理）**
- 尽量让 **KV 传输与 prefill 计算 overlap**
- decode 侧逻辑基本不变（不引入额外同步/通知）

示例模型以 Qwen 这类 decoder-only 模型为背景。

---

## 1. 默认 PD 分离流程回顾（为什么要改）

在默认的 vLLM + LMCache PD 分离实现中：

- prefill 节点在 GPU 上计算 prompt，对应 KV 写入 vLLM 的 paged KV cache。
- KV 的发送触发点在 **prefill forward 结束后**，通过
  LMCacheConnectorV1.wait_for_save → lmcache_engine.store(..., transfer_spec)
  一次性把 KV 发/存出去。
- decode 节点在进入 decode forward 前调用 start_load_kv，从远端拉取 KV，再放入 decode 的 vLLM KV cache。

这个流程的主要问题是：

- KV 传输基本发生在 prefill 末尾，**与 prefill 计算 overlap 很少**。
- decode 侧如果从 remote CPU 拉 KV，路径更慢、更不稳定。

---

## 2. 代码改动的设计目标

这次修改的目标非常明确，也刻意保持克制避免动过多代码：

1. **prefill 侧按 layer 推送 KV**
2. **避免重复发送（按 chunk 对齐）**
3. **decode 侧 KV 落本机 CPU pinned 内存**
4. **decode 逻辑尽量不动**

---

## 3. 关于 hidden state

在当前这条 vLLM + LMCache 的 PD 分离 KV connector 路径中，数据面默认只处理 **KV cache**。
因此本次改动 **不额外处理 hidden state**。

---

## 4. 核心改动点概览

### 4.1 只在 prefill 侧启用 use_layerwise

layer-wise 行为只对 **kv_producer（prefill）** 有意义。
在 vLLM 的 LMCache adapter 初始化阶段强制开启。

### 4.2 save_kv_layer：按 layer 发送 KV

- 维护 per-request 的“已发送边界”
- 只发送新完成的完整 chunk
- 每层触发一次 PD send

### 4.3 store_layer 透传 transfer_spec

保证 layer-wise store 也能走 PD/disagg 路径。

### 4.4 decode 侧落点改为 CPU pinned

PD receiver 使用 LMCache CPU pinned allocator，decode load 时为本机 CPU → GPU copy。

---

## 5. decode 不开启 use_layerwise 的原因

decode 只需要尽早拿到 KV，不需要 layer 级同步。
保持 decode 逻辑简单更稳妥。

---

## 6. tail（不足一个 chunk）的处理

不足一个 chunk 的 token 在 prefill 结束时由 wait_for_save 一次性补发，保证正确性。

---

## 7. 最终效果

- prefill：每层尽早发送 KV
- decode：KV 落decode节点本机 CPU
- 传输与计算 overlap 更充分

## 8. 实际代码修改位置浏览

改动了主要是三个文件
- LMCache/lmcache/integration/vllm/vllm_v1_adapter.py
    - 要注意的是，vLLM的项目代码里，也有vllm/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_integration/vllm_v1_adapter.py
    - 如果要使用vLLM native的LMCache（也就是use_native=True）则应该修改vLLM侧的vllm_v1_adapter.py。目前这个repo里改动的是LMCache侧的文件，因为vLLM那里默认不用native的LMCache（use_native=False）。
- LMCache/lmcache/v1/cache_engine.py
- LMCache/lmcache/v1/storage_backend/pd_backend.py

### 1. vllm_v1_adapter
先设置use_layerwise
```py
# Force-enable layerwise flushing on the producer (prefill) side for PD disaggregation.
if self.kv_role == "kv_producer":
    config.use_layerwise = True
    logger.info(
        "Force enabled lmcache.use_layerwise=True for kv_producer (prefill)."
    )
```

对LMCacheConnectorV1Impl这个类添加属性
```py
# Track flushed boundary (absolute token index) per request.
# Always aligned to chunk boundary (i.e., skip + k * chunk_size).
self._sent_upto_aligned: dict[str, int] = {}

# Track the last observed "prefill end" per request (absolute token index).
# Used to flush tail (<1 chunk) at wait_for_save.
self._last_end_seen: dict[str, int] = {}
```

对LMCacheConnectorV1Impl这个类修改方法save_kv_layer()以将每层发送KV的位置放在该层MoE all-to-all结束之后
```py
@_lmcache_nvtx_annotate
def save_kv_layer(
    self,
    layer_name: str,
    kv_layer: torch.Tensor,
    attn_metadata: "AttentionMetadata",
    **kwargs,
) -> None:
    """Start saving a layer of KV cache from vLLM's paged buffer to the connector."""
    assert self.lmcache_engine is not None

    if not self.use_layerwise:
        return

    if self.kv_role == "kv_consumer":
        # Don't do save if the role is kv_consumer
        return

    if self._parent._connector_metadata is None:
        logger.warning(
            "In connector.save_kv_layer, but the connector metadata is None"
        )
        return

    connector_metadata = self._parent._get_connector_metadata()
    assert isinstance(connector_metadata, LMCacheConnectorMetadata)

    assert len(self.kv_caches) > 0

    # ---- NEW: init per-request flush bookkeeping (lazy init to avoid crashes) ----
    # Track how many tokens have been flushed (aligned to chunk boundary) per request.
    if not hasattr(self, "_sent_upto_aligned"):
        self._sent_upto_aligned = {}  # type: ignore[attr-defined]

    kvcaches = list(self.kv_caches.values())

    if self.current_layer == 0:
        self.layerwise_storers = []
        is_first = True

        for idx, request in enumerate(connector_metadata.requests):
            save_spec = request.save_spec
            if save_spec is None or not save_spec.can_save:
                continue

            token_ids = request.token_ids
            assert isinstance(token_ids, list)

            slot_mapping = request.slot_mapping
            assert isinstance(slot_mapping, torch.Tensor)
            assert len(slot_mapping) == len(token_ids)

            # TODO: have a pre-allocated buffer to hold the slot_mappings
            slot_mapping = slot_mapping.to(self.device)

            if self.kv_role == "kv_producer":
                skip_leading_tokens = 0
            else:
                skip_leading_tokens = save_spec.skip_leading_tokens
                if skip_leading_tokens == len(token_ids):
                    continue  # skip this request

                # Align to lmcache chunk size (existing behavior)
                skip_leading_tokens = (
                    skip_leading_tokens
                    // self._lmcache_chunk_size
                    * self._lmcache_chunk_size
                )

            # ---- NEW: only flush NEW FULL chunks to avoid re-sending boundary chunks ----
            # We assume that by the time save_kv_layer is called for this request,
            # KV for all tokens after skip_leading_tokens is valid (same assumption as original code).
            end = len(token_ids)

            # Align end down to chunk boundary.
            end_aligned = (
                skip_leading_tokens
                + ((end - skip_leading_tokens) // self._lmcache_chunk_size)
                * self._lmcache_chunk_size
            )

            prev_sent = self._sent_upto_aligned.get(
                request.req_id, skip_leading_tokens
            )  # type: ignore[attr-defined]
            # Safety clamp
            if prev_sent < skip_leading_tokens:
                prev_sent = skip_leading_tokens
            if prev_sent > end_aligned:
                prev_sent = end_aligned

            # No new full chunk -> create a no-op storer for this request (so later loop stays simple)
            if prev_sent >= end_aligned:
                # Still append a dummy generator that yields immediately.
                def _noop_gen():
                    if False:
                        yield None

                self.layerwise_storers.append(_noop_gen())
                continue

            # Build mask over the full token_ids length, but only mark [prev_sent, end_aligned).
            store_mask = torch.zeros(len(token_ids), dtype=torch.bool)
            store_mask[prev_sent:end_aligned] = True
            # Ensure we never store before skip_leading_tokens.
            if skip_leading_tokens > 0:
                store_mask[:skip_leading_tokens] = False

            logger.info(
                "Layerwise storing KV cache for %d NEW tokens out of %d tokens "
                "(skip_leading_tokens=%d, prev_sent=%d, end_aligned=%d) for request %s",
                int(store_mask.sum().item()),
                len(token_ids),
                skip_leading_tokens,
                prev_sent,
                end_aligned,
                request.req_id,
            )

            # ---- NEW: make layerwise storing compatible with disagg spec ----
            # request.disagg_spec should exist in PD-disagg path; use getattr to be safe.
            transfer_spec = getattr(request, "disagg_spec", None)

            layerwise_storer = self.lmcache_engine.store_layer(
                token_ids,
                mask=store_mask,
                kvcaches=kvcaches,
                slot_mapping=slot_mapping,
                offset=skip_leading_tokens,
                sync=is_first,
                req_id=request.req_id,
                transfer_spec=transfer_spec,
            )
            self.layerwise_storers.append(layerwise_storer)

            # Update sent boundary immediately (we're committing to sending these full chunks).
            self._sent_upto_aligned[request.req_id] = end_aligned  # type: ignore[attr-defined]

            if is_first:
                is_first = False

    # Advance all per-request layerwise storers once for this layer.
    for layerwise_storer in self.layerwise_storers:
        next(layerwise_storer)

    self.current_layer += 1
```

然后修改LMCacheConnectorV1Impl这个类修改方法wait_for_save()对最后不足一个chunk的数据进行处理
```py
if self.use_layerwise:
    # Flush tail tokens (<1 chunk) once at the end, to guarantee correctness.
    for request in self._get_requests_need_store_somehow():
        token_ids = request.token_ids
        skip = (
            request.skip_leading_tokens
        )  # If our request does not have this field, then use the original one.
        end = self._last_end_seen.get(request.req_id, len(token_ids))
        sent = self._sent_upto_aligned.get(request.req_id, skip)

        if sent >= end:
            continue  # no tail

        # Tail mask: [sent, end)
        mask_len = len(token_ids) - skip
        tail_mask = torch.zeros(mask_len, dtype=torch.bool, device="cpu")
        tail_mask[(sent - skip) : (end - skip)] = True

        slot_mapping = request.slot_mapping
        if len(slot_mapping) == len(token_ids):
            slot_mapping = slot_mapping[skip:]

        # IMPORTANT: flush all layers at once using store(), simplest and avoids per-layer loop.
        self.lmcache_engine.store(
            token_ids,
            mask=tail_mask,
            kvcaches=request.kvcaches_all_layers,  # you already have full kvcaches in wait_for_save path
            slot_mapping=slot_mapping,
            offset=skip,
            transfer_spec=request.disagg_spec,
            request_configs=request.request_configs,
            req_id=request.req_id,
        )

        self._sent_upto_aligned[request.req_id] = end
```

定义一个get_chunk_size的方法
```py
def _get_chunk_size(self) -> int:
    # Try common places; adjust if your config uses a different name.
    for attr in ("chunk_size", "tokens_per_chunk", "kv_chunk_size"):
        if hasattr(self.lmcache_engine, attr):
            return int(getattr(self.lmcache_engine, attr))
        if hasattr(self.lmcache_engine, "config") and hasattr(
            self.lmcache_engine.config, attr
        ):
            return int(getattr(self.lmcache_engine.config, attr))
    # Conservative default (you should replace this with your real chunk size if different)
    return 256
```

### 2. cache_engine
cache_engine处需要实现layerwise的KV传输，需要提供transfer_spec数据告知PD context
```py
for layer_id in range(self.num_layers):
    yield
    next(mem_obj_generator)
    # NOTE: enable PD/disagg on layer-wise storing as well.
    transfer_spec = kwargs.get("transfer_spec", None)
    self.storage_manager.batched_put(
        keys[layer_id],
        memory_objs[layer_id],
        transfer_spec=transfer_spec,
    )
```

### 3. pd_backend
这个主要是将KV发到decode节点的CPU内存中

如果是decode，则后端为CPU内存空间
```py
if self.pd_config.role == "receiver":
    buf_alloc = self.memory_allocator.cpu_allocator
else:
    buf_alloc = self.memory_allocator.gpu_allocator

self.transfer_channel = CreateTransferChannel(
    async_mode=False,
    channel_type=config.transfer_channel,
    role=self.pd_config.role,
    buffer_ptr=buf_alloc.buffer_ptr,
    buffer_size=buf_alloc.buffer_size,
    align_bytes=buf_alloc.align_bytes,
    tp_rank=self.tp_rank,
    peer_init_url=peer_init_url,
    backends=config.nixl_backends,
)
```

PDBackend的initialize_allocator的修改
```py
# ✅ receiver landing zone: CPU pinned pool
    if config.pd_role == "receiver":
        paged_mem_allocator.init_cpu_memory_allocator(
            config.pd_buffer_size,
            [torch.Size(metadata.kv_shape)],
            [metadata.kv_dtype],
            MemoryFormat.KV_2LTD,
            numa_mapping=None,
        )
    else:
        paged_mem_allocator.init_gpu_memory_allocator(
            config.pd_buffer_size,
            [torch.Size(metadata.kv_shape)],
            [metadata.kv_dtype],
            MemoryFormat.KV_2LTD,
            corrected_device,
        )
```

然后修改allocator和batched_allocator两个函数
```py
# allocator()
allocator_type = "cpu" if self.pd_config.role == "receiver" else "gpu"
return self.memory_allocator.allocate(
    shapes, dtypes, fmt=fmt, allocator_type=allocator_type
)

# batched_allocator()
allocator_type = "cpu" if self.pd_config.role == "receiver" else "gpu"
return self.memory_allocator.batched_allocate(
    shapes, dtypes, batch_size, fmt, allocator_type=allocator_type
)
```

"""Opt-in eager fixture. This module does not provide a serving scheduler."""
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
import os
import time

from poc_ouro_parallelism_deep_discovery.c07_fixture import (
    C07ValidationError, check_local_validation, clone_metadata, compare_bf16_words,
    lease_pages, observe_dense_padding, payload_bytes, rejected_cpu_batch, require_bitwise, selectors,
    select_decode_metadata, select_v2_padding, snapshot_v2_batch, token_addresses, v2_batch_rejection, V2CaptureHooks)
from poc_ouro_parallelism_deep_discovery.contract import sha, write
from poc_ouro_parallelism_deep_discovery.worker_probe import CampaignProbe


class C07FixtureProbe(CampaignProbe):
    def c07_capture_start(self, position, row=0, depth=0):
        import torch
        from vllm import envs
        from vllm.distributed import get_tensor_model_parallel_world_size
        runner = self.model_runner
        core = self.get_model().model
        if getattr(self, '_c07', None) is not None:
            raise RuntimeError('The fixture is one-shot. Start a new guarded job.')
        if (not self.vllm_config.model_config.enforce_eager
                or int(self.vllm_config.compilation_config.mode) != 0
                or core.total_ut_steps != 4 or core.start_layer != 0
                or core.end_layer != len(core.layers)):
            raise RuntimeError('The fixture requires the complete eager R4 core and PP1.')
        if type(runner).__module__ != 'vllm.v1.worker.gpu.model_runner':
            raise RuntimeError('INFRASTRUCTURE_BLOCKED: this capture adapter requires the native V2 model runner.')
        if type(core).__module__ != 'vllm.model_executor.models.ouro' or type(core).__name__ != 'OuroModel':
            raise RuntimeError('The padding interpretation requires the original native dense Ouro core.')
        if envs.VLLM_BATCH_INVARIANT:
            raise RuntimeError('The bounded padding proof requires the original default unquantized GEMM route.')
        tp = get_tensor_model_parallel_world_size()
        selected = selectors(tp, row, depth)
        for layer in core.layers:
            if (type(layer).__module__ != 'vllm.model_executor.models.ouro'
                    or type(layer).__name__ != 'OuroDecoderLayer'
                    or type(layer.mlp).__module__ != 'vllm.model_executor.models.ouro'
                    or type(layer.mlp).__name__ != 'OuroMLP'):
                raise RuntimeError('The padding interpretation does not accept a different decoder or MoE route.')
            for projection in [layer.self_attn.qkv_proj, layer.self_attn.o_proj,
                               layer.mlp.gate_up_proj, layer.mlp.down_proj]:
                if (projection.quant_config is not None
                        or type(projection.quant_method).__name__ != 'UnquantizedLinearMethod'
                        or type(projection.quant_method).__module__ != 'vllm.model_executor.layers.linear'):
                    raise RuntimeError('The padding interpretation requires original unquantized dense projections.')
            if len(layer.self_attn.attn) != 4:
                raise RuntimeError('The physical layer does not retain all four original Attention objects.')
            for attention in layer.self_attn.attn:
                cache = attention.kv_cache
                if (not torch.is_tensor(cache) or cache.ndim != 4 or cache.shape[2:] != (16, 256)
                        or cache.dtype != torch.bfloat16 or cache.shape[1] != 16 // tp
                        or getattr(attention.impl, 'vllm_flash_attn_version', None) != 3):
                    raise RuntimeError('The fixture requires original BF16 FA3 cache allocations before capture.')
        self._c07 = state = dict(phase='ARMED', position=int(position), selected=selected,
                                inputs=[], outputs=[], hooks=[], rejected=[], active=False,
                                capture_host_enqueue_ns=0, calls_seen=0, pending_batch=None,
                                pending_padding=None, runner_class=f'{type(runner).__module__}.{type(runner).__qualname__}')

        def first_layer(module, args):
            if not state['active']:
                return
            positions, hidden, current_ut, residual = args
            if (current_ut != len(state['inputs']) or residual is not None
                    or hidden.shape != (4, 2048) or hidden.dtype != torch.bfloat16):
                raise RuntimeError('The native pass-input order or residual boundary differs.')
            state['inputs'].append(hidden.detach().clone())

        def norm_output(module, args, output):
            if not state['active']:
                return
            if len(state['outputs']) >= len(state['inputs']) or not isinstance(output, tuple):
                raise RuntimeError('The native recurrent normalization order differs.')
            state['outputs'].append(output[0].detach().clone())

        state['hooks'] = [core.layers[0].register_forward_pre_hook(first_layer),
                          core.norm.register_forward_hook(norm_output)]

        def observed(*args, **kwargs):
            from vllm.forward_context import get_forward_context
            context = get_forward_context()
            metadata = context.attn_metadata
            positions = kwargs.get('positions', args[1] if len(args) > 1 else None)
            batch, padding = state['pending_batch'], state['pending_padding']
            state['pending_batch'], state['pending_padding'] = None, None
            state['calls_seen'] += 1
            reason = self._c07_suitability(context, metadata, positions, batch, padding)
            if state['phase'] != 'ARMED' or reason is not None:
                if state['phase'] == 'ARMED':
                    state['rejected'].append(rejected_cpu_batch(state['calls_seen'], reason, batch))
                return state['original'](*args, **kwargs)
            start = time.perf_counter_ns()
            state['capture_call'] = state['calls_seen']
            memo = {}
            # Do not clone no_compile_layers. It owns the original Attention modules.
            state['context'] = replace(context,
                attn_metadata=clone_metadata(context.attn_metadata, torch.is_tensor, memo),
                slot_mapping=clone_metadata(context.slot_mapping, torch.is_tensor, memo),
                is_padding=clone_metadata(context.is_padding, torch.is_tensor, memo),
                additional_kwargs=clone_metadata(context.additional_kwargs, torch.is_tensor, memo))
            state['metadata_tensor_ownership'] = []
            seen = set()

            def record_clones(value, path):
                if id(value) in seen:
                    return
                seen.add(id(value))
                if torch.is_tensor(value):
                    cloned = memo[id(value)]
                    if value.numel() and cloned.data_ptr() == value.data_ptr():
                        raise RuntimeError('A metadata clone borrows the original mutable tensor address.')
                    state['metadata_tensor_ownership'].append(dict(path=path,
                        source_pointer=value.data_ptr(), captured_pointer=cloned.data_ptr(),
                        source_storage_offset=value.storage_offset(), captured_storage_offset=cloned.storage_offset(),
                        shape=list(value.shape), source_stride=list(value.stride()),
                        captured_stride=list(cloned.stride()), dtype=str(value.dtype),
                        bytes=value.numel() * value.element_size()))
                elif is_dataclass(value) and not isinstance(value, type):
                    for field in fields(value):
                        record_clones(getattr(value, field.name), path + '.' + field.name)
                elif isinstance(value, dict):
                    for name, item in value.items():
                        record_clones(item, path + '.' + str(name))
                elif isinstance(value, (list, tuple)):
                    for index, item in enumerate(value):
                        record_clones(item, path + '.' + str(index))

            record_clones(context.attn_metadata, 'attn_metadata')
            record_clones(context.slot_mapping, 'slot_mapping')
            record_clones(context.is_padding, 'is_padding')
            state['positions'] = positions.detach().clone()
            state['request_ids'] = list(batch['request_ids'])
            state['captured_batch'] = batch
            state['native_metadata'] = state['context'].attn_metadata
            first = next(iter(state['native_metadata'].values()))
            state['source_table'] = first.block_table.detach().clone()
            state['prefixes'] = []
            p = state['position']
            token_positions = torch.arange(p, device=positions.device)
            for source_row, source_depth in selected:
                ids = state['source_table'][source_row, token_positions // 16].long()
                values = []
                for layer in core.layers:
                    cache = layer.self_attn.attn[source_depth].kv_cache
                    # Advanced indexing allocates an owned snapshot. Do not add a second payload copy.
                    values.append(cache[ids, :, token_positions % 16, :].detach())
                state['prefixes'].append(values)
            state['active'] = True
            try:
                result = state['original'](*args, **kwargs)
                if len(state['inputs']) != 4 or len(state['outputs']) != 4:
                    raise RuntimeError('The native invocation did not expose all four complete passes.')
                state['current'] = []
                for source_row, source_depth in selected:
                    # Keep a one-element device index. A scalar device index can cause a host read.
                    block = state['source_table'][source_row, p // 16:p // 16 + 1].long()
                    state['current'].append([layer.self_attn.attn[source_depth].kv_cache[
                        block, :, p % 16, :].squeeze(0).detach() for layer in core.layers])
                state['phase'] = 'CAPTURED'
                return result
            except BaseException:
                state['phase'] = 'CAPTURE_FAILED'
                raise
            finally:
                state['active'] = False
                state['capture_host_enqueue_ns'] = time.perf_counter_ns() - start

        def prepared(batch):
            state['pending_batch'] = snapshot_v2_batch(batch)
            state['pending_padding'] = batch.is_padding

        try:
            state['capture_hooks'] = V2CaptureHooks(runner, prepared, observed)
            state['original'] = state['capture_hooks'].original_forward
            state['capture_hooks'].install()
        except BaseException:
            for hook in state['hooks']:
                hook.remove()
            state['hooks'] = []
            raise
        return dict(phase=state['phase'], selected=selected, position=position,
                    scope='One actual native C4 cached decode. Fixture capture is not timing.')

    def _c07_suitability(self, context, metadata, positions, batch, padding):
        import torch
        from vllm.config import CUDAGraphMode
        from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
        state = self._c07
        if (not isinstance(metadata, dict) or not metadata or not isinstance(context.slot_mapping, dict)
                or context.cudagraph_runtime_mode != CUDAGraphMode.NONE or context.dp_metadata is not None
                or context.ubatch_slices is not None
                or context.all_moe_layers is not None or context.additional_kwargs):
            return 'UNSUPPORTED_FORWARD_CONTEXT'
        batch_reason = v2_batch_rejection(batch, state['position'])
        if batch_reason is not None:
            return batch_reason
        if positions is None or tuple(positions.shape) != (4,):
            return 'NOT_FOUR_UNPADDED_ROWS'
        if (not torch.is_tensor(context.is_padding) or context.is_padding.dtype != torch.bool
                or tuple(context.is_padding.shape) != (4,) or padding is None
                or context.is_padding.data_ptr() != padding.data_ptr()):
            return 'UNSUPPORTED_V2_PADDING_IDENTITY'
        for meta in metadata.values():
            if (not isinstance(meta, FlashAttentionMetadata) or meta.num_actual_tokens != 4
                    or meta.max_query_len != 1 or meta.max_seq_len != state['position'] + 1
                    or meta.use_cascade or meta.common_prefix_len != 0 or meta.causal is not True
                    or meta.num_prefill_reqs or meta.num_prefill_tokens
                    or meta.max_dcp_context_kv_len not in (None, 0) or meta.dcp_context_kv_lens is not None
                    or meta.prefix_scheduler_metadata is not None or meta.mm_prefix_range_tensor is not None
                    or meta.rswa_prefix_lens is not None or meta.rswa_window_tensor is not None
                    or meta.sliding_window not in (None, (-1, -1))):
                return 'UNSUPPORTED_OR_NOT_CACHED_DECODE_METADATA'
        # Tensor contents are checked after the request. No host synchronization occurs here.
        if not torch.is_tensor(positions):
            return 'POSITIONS_NOT_A_TENSOR'
        return None

    def c07_capture_finish(self, driver_quiescent=False):
        import torch
        from vllm.distributed import get_tensor_model_parallel_rank
        state = self._c07
        runner = self.model_runner
        state['capture_hooks'].restore()
        for hook in state['hooks']:
            hook.remove()
        state['hooks'] = []
        if state['phase'] != 'CAPTURED':
            return dict(phase=state['phase'], rejected=state['rejected'], calls_seen=state['calls_seen'])
        if not driver_quiescent or getattr(runner, 'execute_model_state', None) is not None:
            raise RuntimeError('The driver and native worker must finish all serving work before the lease.')
        torch.cuda.synchronize()
        state['padding_observation'] = observe_dense_padding(state['context'].is_padding.cpu().tolist())
        p = state['position']
        if state['positions'].cpu().tolist() != [p] * 4:
            raise RuntimeError('The captured positions differ from the selected cached-decode boundary.')
        metadata = state['native_metadata']
        first = next(iter(metadata.values()))
        expected_fields = dict(query_start_loc=[0, 1, 2, 3, 4], seq_lens=[p + 1] * 4)
        for name, expected in expected_fields.items():
            if getattr(first, name).cpu().tolist() != expected:
                raise RuntimeError('The captured query or sequence lengths differ.')
        core = self.get_model().model
        cache_descriptions = []
        for layer_index, layer in enumerate(core.layers):
            for depth, attention in enumerate(layer.self_attn.attn):
                cache = attention.kv_cache
                if (cache.ndim != 4 or cache.shape[2:] != (16, 256) or cache.dtype != torch.bfloat16
                        or attention.impl.vllm_flash_attn_version != 3
                        or attention.impl.num_heads != cache.shape[1]
                        or attention.impl.num_kv_heads != cache.shape[1]
                        or attention.impl.head_size != 128):
                    raise RuntimeError('The fixture requires native packed BF16 FA3 MHA cache pages.')
                name = attention.layer_name
                if name not in metadata or state['context'].no_compile_layers[name] is not attention:
                    raise RuntimeError('The captured native context does not own the original Attention module.')
                candidate = metadata[name]
                for field in fields(first):
                    left, right = getattr(first, field.name), getattr(candidate, field.name)
                    equal = torch.equal(left, right) if torch.is_tensor(left) else left == right
                    if not equal:
                        raise RuntimeError('Depth slots use different metadata. This fixture cannot remap them.')
                mapping = state['context'].slot_mapping[name]
                expected = first.block_table[:, p // 16].long() * 16 + p % 16
                if not torch.equal(mapping.long(), expected):
                    raise RuntimeError('The native slot mapping differs from its block-table addresses.')
                cache_descriptions.append(dict(layer=layer_index, depth=depth, name=name,
                    pointer=cache.data_ptr(), storage_offset=cache.storage_offset(), shape=list(cache.shape),
                    storage_pointer=cache.untyped_storage().data_ptr(),
                    storage_bytes=cache.untyped_storage().nbytes(),
                    stride=list(cache.stride()), dtype=str(cache.dtype), element_size=cache.element_size(),
                    address_span_bytes=(sum((size - 1) * stride for size, stride
                                             in zip(cache.shape, cache.stride())) + 1) * cache.element_size(),
                    payload_bytes=cache.numel() * cache.element_size()))
        if len({item['pointer'] for item in cache_descriptions}) != 4 * len(core.layers):
            raise RuntimeError('The native depth cache slots alias. The required historical versions are not private.')
        if any(item['address_span_bytes'] != item['payload_bytes'] for item in cache_descriptions):
            raise RuntimeError('The cache uses an interleaved layer arena. This bounded fixture does not lease that layout.')
        spans = sorted((item['pointer'], item['pointer'] + item['address_span_bytes']) for item in cache_descriptions)
        if any(first_end > second_start for (_, first_end), (second_start, _) in zip(spans, spans[1:])):
            raise RuntimeError('Original depth cache address ranges overlap.')
        state['cache_descriptions'] = cache_descriptions
        for target_row, (_, source_depth) in enumerate(state['selected']):
            for layer_index, layer in enumerate(core.layers):
                prefix = state['prefixes'][target_row][layer_index]
                source = layer.self_attn.attn[source_depth].kv_cache
                if (prefix.untyped_storage().data_ptr() == source.untyped_storage().data_ptr()
                        or prefix.shape != (p, source.shape[1], 256)):
                    raise RuntimeError('A historical snapshot borrows native cache storage or has an incorrect shape.')
        state['metadata_ownership'] = dict(original_attention_modules_preserved=True,
            cloned_tensor_pointers_distinct=True, captured_field_count=len(fields(first)))
        for name, meta in metadata.items():
            for field in fields(meta):
                cloned = getattr(meta, field.name)
                # The original buffers can change after this invocation. The capture owns its copies.
                if torch.is_tensor(cloned) and cloned.numel() and cloned._base is not None:
                    raise RuntimeError('A captured metadata tensor still borrows a mutable native view.')
        state['phase'] = 'QUIESCENT_CAPTURED'
        state['quiescent_execute_original'] = runner.execute_model

        def reject_serving(*args, **kwargs):
            raise RuntimeError('The C07 fixture owns an exclusive quiescent-worker lease until teardown.')

        runner.execute_model = reject_serving
        return dict(phase=state['phase'], rank=get_tensor_model_parallel_rank(), calls_seen=state['calls_seen'],
                    rejected=state['rejected'], request_ids=state['request_ids'], selected=state['selected'],
                    cache_slots=len(cache_descriptions), all_depth_cache_pointers_distinct=True,
                    runner_class=state['runner_class'], captured_batch=state['captured_batch'],
                    padding_observation=state['padding_observation'],
                    capture_host_enqueue_ns=state['capture_host_enqueue_ns'],
                    clock_scope='Host enqueue interval. It is not completed device-copy time.')

    def c07_prepare_fixture(self):
        import torch
        with torch.inference_mode():
            return self._c07_prepare_fixture()

    def _c07_prepare_fixture(self):
        import torch
        from vllm.v1.attention.backends.flash_attn import get_scheduler_metadata
        state = self._c07
        if state['phase'] != 'QUIESCENT_CAPTURED':
            raise RuntimeError('The fixture requires a completed, quiescent capture.')
        start = time.perf_counter_ns()
        core = self.get_model().model
        p, selected = state['position'], state['selected']
        first = next(iter(state['native_metadata'].values()))
        used_columns = (p + 1 + 15) // 16
        protected = set(first.block_table[:, :used_columns].cpu().reshape(-1).tolist())
        if 0 in protected:
            raise RuntimeError('A real source token addresses the reserved null page.')
        if len(protected) != 4 * used_columns:
            raise RuntimeError('The four source requests do not own distinct non-prefix-shared pages.')
        cache = core.layers[0].self_attn.attn[0].kv_cache
        pages = lease_pages(cache.shape[0], protected, len(selected), p)
        table = first.block_table.new_zeros((len(selected), first.block_table.shape[1]))
        mapping = first.slot_mapping.new_empty((len(selected),))
        selected_rows = torch.tensor([row for row, _ in selected], device=cache.device, dtype=torch.long)
        positions = torch.full((len(selected),), p, device=cache.device, dtype=state['positions'].dtype)
        address_records = []
        for target_row, (source_row, source_depth) in enumerate(selected):
            table[target_row, :len(pages[target_row])] = torch.tensor(pages[target_row], device=cache.device,
                                                                     dtype=table.dtype)
            mapping[target_row] = pages[target_row][p // 16] * 16 + p % 16
            token_positions = torch.arange(p, device=cache.device)
            block_ids = table[target_row, token_positions // 16].long()
            for layer_index, layer in enumerate(core.layers):
                destination = layer.self_attn.attn[0].kv_cache
                prefix = state['prefixes'][target_row][layer_index]
                destination[block_ids, :, token_positions % 16, :] = prefix
                # Clear only the leased append address. Keep the historical prefix intact.
                destination[pages[target_row][p // 16], :, p % 16, :].zero_()
                gathered = destination[block_ids, :, token_positions % 16, :]
                if not torch.equal(gathered.view(torch.int16), prefix.view(torch.int16)):
                    raise RuntimeError('A remapped historical KV word differs from its captured version.')
                addresses = token_addresses(pages[target_row], p + 1)
                for token, (block, offset) in enumerate(addresses):
                    expected_address = destination.data_ptr() + (block * destination.stride(0)
                        + offset * destination.stride(2)) * destination.element_size()
                    if destination[block, :, offset, :].data_ptr() != expected_address:
                        raise RuntimeError('A leased token view has an unexpected physical address.')
                    if token not in (0, p - 1, p):
                        continue
                    address_records.append(dict(row=target_row, source_row=source_row, source_depth=source_depth,
                        layer=layer_index, token=token, block=block, offset=offset,
                        byte_address=expected_address))
        # TP4 preserves the captured FA3 schedule. TP1 rebuilds the smaller schedule with the native interface.
        meta = select_decode_metadata(first, num_rows=len(selected),
                       query_start_loc=torch.arange(len(selected) + 1, device=cache.device,
                                                    dtype=first.query_start_loc.dtype),
                       seq_lens=first.seq_lens.index_select(0, selected_rows).clone(),
                       block_table=table, slot_mapping=mapping)
        schedule_action = 'PRESERVED_CAPTURED_C4_SCHEDULE'
        if len(selected) == 1:
            schedule_action = 'NATIVE_SCHEDULER_REBUILT' if first.scheduler_metadata is not None else 'NATIVE_NO_AOT_POLICY_PRESERVED'
            if first.scheduler_metadata is not None:
                impl = core.layers[0].self_attn.attn[0].impl
                meta = replace(meta, scheduler_metadata=get_scheduler_metadata(
                    batch_size=1, max_seqlen_q=1, max_seqlen_k=meta.max_seq_len,
                    num_heads_q=impl.num_heads, num_heads_kv=impl.num_kv_heads, headdim=impl.head_size,
                    cache_seqlens=meta.seq_lens, qkv_dtype=cache.dtype, cu_seqlens_q=meta.query_start_loc,
                    page_size=16, causal=True, window_size=(-1, -1), num_splits=meta.max_num_splits))
        names = [layer.self_attn.attn[0].layer_name for layer in core.layers]
        state['fixture_context'] = replace(state['context'],
            attn_metadata={name: meta for name in names}, slot_mapping={name: mapping for name in names},
            is_padding=select_v2_padding(state['context'].is_padding, selected_rows),
            batch_descriptor=None)
        state['fixture_input'] = torch.cat([state['inputs'][depth][row:row + 1].clone()
                                          for row, depth in selected])
        state['fixture_reference'] = torch.cat([state['outputs'][depth][row:row + 1].clone()
                                              for row, depth in selected])
        state['fixture_working'] = state['fixture_input'].clone()
        state['fixture_positions'] = positions
        state['pages'] = pages
        state['protected'] = sorted(protected)
        state['address_records'] = address_records
        state['fixture_metadata'] = meta
        torch.cuda.synchronize()
        fixture_padding = state['fixture_context'].is_padding.cpu().tolist()
        if fixture_padding != [state['padding_observation']['flags'][row] for row, _ in selected]:
            raise C07ValidationError('The fixture padding does not select the original native row flags.')
        state['preparation'] = dict(phase='PREPARED', materialize_wall_ns=time.perf_counter_ns() - start,
            logical_payload_bytes=payload_bytes(len(core.layers), cache.shape[1], p, len(selected)),
            pages=pages, protected_pages=sorted(protected), reserved_null_page=0,
            page_size=16, selected=selected, source_request_ids=state['request_ids'],
            schedule_action=schedule_action, source_slots_retained=4 * len(core.layers),
            history_bitwise_checked=True, addresses=address_records,
            selected_versions=[dict(request_id=state['request_ids'][row], captured_row=row,
                logical_depth=depth, prefix_start=0, prefix_end_exclusive=p,
                current_position=p, capture_call=state['capture_call']) for row, depth in selected],
            input_reset_bytes=state['fixture_input'].numel() * state['fixture_input'].element_size(),
            padding_observation=state['padding_observation'], fixture_is_padding=fixture_padding,
            captured_batch=state['captured_batch'],
            lease='Exclusive manual lease after all requests finish. No subsequent serving call is permitted.',
            cost_scope='Completed materialization wall includes remap, address checks, and synchronization. Capture request and export are separate.')
        state['phase'] = 'PREPARED'
        return state['preparation']

    def _c07_one_pass(self, hidden):
        from vllm.forward_context import override_forward_context
        state = self._c07
        core = self.get_model().model
        with override_forward_context(state['fixture_context']):
            residual = None
            for layer in core.layers[core.start_layer:core.end_layer]:
                hidden, residual = layer(state['fixture_positions'], hidden, 0, residual)
            hidden, _ = core.norm(hidden, residual)
        return hidden

    def _c07_compare(self, output):
        import torch
        state = self._c07
        core = self.get_model().model
        checks = []
        actual_current = []
        for description in state['cache_descriptions']:
            cache = core.layers[description['layer']].self_attn.attn[description['depth']].kv_cache
            if (cache.data_ptr() != description['pointer'] or list(cache.shape) != description['shape']
                    or list(cache.stride()) != description['stride']
                    or cache.storage_offset() != description['storage_offset']):
                raise C07ValidationError('The fixture changed an original depth-cache allocation.')

        def compare(name, actual, reference):
            checks.append(compare_bf16_words(name, actual, reference, torch))

        compare('normalized_pass_output', output, state['fixture_reference'])
        p = state['position']
        for target_row, (source_row, source_depth) in enumerate(state['selected']):
            values = []
            block = state['pages'][target_row][p // 16]
            for layer_index, layer in enumerate(core.layers):
                actual = layer.self_attn.attn[0].kv_cache[block, :, p % 16, :].detach().clone()
                values.append(actual)
                compare(f'current_KV.row{target_row}.layer{layer_index}', actual,
                        state['current'][target_row][layer_index])
                token_positions = torch.arange(p, device=actual.device)
                table = state['fixture_metadata'].block_table[target_row]
                prefix = layer.self_attn.attn[0].kv_cache[table[token_positions // 16].long(),
                                                          :, token_positions % 16, :]
                compare(f'history_KV.row{target_row}.layer{layer_index}', prefix,
                        state['prefixes'][target_row][layer_index])
                source_table = state['source_table'][source_row]
                original_prefix = layer.self_attn.attn[source_depth].kv_cache[
                    source_table[token_positions // 16].long(), :, token_positions % 16, :]
                compare(f'original_history_KV.row{target_row}.depth{source_depth}.layer{layer_index}',
                        original_prefix, state['prefixes'][target_row][layer_index])
            actual_current.append(values)
        state['actual_current'] = actual_current
        state['actual_output'] = output.detach().clone()
        return dict(status=require_bitwise(checks), checks=checks, tolerance='BITWISE_BF16_WORDS_ONLY')

    def _c07_all_ranks_pass(self, parity):
        import torch
        import torch.distributed as dist
        from vllm.distributed import get_tp_group, get_tensor_model_parallel_world_size
        flag = torch.tensor([int(parity['status'] == 'BITWISE_PASS')], dtype=torch.int32, device='cpu')
        if get_tensor_model_parallel_world_size() > 1:
            dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=get_tp_group().cpu_group)
        return bool(flag.item())

    def c07_service(self, repeats=5):
        import torch
        from vllm.distributed import get_tp_group, get_tensor_model_parallel_world_size
        state = self._c07
        if state['phase'] != 'PREPARED' or not 1 <= repeats <= 32:
            raise RuntimeError('The eager service requires a prepared one-shot fixture and one through32 repeats.')
        with torch.inference_mode():
            output = self._c07_one_pass(state['fixture_input'].clone())
            torch.cuda.synchronize()
            parity = check_local_validation(lambda: self._c07_compare(output), 'INITIAL')
            state['parity'] = parity
            state['service'] = []
            if not self._c07_all_ranks_pass(parity):
                state['phase'] = 'UNRESOLVED'
                return dict(phase=state['phase'], parity=parity, service=[],
                            reason='Local validation or bitwise parity failed. No service timing is accepted.')
            for ordinal in range(repeats):
                # Fused residual operations can mutate inputs. Charge the reset copy inside service.
                hidden = state['fixture_working']
                first, last = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                first.record(); last.record()
                torch.cuda.synchronize()
                if get_tensor_model_parallel_world_size() > 1:
                    get_tp_group().barrier()
                torch.cuda.synchronize()
                start = time.perf_counter_ns()
                first.record()
                hidden.copy_(state['fixture_input'])
                output = self._c07_one_pass(hidden)
                last.record()
                last.synchronize()
                state['service'].append(dict(ordinal=ordinal, rank_local_cuda_event_ms=first.elapsed_time(last),
                    rank_local_host_wall_ns=time.perf_counter_ns() - start))
                trial_parity = check_local_validation(lambda: self._c07_compare(output), f'TRIAL.{ordinal}')
                state['service'][-1]['parity'] = trial_parity
                if not self._c07_all_ranks_pass(trial_parity):
                    state['phase'] = 'UNRESOLVED'
                    state['final_parity'] = trial_parity
                    return dict(phase=state['phase'], initial_parity=parity, final_parity=trial_parity,
                                service=state['service'], reason='A service trial failed bitwise parity. Timings are unvalidated.')
            torch.cuda.synchronize()
            final = check_local_validation(lambda: self._c07_compare(output), 'FINAL')
            state['final_parity'] = final
            state['phase'] = 'BITWISE_PASS' if self._c07_all_ranks_pass(final) else 'UNRESOLVED'
            return dict(phase=state['phase'], initial_parity=parity, final_parity=final,
                service=state['service'], scope='Eager physical pass only. Not native graph service or a speculation result.',
                input_reset='One working-input copy is inside each service interval.',
                clock_scope='Rank-local events and wall intervals. They do not establish aligned cross-rank makespan.')

    def c07_export(self):
        import torch
        from vllm.distributed import get_tensor_model_parallel_rank
        state = self._c07
        if state.get('hooks'):
            state['capture_hooks'].restore()
            for hook in state['hooks']:
                hook.remove()
            state['hooks'] = []
        rank = get_tensor_model_parallel_rank()
        root = Path(os.environ['C07_FIXTURE_OUTPUT']) / f'rank{rank}'
        root.mkdir(parents=True, exist_ok=False)

        def cpu(value):
            if torch.is_tensor(value):
                return value.detach().cpu()
            if is_dataclass(value) and not isinstance(value, type):
                return {field.name: cpu(getattr(value, field.name)) for field in fields(value)}
            if isinstance(value, dict):
                return {key: cpu(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [cpu(item) for item in value]
            if value is None or isinstance(value, (str, int, float, bool)):
                return value
            return dict(opaque_type=f'{type(value).__module__}.{type(value).__qualname__}')

        keys = ['positions', 'inputs', 'outputs', 'native_metadata', 'source_table', 'prefixes', 'current',
                'fixture_metadata', 'fixture_input', 'fixture_reference', 'actual_output', 'actual_current']
        bundle = {key: cpu(state[key]) for key in keys if key in state}
        if 'context' in state:
            bundle['native_slot_mapping'] = cpu(state['context'].slot_mapping)
            bundle['native_forward_context'] = cpu(state['context'])
            bundle['native_attention_module_identity'] = {
                name: dict(python_id=id(module), type=f'{type(module).__module__}.{type(module).__qualname__}')
                for name, module in state['context'].no_compile_layers.items()}
        path = root / 'fixture.pt'
        start = time.perf_counter_ns()
        torch.save(bundle, path)
        manifest = dict(rank=rank, phase=state['phase'], selected=state['selected'],
            position=state['position'], request_ids=state.get('request_ids'),
            calls_seen=state['calls_seen'], rejected=state['rejected'],
            native_caches=state.get('cache_descriptions'), preparation=state.get('preparation'),
            metadata_ownership=state.get('metadata_ownership'),
            metadata_tensor_ownership=state.get('metadata_tensor_ownership'),
            parity=state.get('parity'), final_parity=state.get('final_parity'), service=state.get('service'),
            fixture_path=str(path), fixture_sha256=sha(path), fixture_file_bytes=path.stat().st_size,
            export_wall_ns=time.perf_counter_ns() - start,
            scope='Captured actual native reference and one remapped eager physical pass. Not a branch engine.')
        write(root / 'manifest.json', manifest)
        return manifest

"""Retain original encoder outputs for stable image identities in stock EPD."""

from collections import OrderedDict
import hashlib
import json
import os
import re
import time

from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorRole

from ..mllm_epd_prefill_heavy_token_split import cuda_ipc_ec as ipc
from ..mllm_epd_prefill_heavy_token_split.worker_observer import CampaignWorkerExtension


ALIAS_NAME = "memory_pool_encoder_hit.json"
HASH = re.compile(r"[0-9a-f]{64}")
ALIAS = re.compile(r"e_[0-9a-f]{64}")


def checked_hash(key):
    if not isinstance(key, str) or not HASH.fullmatch(key):
        raise ValueError("Expected a lowercase SHA-256 encoder identity")
    return key


def source_alias(owner, key):
    value = f"encoder:{checked_hash(owner)}:{checked_hash(key)}".encode()
    return "e_" + hashlib.sha256(value).hexdigest()


def configure_connector(command, role, *, cap_bytes, reuse_encoder, retain_bytes=None, cache_device="gpu"):
    """Configure the campaign command without editing the historical runtime."""
    if role not in ("encoder", "prefill", "decode"):
        raise ValueError(f"Unsupported EPD role {role!r}")
    if type(reuse_encoder) is not bool or type(cap_bytes) is not int or cap_bytes <= 0:
        raise ValueError("Invalid encoder cache role or capacity")
    retained = (cap_bytes if reuse_encoder else 0) if retain_bytes is None else retain_bytes
    if type(retained) is not int or not 0 <= retained <= cap_bytes or (not reuse_encoder and retained):
        raise ValueError("Invalid retained encoder cache capacity")
    if cache_device not in ("gpu", "cpu") or (cache_device == "cpu" and not reuse_encoder):
        raise ValueError("Pinned CPU retention requires encoder cache reuse")

    package = __package__
    index = command.index("--ec-transfer-config") + 1
    config = json.loads(command[index])
    config["ec_connector"] = "StableCUDAIPCEncoderConnector"
    config["ec_connector_module_path"] = package + ".stable_ec"
    config["ec_connector_extra_config"].update(
        max_storage_bytes=cap_bytes, retained_storage_bytes=retained,
        reuse_encoder=reuse_encoder, cache_device=cache_device,
    )
    command[index] = json.dumps(config)
    worker = command.index("--worker-extension-cls") + 1
    command[worker] = package + ".stable_ec.StableECWorkerExtension"
    if role == "encoder":
        entry = command.index("-m") + 1
        command[entry] = package + ".stable_ec_server"
        middleware = command.index("--middleware") + 1
        command[middleware] = package + ".stable_ec_server.StableECMiddleware"


class StableProducerRegistry(ipc.ProducerRegistry):
    """Keep one immutable source per content and one lease per EPD request."""

    def __init__(self, root, max_storage_bytes, retained_storage_bytes, reuse_encoder):
        super().__init__(root, max_storage_bytes)
        if self.crossing_preissue:
            raise ValueError("Stable EC does not use token-split descriptor preissue")
        self.retained_storage_bytes = retained_storage_bytes
        self.reuse_encoder = reuse_encoder
        self.owner_keys = {}
        self.key_owners = {}
        self.alias_keys = {}
        self.idle = OrderedDict()
        self.acquire_hits = 0
        self.acquire_misses = 0
        self.encoder_saves = 0
        self.late_saves = 0
        self.evictions = 0
        self.audit = os.environ.get("MLLM_EPD_STABLE_EC_AUDIT") == "1"

    def acquire(self, owner, keys):
        checked_hash(owner)
        ordered_keys = [checked_hash(key) for key in keys]
        keys = list(dict.fromkeys(ordered_keys))
        if not keys or owner in self.owner_keys:
            raise ValueError("Expected a new request owner with at least one image")
        self.drain_file_acks()
        self.owner_keys[owner] = keys
        hits = []
        try:
            for key in keys:
                self.key_owners.setdefault(key, set()).add(owner)
                self.idle.pop(key, None)
                alias = source_alias(owner, key)
                self.alias_keys[alias] = (key, owner)
                if self.reuse_encoder and self._acquire_source(key):
                    path = ipc.directory(self.root, alias)
                    ipc.write(path / ALIAS_NAME, {"key": key, "owner": owner})
                    hits.append(key)
        except BaseException:
            self.release_owner(owner)
            raise
        self.acquire_hits += len(hits)
        self.acquire_misses += len(keys) - len(hits)
        if self.audit:
            ipc.append(ipc.ROOT / "raw" / "stable_ec_encoder_audit.jsonl", {
                "kind": "acquire", "owner": owner, "ordered_image_keys": ordered_keys,
                "resident_hits": hits, "reuse_encoder": self.reuse_encoder,
            })
        return {"owner": owner, "hits": hits, "images": len(keys)}

    def _has_exports(self, key):
        return any(key in outstanding["keys"] for outstanding in self.exports.values())

    def _unowned(self, key):
        if key in self.held and key not in self.key_owners and not self._has_exports(key):
            self.idle.setdefault(key, None)

    def _acquire_source(self, key):
        return key in self.held

    def _discard(self, key, *, eviction=True):
        if key in self.key_owners or self._has_exports(key):
            raise RuntimeError("Cannot evict a leased or exported encoder allocation")
        folder = ipc.directory(self.root, key)
        (folder / "cuda_ipc_ready.json").unlink()
        folder.rmdir()
        self.held.pop(key)
        self.ready.pop(key)
        self.idle.pop(key, None)
        self.released_count += 1
        self.evictions += eviction

    def _extra_bytes(self, tensor):
        storage = tensor.untyped_storage()
        pointer = storage.data_ptr()
        return 0 if any(item.untyped_storage().data_ptr() == pointer for item in self.held.values()) else storage.nbytes()

    def _idle_bytes(self):
        active = {tensor.untyped_storage().data_ptr()
                  for key, tensor in self.held.items() if key not in self.idle}
        resident = {}
        for key in self.idle:
            storage = self.held[key].untyped_storage()
            if storage.data_ptr() not in active:
                resident[storage.data_ptr()] = storage.nbytes()
        return sum(resident.values())

    def _trim_retained(self):
        while self.idle and (self.retained_storage_bytes == 0 or
                             self._idle_bytes() > self.retained_storage_bytes):
            self._discard(next(iter(self.idle)))

    def _ensure_gpu_room(self, additional_bytes):
        while self.idle and self.storage_bytes() + additional_bytes > self.max_storage_bytes:
            self._discard(next(iter(self.idle)))
        if self.storage_bytes() + additional_bytes > self.max_storage_bytes:
            raise MemoryError("Held request encoder output exceeds the configured E tier bound")

    def _publish(self, key, tensor):
        self._ensure_gpu_room(self._extra_bytes(tensor))
        super().save(key, tensor)

    def save(self, alias, tensor):
        if not isinstance(alias, str) or not ALIAS.fullmatch(alias):
            raise ValueError("An encoder output must use its request transport alias")
        binding = self.alias_keys.get(alias)
        if binding is None:
            # Encoder work already issued can finish after its caller aborts.
            self.late_saves += 1
            return
        key, owner = binding
        self.encoder_saves += 1
        if self.audit:
            import torch

            array = tensor.detach().contiguous().view(torch.uint8).cpu().numpy()
            ipc.append(ipc.ROOT / "raw" / "stable_ec_encoder_audit.jsonl", {
                "kind": "encoder_compute", "owner": owner, "alias": alias, "key": key,
                "sha256": hashlib.sha256(memoryview(array)).hexdigest(), "shape": list(tensor.shape),
                "dtype": str(tensor.dtype), "logical_bytes": array.nbytes,
                "published_as_source": key not in self.held,
            })
        if key in self.held:
            if tensor.shape != self.held[key].shape or tensor.dtype != self.held[key].dtype:
                raise ValueError("Stable identity collided across encoder output geometries")
            return
        if self.inline_role is not None:
            raise RuntimeError("Stable EC needs on-demand descriptors scoped by live leases")
        self.drain_file_acks()
        self._publish(key, tensor)

    def export(self, keys, consumer):
        if any(key not in self.key_owners for key in keys):
            raise RuntimeError("Export requires a live request lease")
        return super().export(keys, consumer)

    def acknowledge(self, token):
        keys = self.exports[token]["keys"]
        result = super().acknowledge(token)
        for key in keys:
            self._unowned(key)
        self._trim_retained()
        return result

    def release(self, keys):
        raise RuntimeError("Stable identities must be released by request owner")

    def release_owner(self, owner):
        checked_hash(owner)
        self.drain_file_acks()
        for key in self.owner_keys.pop(owner, []):
            alias = source_alias(owner, key)
            folder = ipc.directory(self.root, alias)
            marker = folder / ALIAS_NAME
            if marker.exists():
                marker.unlink()
                folder.rmdir()
            self.alias_keys.pop(alias, None)
            owners = self.key_owners.get(key)
            if owners is not None:
                owners.discard(owner)
                if not owners:
                    del self.key_owners[key]
            self._unowned(key)
        self._trim_retained()
        return self.state()

    def state(self):
        result = super().state()
        return dict(result, active_request_leases=len(self.owner_keys),
                    reuse_encoder=self.reuse_encoder, idle_images=len(self.idle),
                    idle_storage_bytes=self._idle_bytes(),
                    retained_storage_bytes=self.retained_storage_bytes,
                    retained_cache_device="gpu",
                    diagnostic_tensor_hashes=self.audit,
                    acquired_image_hits=self.acquire_hits, acquired_image_misses=self.acquire_misses,
                    computed_image_saves=self.encoder_saves, late_image_saves=self.late_saves,
                    cache_evictions=self.evictions)


class HostRetainedProducerRegistry(StableProducerRegistry):
    """Use pinned CPU storage between requests with the same E→P IPC path."""

    def __init__(self, root, max_storage_bytes, retained_storage_bytes, reuse_encoder):
        super().__init__(root, max_storage_bytes, retained_storage_bytes, reuse_encoder)
        self.host = OrderedDict()
        self.host_bytes = 0
        self.host_peak_bytes = 0
        self.host_loads = self.host_stores = 0
        self.host_load_bytes = self.host_store_bytes = 0
        self.host_load_seconds = self.host_store_seconds = 0.0
        self.host_evictions = 0
        self.host_store_batches = 0
        self._host_store_stream = None
        self._pending_host = None

    def _allocate_host(self, key):
        import torch

        tensor = self.held[key]
        return torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu", pin_memory=True)

    def _copy_to_host(self, pending):
        import torch

        if self._host_store_stream is None:
            self._host_store_stream = torch.cuda.Stream()
        stream = self._host_store_stream
        try:
            with torch.cuda.stream(stream):
                for key, host in pending.items():
                    stream.wait_event(self.ready[key])
                    host.copy_(self.held[key], non_blocking=True)
        finally:
            # Even a failed enqueue may leave earlier copies in flight. Keep
            # both endpoints owned until the dedicated stream's fence succeeds.
            stream.synchronize()
            self._pending_host = None

    def _to_gpu(self, host):
        import torch

        return host.to(device=torch.device("cuda", torch.cuda.current_device()))

    def _pop_host(self, key):
        tensor = self.host.pop(key)
        self.host_bytes -= tensor.untyped_storage().nbytes()

    def _discard(self, key, *, eviction=True):
        if self._pending_host is not None and key in self._pending_host:
            raise RuntimeError("Cannot discard a GPU source before its host copy is fenced")
        super()._discard(key, eviction=eviction)

    def _acquire_source(self, key):
        if key in self.held:
            if key in self.host:
                self.host.move_to_end(key)
            return True
        host = self.host.get(key)
        if host is None:
            return False
        size = host.untyped_storage().nbytes()
        self._ensure_gpu_room(size)
        begin = time.monotonic()
        tensor = self._to_gpu(host)
        self.host_load_seconds += time.monotonic() - begin
        self._publish(key, tensor)
        self.host.move_to_end(key)
        self.host_loads += 1
        self.host_load_bytes += size
        return True

    def _trim_retained(self):
        if not self.idle:
            return
        if self._pending_host is not None:
            raise RuntimeError("Unfenced pinned-host copies still own GPU sources")
        sizes = OrderedDict((key, tensor.untyped_storage().nbytes()) for key, tensor in self.host.items())
        planned_bytes = self.host_bytes
        for key in self.idle:
            if key in sizes:
                sizes.move_to_end(key)
                continue
            if key in self.host:
                size = self.host[key].untyped_storage().nbytes()
            else:
                tensor = self.held[key]
                size = tensor.numel() * tensor.element_size()
            if size > self.retained_storage_bytes:
                continue
            sizes[key] = size
            planned_bytes += size
            while planned_bytes > self.retained_storage_bytes:
                _, evicted_bytes = sizes.popitem(last=False)
                planned_bytes -= evicted_bytes

        evicted = self.host.keys() - sizes.keys()
        for key in evicted:
            self._pop_host(key)
        self.host_evictions += len(evicted)
        self.evictions += len(evicted)

        new_keys = [key for key in sizes if key not in self.host]
        if new_keys:
            begin = time.monotonic()
            pending = {key: self._allocate_host(key) for key in new_keys}
            if any(tensor.untyped_storage().nbytes() != sizes[key] for key, tensor in pending.items()):
                raise RuntimeError("Unexpected physical size for pinned embedding storage")
            self._pending_host = pending
            self.host_peak_bytes = max(self.host_peak_bytes, planned_bytes)
            self._copy_to_host(pending)
            self.host_store_seconds += time.monotonic() - begin
            self.host.update(pending)
            self.host_stores += len(pending)
            self.host_store_bytes += sum(sizes[key] for key in pending)
            self.host_store_batches += 1
        self.host = OrderedDict((key, self.host[key]) for key in sizes)
        self.host_bytes = planned_bytes
        for key in list(self.idle):
            self._discard(key, eviction=key not in self.host and key not in evicted)

    def state(self):
        result = super().state()
        pending_bytes = sum(tensor.untyped_storage().nbytes() for tensor in (self._pending_host or {}).values())
        return dict(result, retained_cache_device="cpu", host_retained_images=len(self.host),
                    host_retained_storage_bytes=self.host_bytes, host_peak_storage_bytes=self.host_peak_bytes,
                    host_loads=self.host_loads, host_stores=self.host_stores, host_evictions=self.host_evictions,
                    host_load_bytes=self.host_load_bytes, host_store_bytes=self.host_store_bytes,
                    host_store_batches=self.host_store_batches, host_pending_storage_bytes=pending_bytes,
                    host_load_seconds=self.host_load_seconds, host_store_seconds=self.host_store_seconds)


class StableCUDAIPCEncoderConnector(ipc.CUDAIPCEncoderConnector):
    def __init__(self, vllm_config, role):
        super().__init__(vllm_config, role)
        if self.slice_aware:
            raise ValueError("Stable EC only supports whole-image native prompt handoff")
        if self.is_producer and role == ECConnectorRole.WORKER:
            extra = vllm_config.ec_transfer_config.ec_connector_extra_config
            registry_type = HostRetainedProducerRegistry if extra.get("cache_device") == "cpu" else StableProducerRegistry
            ipc.REGISTRY = registry_type(self._storage_path, extra["max_storage_bytes"],
                                         extra["retained_storage_bytes"], extra["reuse_encoder"])

    def has_cache_item(self, identifier):
        if not self.is_consumer:
            if not isinstance(identifier, str) or not ALIAS.fullmatch(identifier):
                raise ValueError("Encoder stage requires a request transport alias")
            # The encoder-only execute_model branch never gathers LM inputs.
            # Reporting a resident hit suppresses its Vision forward without
            # changing the consumer's exact stable embedding identity.
            return (ipc.directory(self._storage_path, identifier) / ALIAS_NAME).is_file()
        return super().has_cache_item(checked_hash(identifier))


class StableECWorkerExtension(CampaignWorkerExtension):
    def campaign_stable_ec_acquire(self, owner, keys):
        return ipc.REGISTRY.acquire(owner, keys)

    def campaign_stable_ec_release(self, owner):
        return ipc.REGISTRY.release_owner(owner)

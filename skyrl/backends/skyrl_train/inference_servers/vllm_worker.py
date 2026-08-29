"""
vLLM Worker Extension for SkyRL weight synchronization.

This module provides WorkerWrap, a vLLM worker extension class that enables
efficient NCCL-based and CUDA IPC-based weight updates from the training
process to inference workers.

TODO: This will be removed once vLLM natively supports weight sync APIs.
See: https://github.com/vllm-project/vllm/issues/31848

Usage:
    Pass as --worker-extension-cls to vLLM:

    vllm serve ... --worker-extension-cls skyrl_train.inference_servers.vllm_worker.WorkerWrap
"""

import warnings

import torch

from skyrl.backends.skyrl_train.inference_servers.layerwise_reload import (
    LayerwiseReloadWorkerMixin,
)

# Path to this worker extension class for use in CLI args (derived from module path)
VLLM_WORKER_EXTENSION_CLS = f"{__name__}.WorkerWrap"


def _ix_slice_to_local_shard(src, dest):
    """SkyRL-IsoExec mismatched-engine-TP receiver reshard. The trainer gathers each TP-sharded weight
    to the ENGINE-TOTAL tensor and sends that; this engine worker holds only a 1/engine_tp shard. We
    slice ``src`` (full) to THIS worker's own Megatron TP-rank shard -- using our OWN mpu rank, so the
    result is correct regardless of how the engines are placed across GPUs (fixes the engine-TP>1
    gibberish that came from the sender guessing our shard->GPU map). Returns None when it is not a
    clean single-dim TP shard (then the caller leaves ``src`` unchanged)."""
    try:
        from megatron.core import parallel_state as mpu

        from skyrl.backends.skyrl_train.isoexec.sync.native_weight_sync import (
            _split_full_to_shard,
        )

        dim = None
        etp = None
        for d in range(src.dim()):
            if src.shape[d] != dest.shape[d]:
                if dest.shape[d] == 0 or src.shape[d] % dest.shape[d] != 0 or dim is not None:
                    return None  # zero / non-integer ratio, or >1 differing dim -> not a TP shard
                dim, etp = d, src.shape[d] // dest.shape[d]
        if dim is None or etp is None or etp < 2:
            return None
        stride = int(getattr(dest, "partition_stride", 1))
        # expert params shard over the expert-tensor-parallel group; at EP=1 that == the TP group,
        # so the plain TP rank is correct for the (EP=1, ETP=TP) configs the 35B path uses.
        r = (mpu.get_tensor_model_parallel_rank() if mpu.model_parallel_is_initialized() else 0) % etp
        return _split_full_to_shard(src, dim, stride, etp, r)
    except Exception:
        return None


class WorkerWrap(LayerwiseReloadWorkerMixin):
    """
    vLLM worker extension for SkyRL weight synchronization.

    This class is injected into vLLM workers via --worker-extension-cls and
    provides methods that can be called via engine.collective_rpc() to
    coordinate weight updates across all TP/PP workers.

    Methods:
        init_weight_update_communicator: Initialize the weight receiver
        skyrl_start_weight_update: Begin a sync; initialize vLLM layerwise reload once
        load_weights: Receive and load one chunk of weights from trainer
        skyrl_finish_weight_update: End a sync; finalize vLLM layerwise reload once
        teardown_weight_receiver: Clean up weight receiver resources
    """

    def test_rpc(self, *args, **kwargs):
        """Test RPC call to worker."""
        return args, kwargs

    def init_weight_update_communicator(self, init_info: bytes):
        """
        Initialize weight update communicator from init info.

        Args:
            init_info: Pickled bytes of WeightSyncInitInfo from the sender.
        """
        import pickle

        assert torch.distributed.is_initialized(), "default torch process group must be initialized"

        # Unpickle init_info to restore the original object type
        assert isinstance(init_info, bytes), f"Expected bytes, got {type(init_info).__name__}"
        init_info = pickle.loads(init_info)

        strategy_cls = init_info.strategy_type()

        if hasattr(self, "_weight_receiver") and self._weight_receiver is not None:
            # TODO(haochen): we should get rid of this flag and override existing receiver.
            if init_info.override_existing_receiver:
                self._weight_receiver.teardown()
                self._weight_receiver = None
            else:
                warnings.warn(
                    "Detected an existing weight receiver. "
                    "For overriding, use `generator.inference_engine.override_existing_update_group=enable`"
                )
                return

        self._weight_receiver = strategy_cls.create_receiver(init_info)

    def load_weights(self, request: bytes) -> None:
        """
        Load one chunk of weights using the receiver.

        Called via collective_rpc from the weight loader, once per chunk.
        When the sender brackets the sync with skyrl_start_weight_update / skyrl_finish_weight_update,
        the chunk is loaded raw and the single finalize runs vLLM's post-load weight
        processing exactly once over the whole weight set.

        Args:
            request: Pickled bytes of WeightUpdateRequest.
        """
        import pickle

        from vllm.config import set_current_vllm_config

        # Unpickle request to restore the original object type
        assert isinstance(request, bytes), f"Expected bytes, got {type(request).__name__}"
        request = pickle.loads(request)

        weight_list = []
        for name, tensor in self._weight_receiver.receive_weights(request):
            weight_list.append((name, tensor))

        import os as _os

        if _os.environ.get("SKYRL_ISOEXEC") == "1":
            model = self.model_runner.model
            target = model.gpt if hasattr(model, "gpt") else model
            # DST MAP: never cached ACROSS syncs -- vLLM's colocate sleep/wake (cumem) re-allocates
            # the weight storage on each wake, and a stale dict would copy into freed tensors while
            # generation reads the live (zero) buffers -> gibberish. But it is safe, and worth a lot,
            # to cache it WITHIN one sync: `load_weights` runs once per CHUNK (hundreds of calls on a
            # 35B, each rebuilding a 20,943-entry dict by walking the whole module tree), and the
            # sender's start/finish bracket delimits exactly one sync with no wake inside it
            # (worker_dispatch.save_weights_for_sampler does every wake_up BEFORE the broadcast).
            # `skyrl_start_weight_update` clears the cache and `skyrl_finish_weight_update` drops it.
            _dstmap = getattr(self, "_ix_dstmap", None)
            if _dstmap is None:
                _dstmap = (dict(target.named_parameters()), dict(target.named_buffers()))
                self._ix_dstmap = _dstmap
            params, bufs = _dstmap
            copied = 0
            materialized = 0
            miss = []
            self._ix_miss_names = getattr(self, "_ix_miss_names", [])
            self._ix_miss_ck = getattr(self, "_ix_miss_ck", None)

            def _set_on_module(root, dotted, value, as_param, requires_grad):
                # navigate to the owning submodule and replace the param/buffer object so a META
                # placeholder (from cumem sleep freeing storage) becomes a real GPU tensor.
                *path, attr = dotted.split(".")
                mod = root
                for p in path:
                    mod = getattr(mod, p)
                if as_param:
                    mod._parameters[attr] = torch.nn.Parameter(value, requires_grad=requires_grad)
                else:
                    mod._buffers[attr] = value

            with torch.no_grad(), set_current_vllm_config(self.vllm_config):
                for name, tensor in weight_list:
                    is_param = name in params
                    dest = params.get(name)
                    if dest is None:
                        dest = bufs.get(name)
                    if dest is None:
                        if len(miss) < 3:
                            miss.append(name)
                        if name not in self._ix_miss_names:
                            self._ix_miss_names.append(name)
                        continue
                    tgt_dtype = dest.dtype if dest.dtype.is_floating_point else tensor.dtype
                    src = tensor.to(self.device, tgt_dtype)
                    # RECEIVER-SIDE RESHARD (mismatched engine TP, e.g. the 35B trainer TP=8 ->
                    # engine TP=4): the trainer gathered to full and sent the ENGINE-TOTAL weight; this
                    # worker holds only its TP shard. Slice `src` to OUR OWN mpu tp-rank -- so we never
                    # depend on the sender guessing our shard->GPU placement (the engine-TP>1 gibberish
                    # bug). No-op when shapes already match (matched TP / engine TP=1).
                    # Slice on shape mismatch REGARDLESS of meta: a meta dest still knows its shard
                    # shape, and skipping the slice would install the engine-TOTAL tensor where a
                    # 1/engine_tp shard belongs (post-sleep re-materialization under mismatched TP).
                    if tuple(dest.shape) != tuple(src.shape):
                        _sl = _ix_slice_to_local_shard(src, dest)
                        if _sl is not None:
                            src = _sl
                    if not hasattr(self, "_ix_synced_meta"):
                        self._ix_synced_meta = {}
                    self._ix_synced_meta[name] = (tuple(src.shape), src.dtype)
                    if dest.is_meta or dest.device.type == "meta" or tuple(dest.shape) != tuple(src.shape):
                        # cumem freed the storage -> param is META; replace the object entirely.
                        # .clone(): src is a VIEW of the sender's IPC-mapped packed chunk -- installing
                        # it directly pins that whole chunk buffer in the SENDER for the param's
                        # lifetime. The engine must own its parameter memory.
                        _set_on_module(
                            target, name, src.clone().contiguous(), is_param, getattr(dest, "requires_grad", False)
                        )
                        materialized += 1
                    else:
                        dest.copy_(src)
                    copied += 1
                    if True:
                        _sig_names = getattr(self, "_ix_sig_names", None)
                        if _sig_names is None:
                            _sig_names = self._ix_sig_names = []
                        _sig_floor = getattr(self, "_ix_sig_floor", 1 << 20)
                        if name in _sig_names or (len(_sig_names) < 8 and src.numel() > _sig_floor):
                            if name not in _sig_names:
                                _sig_names.append(name)
                            if not hasattr(self, "_ix_sig"):
                                self._ix_sig = {}
                            self._ix_sig[name] = float(src.detach().float().abs().sum())
            _ix_did_cuda_sync = False
            _ix_did_ipc_collect = False
            # Drain the async pinned D2H cache copies BEFORE this RPC returns: `src` views the
            # sender's IPC-mapped packed chunk, and the sender ipc_collect/reuses that buffer as
            # soon as the apply call completes. One sync for the whole chunk, not one per tensor.
            torch.cuda.synchronize()
            _ix_did_cuda_sync = True
            self._isoexec_copied = getattr(self, "_isoexec_copied", 0) + copied
            # A name miss means the sender exported a param the engine gpt has no slot for -> the
            # sync is silently incomplete. Always surface it (cheap: name list only); it is a real
            # correctness signal, not a diagnostic.
            if self._ix_miss_names:
                print(
                    f"[ISOEXEC-MISS] {len(self._ix_miss_names)} sender names NOT in engine gpt; "
                    f"names={self._ix_miss_names}",
                    flush=True,
                )
            torch.accelerator.synchronize()  # consume IPC tensors before sender drops them
            for weight in weight_list:
                del weight
            weight_list.clear()
            # Release this chunk's cached IPC mapping NOW. The receiver-side allocator caches
            # opened IPC handles; until collected, the SENDER cannot reclaim its packed chunk
            # buffer, so sender memory grows by the full send volume across a sync (observed
            # +8.7GiB/3000 params at 35B EP8 -> OOM at ~62GiB mid-extraction).
            torch.cuda.ipc_collect()
            _ix_did_ipc_collect = True
            # LIFECYCLE ASSERT (drain-before-IPC-return): both drains must have run before this RPC
            # returns and the sender reclaims its IPC-mapped chunk. Observation-only, fail-soft.
            from skyrl.backends.skyrl_train.isoexec.lifecycle import (
                ordering as _ix_order,
            )

            _ix_order.check_ipc_drained_before_return(_ix_did_cuda_sync, _ix_did_ipc_collect)
            return

        weight_update_bracketed = getattr(self, "_skyrl_weight_update_active", False)
        with torch.device(self.device), set_current_vllm_config(self.vllm_config):
            if weight_update_bracketed:
                self.model_runner.model.load_weights(weights=weight_list)
            else:
                self.model_runner.reload_weights(weights_iterator=iter(weight_list))

        if weight_update_bracketed:
            # Finish consuming IPC-backed tensors before the sender drops them on
            # its next barrier; matches NewInferenceWorkerWrap.update_weights_ipc
            torch.accelerator.synchronize()

        for weight in weight_list:
            del weight

    def isoexec_release_cached_blocks(self) -> float:
        """Hand the caching allocator's free-but-reserved segments back to the driver at sleep.

        The training window starts right after ``llm.sleep``, and the trainer measures its
        non-activation floor DEVICE-wide ((total-free)-reserved),
        so anything this worker merely CACHES is charged to the trainer's backward budget
        (~0.4 GiB measured: reserved 43.316 vs alloc 42.899 GiB). Returns the GiB released."""
        import torch

        before = torch.cuda.memory_reserved()
        torch.cuda.empty_cache()
        return (before - torch.cuda.memory_reserved()) / (1024.0**3)

    def isoexec_reapply_cached_weights(self) -> int:
        """Re-copy the last synced weights (cached on CPU at sync time) into the live model.

        On the nightly stack ANY wake_up issued AFTER the native sync clobbers the just-synced
        weights (sleep level 1 -> restores the never-updated step-0 CPU backup; level 2 -> zero
        pages), because the wake path is not reliably scoped to the requested tags. The dispatch
        therefore calls this AFTER the final wake_up(tags=["kv_cache"]), so generation always
        runs on the trainer's bytes regardless of what the wake machinery did to the tensors.
        """
        import torch

        # Rowinv ENGAGEMENT boundary (engine side). This seam runs ONCE PER SYNC in every engine
        # worker process -- the only per-sync engine hook, never per-forward -- and by the first
        # post-generation sync this worker has computed sampled logprobs, so a contract that
        # selects rowinv_leaftree with a census that never served it -- engine served=0 while the
        # trainer served every row, behind a MATCHING contract hash -- refuses here instead of
        # running silently. No-op when the flag is off; the init
        # sync (pre-generation) is granted inside the boundary. Deliberate refusals propagate
        # through collective_rpc; everything else is fail-safe inside the call.
        try:
            from skyrl.backends.skyrl_train.isoexec.core.enforce import (
                rowinv_engagement_boundary,
            )
        except ImportError:  # pragma: no cover - legacy non-IsoExec compatibility
            pass
        else:
            rowinv_engagement_boundary("engine")

        model = self.model_runner.model
        target = model.gpt if hasattr(model, "gpt") else model

        # ---- post-sync drift check (see the fingerprint block in load_weights) ----------------
        _sig = getattr(self, "_ix_sig", None)
        if _sig:
            _params = dict(target.named_parameters())
            _bufs = dict(target.named_buffers())
            _bad = []
            with torch.no_grad():
                for _n, _want in _sig.items():
                    _t = _params.get(_n, _bufs.get(_n))
                    if _t is None or _t.device.type == "meta":
                        _bad.append((_n, _want, None))
                        continue
                    _got = float(_t.detach().float().abs().sum())
                    if _got != _want:
                        _bad.append((_n, _want, _got))
            if _bad:
                msg = (
                    f"[ISOEXEC-REAPPLY-DRIFT] {len(_bad)}/{len(_sig)} synced-weight fingerprints moved "
                    f"between the apply and the post-wake seam; first={_bad[:3]}."
                )
                print(msg, flush=True)
                raise RuntimeError(msg)
            else:
                print(f"[ISOEXEC-REAPPLY-DRIFT] clean ({len(_sig)}/{len(_sig)} fingerprints held)", flush=True)
        else:
            # NO WITNESS THIS SYNC. Previously this branch did not exist: `if _sig:` simply fell
            # through and the whole drift check became a silent no-op, so a model with no param
            # above the 1 MiB selection floor (every small test model, and the GPU-CI shapes that
            # exercise this path) reported "(1,0) stale" as success. Drop the floor for good so the
            # NEXT sync is checked with whatever params exist, and say so out loud.
            _floor = getattr(self, "_ix_sig_floor", 1 << 20)
            if _floor > 0:
                self._ix_sig_floor = 0
                print(
                    "[ISOEXEC-REAPPLY-DRIFT] NO WITNESS: no synced param exceeded the "
                    f"{_floor}-element selection floor, so this sync was NOT drift-checked. "
                    "Floor dropped to 0; the next sync selects the first 8 params of any size.",
                    flush=True,
                )
            elif getattr(self, "_isoexec_copied", 0):
                print(
                    "[ISOEXEC-REAPPLY-DRIFT] NO WITNESS at floor 0 despite a non-empty sync -- the "
                    "fingerprint selector never ran. The engine is UNCHECKED this sync.",
                    flush=True,
                )

        from skyrl.backends.skyrl_train.isoexec.ops.moe.moe_fused_weights import (
            bump_sync_epoch as _bump,
        )

        _bump()
        return 0

    def teardown_weight_receiver(self):
        """Clean up weight receiver resources."""
        if not hasattr(self, "_weight_receiver") or self._weight_receiver is None:
            warnings.warn("No weight receiver to teardown")
            return
        self._weight_receiver.teardown()

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 zdfb
"""YOCO-style consumer-tail prefill truncation for DeepSeek-V4.1.

Every layer after the last KV source (layer 20) is a pure *consumer*: it owns
only a 128-token sliding-window cache and reads its long-context KV from the
source layer.  Two consequences matter for prefill:

* A consumer's output at position ``t`` is only ever read for positions inside
  a sliding window that ends at ``t``; nothing outside the last
  ``window + margin`` tokens can reach a later logit.
* Consumers never write long-context KV, so skipping positions cannot corrupt
  any cache that a later chunk reads.

So a prefill chunk can run the consumer tail on just the last ``KEEP`` tokens
of every request.  ``KEEP`` must cover the sliding window (128) plus the
speculative backtracking margin, hence the default of 140 used by DeepInfer.

The tail runs with a *trimmed* view of the attention metadata: the query rows
become each request's suffix, while ``seq_lens`` / block tables stay untouched
so visibility is unchanged.  This is exactly a normal "last chunk" prefill from
the kernels' point of view.
"""

from __future__ import annotations

import copy
import itertools
import os
from dataclasses import dataclass

import torch

# Sliding window (128) + DSpark backtracking margin.
DEFAULT_KEEP_TOKENS = 140


def configured_keep_tokens() -> int:
    raw = os.environ.get("V41_YOCO_KEEP", "0")
    try:
        return int(raw)
    except ValueError:
        return 0


@dataclass
class TailPlan:
    idx: torch.Tensor          # [sum(kept)] long, indices into the full chunk
    qsl: torch.Tensor          # [num_reqs + 1] int32, cumulative kept lengths
    qsl_cpu: torch.Tensor      # [num_reqs + 1] cpu
    total: int
    max_kept: int


class YocoTail:
    """Per-model state for consumer-tail truncation."""

    def __init__(self, *, start_layer: int, keep: int, num_layers: int, model_config, parallel_config, topology):
        self.start_layer = start_layer
        self.keep = keep
        text = model_config.hf_text_config
        tp = parallel_config.tensor_parallel_size
        self.cfg = dict(
            n_local_heads=int(getattr(text, "num_attention_heads")) // tp,
            head_dim=int(getattr(text, "head_dim")),
            index_n_heads=int(getattr(text, "index_n_heads")),
            index_head_dim=int(getattr(text, "index_head_dim")),
            index_topk=int(getattr(text, "index_topk")),
            window=int(getattr(text, "sliding_window")),
        )
        if self.keep < self.cfg["window"]:
            raise ValueError(
                f"V41_YOCO_KEEP={keep} is smaller than the sliding window "
                f"({self.cfg['window']}); truncated tokens would still be visible"
            )
        # Every consumer owns its own sliding-window cache, so contamination in
        # the retained prefix propagates one window per layer.  Keeping the tail
        # exact for the request's last token therefore needs
        #   window * (number of consumer layers + 1)
        # tokens.  DeepInfer's 140 assumes a single shared consumer cache, which
        # is not the case here; the small value means "approximate" and is only
        # used when explicitly requested.
        self.tail_layers = num_layers - start_layer
        self.exact_keep = self.cfg["window"] * (self.tail_layers + 1)
        if self.keep < self.exact_keep:
            if os.environ.get("V41_YOCO_APPROX", "0") == "1":
                print(
                    f"[YOCO-TAIL] APPROX mode: keep={self.keep} < exact={self.exact_keep}; "
                    "outputs are not bit-equivalent to stock",
                    flush=True,
                )
            else:
                print(
                    f"[YOCO-TAIL] keep={self.keep} raised to {self.exact_keep} "
                    f"({self.tail_layers} consumer layers x window {self.cfg['window']}) "
                    "to stay exact; set V41_YOCO_APPROX=1 for the faster approximation",
                    flush=True,
                )
                self.keep = self.exact_keep
        self.candidate_topk_blocks = int(topology.candidate_topk_blocks)
        self.stats = dict(chunks=0, rows_skipped=0)

    # -- planning ---------------------------------------------------------
    def plan(self, metadata) -> TailPlan | None:
        swa = next((b for b in metadata.values() if b.cache_kind == "swa"), None)
        if swa is None or swa.query_start_loc_cpu is None:
            return None
        num_reqs = int(swa.num_reqs)
        qsl_cpu = swa.query_start_loc_cpu
        starts = [int(qsl_cpu[i]) for i in range(num_reqs + 1)]
        kept = [min(starts[i + 1] - starts[i], self.keep) for i in range(num_reqs)]
        total = sum(kept)
        if total <= 0 or total >= starts[-1]:
            return None
        parts = [
            torch.arange(starts[i + 1] - kept[i], starts[i + 1], dtype=torch.long)
            for i in range(num_reqs)
            if kept[i] > 0
        ]
        idx = torch.cat(parts).to(swa.slot_mapping.device)
        qsl = torch.tensor([0] + list(itertools.accumulate(kept)), dtype=torch.int32, device=idx.device)
        qsl_cpu = torch.tensor([0] + list(itertools.accumulate(kept)), dtype=torch.int32)
        return TailPlan(idx=idx, qsl=qsl, qsl_cpu=qsl_cpu, total=total, max_kept=max(kept))

    # -- metadata ---------------------------------------------------------
    def trim(self, metadata, plan: TailPlan) -> dict:
        """Return a metadata dict whose query rows are the request suffixes.

        Only cache planes the consumer tail actually reads are trimmed; ratio-2
        compressor planes stay as they are (they belong to layers <= 20).
        """
        out = dict(metadata)
        smla_done: dict = {}
        qli_done: dict = {}
        for prefix, bundle in metadata.items():
            if getattr(bundle, "cache_kind", None) not in ("swa", "long_kv", "index_k"):
                continue
            if bundle.compress_ratio not in (0, 1):
                continue
            nb = copy.copy(bundle)
            if bundle.slot_mapping is not None:
                nb.slot_mapping = bundle.slot_mapping[plan.idx]
            if bundle.positions is not None:
                nb.positions = bundle.positions[plan.idx]
            # cos/sin are RopeDataProxy objects: a tensor index gathers every
            # (config, group) plane inside them and returns a new proxy.
            if bundle.cos is not None:
                nb.cos = bundle.cos[plan.idx]
            if bundle.sin is not None:
                nb.sin = bundle.sin[plan.idx]
            nb.query_start_loc = plan.qsl
            nb.query_start_loc_cpu = plan.qsl_cpu
            nb.num_actual_tokens = plan.total
            nb.num_input_tokens = plan.total
            nb.max_query_len = plan.max_kept
            if bundle.smla_metadata is not None:
                key = (bundle.cache_kind, bundle.compress_ratio)
                if key not in smla_done:
                    smla_done[key] = self._rebuild_smla(bundle, plan)
                nb.smla_metadata = smla_done[key]
            if getattr(bundle, "qli_metadata", None) is not None:
                key = bundle.compress_ratio
                if key not in qli_done:
                    qli_done[key] = self._rebuild_qli(bundle, plan)
                nb.qli_metadata = qli_done[key]
            out[prefix] = nb
        return out

    def _rebuild_smla(self, bundle, plan: TailPlan) -> torch.Tensor:
        """Rebuild SMLA launch metadata for the trimmed query layout.

        The tensor object is reused (and overwritten in place) so that the
        device-metadata executor's already-registered event ids stay valid;
        every untrimmed user of this plane has finished by the time the tail
        starts.
        """
        ratio = 0 if bundle.cache_kind == "swa" else int(bundle.compress_ratio)
        has_cmp = ratio in (1, 2)
        num_reqs = int(bundle.num_reqs)
        max_ori = int(bundle.max_cache_seq_len) * (int(bundle.compress_ratio) if bundle.cache_kind != "swa" else 1)
        value = torch.ops._C_ascend.npu_sparse_flash_mla_metadata(
            self.cfg["n_local_heads"],
            1,
            self.cfg["head_dim"],
            cu_seqlens_q=plan.qsl,
            seqused_ori_kv=bundle.seq_lens[:num_reqs],
            seqused_cmp_kv=(bundle.cache_seq_lens[:num_reqs] if has_cmp else None),
            cmp_residual_kv=(bundle.cmp_residual if has_cmp else None),
            batch_size=num_reqs,
            max_seqlen_q=plan.max_kept,
            max_seqlen_ori_kv=max_ori,
            max_seqlen_cmp_kv=(int(bundle.max_cache_seq_len) if has_cmp else 0),
            ori_topk=0,
            cmp_topk=(self.cfg["index_topk"] if has_cmp else 0),
            cmp_ratio=ratio,
            ori_mask_mode=4,
            cmp_mask_mode=(3 if has_cmp else 0),
            ori_win_left=max(0, self.cfg["window"] - 1),
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_BBND",
            has_ori_kv=True,
            has_cmp_kv=has_cmp,
        )
        bundle.smla_metadata.copy_(value)
        return bundle.smla_metadata

    def _rebuild_qli(self, bundle, plan: TailPlan) -> torch.Tensor:
        num_reqs = int(bundle.num_reqs)
        value = torch.ops._C_ascend.npu_quant_lightning_indexer_v2_metadata(
            self.cfg["index_n_heads"],
            1,
            self.cfg["index_head_dim"],
            self.cfg["index_topk"],
            2,
            cu_seqlens_q=plan.qsl,
            seqused_k=bundle.cache_seq_lens[:num_reqs],
            cmp_residual_k=bundle.cmp_residual,
            batch_size=num_reqs,
            max_seqlen_q=plan.max_kept,
            max_seqlen_k=int(bundle.max_cache_seq_len),
            layout_q="TND",
            layout_k="PA_BBND",
            mask_mode=3,
            cmp_ratio=int(bundle.compress_ratio),
        )
        bundle.qli_metadata.copy_(value)
        return bundle.qli_metadata

    # -- token tensors ----------------------------------------------------
    def gather_shared(self, shared, plan: TailPlan) -> None:
        """Move the shared TopK/candidate rows of the kept tokens to the front.

        Layer 20 already filled these buffers for the whole chunk; the tail
        reads them as ``[:total]``, so the kept rows must be compacted first.
        """
        for buf in (shared.topk_indices, shared.candidates):
            if buf is None:
                continue
            tmp = buf.index_select(0, plan.idx)
            buf[: plan.total].copy_(tmp)
        self.stats["chunks"] += 1

    def scatter_full(self, trimmed: torch.Tensor, plan: TailPlan, rows: int) -> torch.Tensor:
        """Place the trimmed tail result back into a full-chunk tensor.

        Only the kept rows carry values; the runner samples (and DSpark reads)
        the request tails, which are always kept.
        """
        out = torch.zeros((rows,) + tuple(trimmed.shape[1:]), dtype=trimmed.dtype, device=trimmed.device)
        out[plan.idx] = trimmed
        self.stats["rows_skipped"] += rows - plan.total
        return out

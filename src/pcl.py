# Copyright (C) 2026 Efficient Computing Lab - NTUA <vpsomak@mail.ntua.gr>
#
# This file is part of InferLens.
#
# InferLens is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# InferLens is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with InferLens. If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import json
import math
import os
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Rank-1 Associative Memory Expert
# ─────────────────────────────────────────────────────────────────────────────

class Rank1Expert(nn.Module):
    """
    One atomic rank-1 LoRA expert — one key-value associative memory unit.

    For weight matrix W ∈ R^{out × in}:
        ΔW  =  value ⊗ key      (rank-1 outer product)

    Applied to input x:
        Δy  =  (keyᵀ · x) · value · scale

    'key'   encodes what input direction this expert responds to.
    'value' encodes what it contributes when activated.
    No external router: activation is purely content-addressed via dot product.
    """

    def __init__(self, in_features: int, out_features: int, scale: float = 0.1):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.scale        = scale

        # key  — small random init  (analogous to LoRA-A row)
        self.key   = nn.Parameter(torch.empty(in_features))
        nn.init.normal_(self.key, mean=0.0, std=1.0 / math.sqrt(in_features))

        # value — zero init so expert is neutral at spawn  (analogous to LoRA-B col = 0)
        self.value = nn.Parameter(torch.zeros(out_features))

    def activation_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Cosine similarity between input x and this expert's key.
        x: (..., in_features)  →  returns (...,)
        """
        k_n = F.normalize(self.key.unsqueeze(0), dim=-1)    # (1, in)
        x_n = F.normalize(x, dim=-1)                         # (..., in)
        return (x_n @ k_n.T).squeeze(-1)                     # (...,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., in_features)  →  (..., out_features)"""
        score = x @ self.key                                  # (...,)
        return score.unsqueeze(-1) * self.value * self.scale  # (..., out)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Associative Memory Bank
# ─────────────────────────────────────────────────────────────────────────────

class AssociativeMemoryBank(nn.Module):
    """
    Replaces a target linear layer (e.g. q_proj, v_proj) with a content-
    addressable mixture of rank-1 experts layered on top of the frozen weights.

    Forward:
        output = W_frozen(x)  +  Σ_i softmax(scores)_i · expert_i(x)

    Experts accumulate incrementally; old experts are frozen in place.
    Gradient flows only through the top-k most activated experts,
    preventing inactive experts from receiving noisy signals.
    """

    def __init__(
        self,
        frozen_linear:   nn.Linear,
        lora_scale:      float = 0.1,
        top_k_trainable: int   = 1,
    ):
        super().__init__()
        self.frozen_linear   = frozen_linear
        self.in_features     = frozen_linear.in_features
        self.out_features    = frozen_linear.out_features
        self.lora_scale      = lora_scale
        self.top_k_trainable = top_k_trainable

        for p in self.frozen_linear.parameters():
            p.requires_grad = False

        self.frozen_experts: nn.ModuleList        = nn.ModuleList()
        self.active_expert:  Optional[Rank1Expert] = None

        # Populated on every forward pass — no grad, used for routing diagnostics
        self._last_activation_profile: Dict[int, float] = {}

    # ── Pool management ────────────────────────────────────────────────

    @property
    def all_experts(self) -> List[Rank1Expert]:
        experts = list(self.frozen_experts)
        if self.active_expert is not None:
            experts.append(self.active_expert)
        return experts

    def expert_count(self) -> int:
        return len(self.frozen_experts) + (1 if self.active_expert else 0)

    def add_new_expert(self):
        """Freeze any active expert, spawn a fresh trainable one."""

        if self.active_expert is not None:
            self._freeze_active()
        ref_param = next(self.frozen_linear.parameters())
        device    = ref_param.device
        dtype     = ref_param.dtype
        self.active_expert = (
            Rank1Expert(self.in_features, self.out_features, scale=self.lora_scale)
            .to(device=device, dtype=dtype)
        )

    def commit(self):
        """Explicitly move the active expert to the frozen pool."""

        if self.active_expert is not None:
            self._freeze_active()

    def _freeze_active(self):
        for p in self.active_expert.parameters():
            p.requires_grad = False
        self.frozen_experts.append(self.active_expert)
        self.active_expert = None

    # ── Forward ────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.frozen_linear(x)

        experts = self.all_experts
        if not experts:
            return base_out

        scores  = torch.stack([e.activation_score(x) for e in experts], dim=0)
        weights = F.softmax(scores, dim=0)

        # Record mean activation per expert
        with torch.no_grad():
            reduce_dims = tuple(range(1, weights.dim()))
            mean_w      = weights.mean(dim=reduce_dims) if reduce_dims else weights
            self._last_activation_profile = {
                i: float(mean_w[i].item()) for i in range(len(experts))
            }

        # Selective gradient flow — detach experts below top-k activation
        reduce_dims = tuple(range(1, weights.dim()))
        mean_act    = weights.mean(dim=reduce_dims) if reduce_dims else weights
        topk_n      = min(self.top_k_trainable, len(experts))
        topk_ids    = set(mean_act.topk(topk_n).indices.tolist())

        expert_out = torch.zeros_like(base_out)
        for i, expert in enumerate(experts):
            w      = weights[i].unsqueeze(-1)
            contrib = expert(x)
            if i not in topk_ids:
                contrib = contrib.detach()
            expert_out = expert_out + w * contrib

        return base_out + expert_out


# ─────────────────────────────────────────────────────────────────────────────
# 3.  PCLStats
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PCLStats:
    """
    Returned by PCLAdapter.observe() every turn.
    Call .to_dict() to embed in the "prefill" JSON event under key "pcl".
    """

    # Learning decision
    trained:             bool  = False
    skipped:             bool  = False
    skip_reason:         str   = ""

    # Signal values
    lm_loss:             Optional[float] = None  # NLL loss on this turn
    entropy:             Optional[float] = None  # mean token entropy
    loss_ema:            Optional[float] = None  # drift EMA value
    loss_ema_peak:       Optional[float] = None  # EMA historical minimum

    # Expert state
    drift_detected:      bool            = False
    experts_total:       int             = 0     # per bank
    experts_spawned:     int             = 0     # cumulative
    expert_updated_idx:  Optional[int]   = None  # which expert was refined

    # Activation profile averaged across banks: {"expert_0": 0.4, ...}
    activation_profile:  Dict[str, float] = field(default_factory=dict)

    # Timing
    observe_ms:          float = 0.0

    # Session counters
    total_turns:         int   = 0
    total_skipped:       int   = 0
    total_trained:       int   = 0
    replay_buffer_size:  int   = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  PCLAdapter
# ─────────────────────────────────────────────────────────────────────────────

class PCLAdapter:
    """
    Parameters
    ----------
    model             : the model to be used
    tokenizer         : matching tokenizer
    device            : "cuda" | "cpu"
    target_modules    : which linear projection names to wrap
                        default: ("q_proj", "v_proj")
    lora_scale        : scaling factor for rank-1 expert contributions
    lr                : learning rate for active expert optimizer
    loss_ema_alpha    : EMA smoothing coefficient for drift detection  (0 < α < 1)
    drift_threshold   : fractional EMA loss rise triggering expert spawn
    entropy_threshold : minimum avg token entropy to justify a gradient step
    replay_size       : max size of the tiny experience replay buffer
    replay_ratio      : probability of mixing a replay sample into a step
    accum_steps       : gradient accumulation steps before optimizer.step()
    checkpoint_dir    : if set, expert weights are saved here on commit
    """

    def __init__(
        self,
        model:             nn.Module,
        tokenizer:         Any,
        device:            str             = "cuda",
        target_modules:    Tuple[str, ...] = ("q_proj", "v_proj"),
        lora_scale:        float           = 0.1,
        lr:                float           = 2e-4,
        loss_ema_alpha:    float           = 0.1,
        drift_threshold:   float           = 0.20,
        entropy_threshold: float           = 0.5,
        replay_size:       int             = 50,
        replay_ratio:      float           = 0.3,
        accum_steps:       int             = 4,
        checkpoint_dir:    Optional[str]   = None,
    ):
        self.model              = model
        self.tokenizer          = tokenizer
        self.device             = device
        self.target_modules     = set(target_modules)
        self.lora_scale         = lora_scale
        self.lr                 = lr
        self._ema_alpha         = loss_ema_alpha
        self.drift_threshold    = drift_threshold
        self.entropy_threshold  = entropy_threshold
        self.replay_size        = replay_size
        self.replay_ratio       = replay_ratio
        self.accum_steps        = accum_steps
        self.checkpoint_dir     = checkpoint_dir

        # Drift EMA state
        self._loss_ema:      Optional[float] = None
        self._peak_loss_ema: Optional[float] = None

        # Optimizer + accumulation
        self._optimizer:   Optional[torch.optim.AdamW] = None
        self._accum_count: int = 0

        # Replay buffer
        self._replay_buffer: deque = deque(maxlen=replay_size)

        # Counters
        self._total_turns:     int = 0
        self._total_skipped:   int = 0
        self._total_trained:   int = 0
        self._experts_spawned: int = 0

        # Temporary-unfreeze tracking
        self._temp_unfrozen_idx: Optional[int] = None

        # Inject banks into model
        self._banks: Dict[str, AssociativeMemoryBank] = {}
        self._inject_banks()

    # ── Bank injection ────────────────────────────────────────────────

    def _inject_banks(self):
        """Replace every target nn.Linear in the model with a memory bank."""
        pending: List[Tuple[Any, str, AssociativeMemoryBank]] = []

        for name, module in self.model.named_modules():
            attr = name.split(".")[-1]
            if (
                attr in self.target_modules
                and isinstance(module, nn.Linear)
                and not isinstance(module, AssociativeMemoryBank)
            ):
                bank   = AssociativeMemoryBank(module, lora_scale=self.lora_scale)
                parent = self._get_parent(name)
                pending.append((parent, attr, bank))
                self._banks[name] = bank

        for parent, attr, bank in pending:
            setattr(parent, attr, bank)

        print(
            f"[PCLAdapter] Injected {len(self._banks)} memory bank(s) "
            f"on modules: {self.target_modules}"
        )

    def _get_parent(self, dotted_name: str) -> nn.Module:
        parts = dotted_name.split(".")[:-1]
        mod   = self.model
        for p in parts:
            mod = getattr(mod, p)
        return mod

    # ── Dtype helpers ─────────────────────────────────────────────────

    def _model_dtype(self) -> torch.dtype:
        # ── Case 1: BitsAndBytes quantised model ─────────────────────
        cfg = getattr(self.model, "config", None)
        bnb_cfg = getattr(cfg, "quantization_config", None)
        if bnb_cfg is not None:
            # transformers >= 4.30 stores this as a QuantizationConfig
            compute = getattr(bnb_cfg, "bnb_4bit_compute_dtype", None)
            if compute is not None and isinstance(compute, torch.dtype):
                return compute
            # Might be stored as a string ("bfloat16", "float16")
            if isinstance(compute, str):
                _map = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                        "float32": torch.float32}
                if compute in _map:
                    return _map[compute]
            # Fallback for 4-bit: most users run bf16 or fp16 compute
            return (
                torch.bfloat16
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                else torch.float16
            )

        # ── Case 2: walk parameters looking for a floating-point one ─
        _float_dtypes = {torch.float16, torch.bfloat16, torch.float32, torch.float64}
        for bank in self._banks.values():
            for p in bank.frozen_linear.parameters():
                if p.dtype in _float_dtypes:
                    return p.dtype

        # Also check lm_head or embed_tokens as fallback
        for name in ("lm_head", "model"):
            sub = getattr(self.model, name, None)
            if sub is None:
                continue
            for p in sub.parameters():
                if p.dtype in _float_dtypes:
                    return p.dtype

        # ── Case 3: safe default ─────────────────────────────────────
        return torch.float32

    def _cast_expert(self, expert: "Rank1Expert") -> "Rank1Expert":
        """Cast an expert's parameters to the model dtype in-place."""
        dt = self._model_dtype()
        with torch.no_grad():
            expert.key.data   = expert.key.data.to(dtype=dt)
            expert.value.data = expert.value.data.to(dtype=dt)
        return expert

    # ── Expert lifecycle ──────────────────────────────────────────────

    def _expert_count(self) -> int:
        for bank in self._banks.values():
            return bank.expert_count()
        return 0

    def _trainable_params(self) -> List[nn.Parameter]:
        return [
            p for bank in self._banks.values()
            for mod in ([bank.active_expert] + list(bank.frozen_experts))
            if mod is not None
            for p in mod.parameters()
            if p.requires_grad
        ]

    def _add_experts(self):
        """Freeze current active expert; spawn a fresh trainable one."""
        for bank in self._banks.values():
            bank.add_new_expert()
        self._reset_optimizer()
        self._experts_spawned += 1
        print(
            f"[PCLAdapter] Expert #{self._experts_spawned} spawned. "
            f"Total per bank: {self._expert_count()}"
        )

    def _commit_experts(self):
        """Freeze all active experts and optionally save to disk."""
        for bank in self._banks.values():
            bank.commit()
        if self.checkpoint_dir:
            self._save_experts()

    def _reset_optimizer(self):
        """
        Reset Adam state completely.
        Old gradient moments must not push a new expert into the same
        subspace as its predecessor — that would defeat the purpose of
        spawning a fresh one.
        """
        params = self._trainable_params()
        if params:
            self._optimizer = torch.optim.AdamW(
                params, lr=self.lr, weight_decay=0.01
            )
        self._accum_count = 0

    # ── Temporary unfreezing  ─────────────────────────────────────

    def _temporarily_unfreeze(self, expert_idx: int):
        """
        Make a frozen expert temporarily trainable for one gradient step.
        """

        self._temp_unfrozen_idx = expert_idx
        params: List[nn.Parameter] = []
        for bank in self._banks.values():
            if expert_idx < len(bank.frozen_experts):
                exp = bank.frozen_experts[expert_idx]
                # Ensure dtype matches model before gradients flow
                self._cast_expert(exp)
                for p in exp.parameters():
                    p.requires_grad = True
                params.extend(exp.parameters())
        if params:
            self._optimizer = torch.optim.AdamW(
                params, lr=self.lr * 0.1, weight_decay=0.01
            )

    def _refreeze_temporary(self):
        """Re-freeze any temporarily unfrozen expert."""

        if self._temp_unfrozen_idx is None:
            return
        for bank in self._banks.values():
            idx = self._temp_unfrozen_idx
            if idx < len(bank.frozen_experts):
                for p in bank.frozen_experts[idx].parameters():
                    p.requires_grad = False
        self._temp_unfrozen_idx = None

    # ── Signal computation ────────────────────────────────────────────

    @torch.no_grad()
    def _compute_entropy(self, prompt: str) -> float:
        """
        Average Shannon entropy of the model's next-token distribution
        across all positions in the prompt.

        High entropy → model is uncertain → gradient step is informative.
        Low entropy  → model is confident → gradient step is wasteful.
        """

        enc = self.tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=512
        )
        input_ids = enc["input_ids"].to(self.device)
        attn_mask = enc.get("attention_mask")
        if attn_mask is not None:
            attn_mask = attn_mask.to(self.device)

        self.model.eval()
        out   = self.model(input_ids=input_ids, attention_mask=attn_mask)
        probs = F.softmax(out.logits, dim=-1)                 # (1, seq, vocab)
        h     = -(probs * (probs + 1e-9).log()).sum(dim=-1)   # (1, seq)
        return float(h.mean().item())

    def _compute_lm_loss(self, prompt: str, response: str) -> torch.Tensor:
        """
        Self-supervised next-token prediction loss.
        """

        # ── Tokenise ──────────────────────────────────────────────────
        p_ids = self.tokenizer(
            prompt, add_special_tokens=True
        )["input_ids"]
        r_ids = self.tokenizer(
            response.strip(), add_special_tokens=False
        )["input_ids"]

        # Guard: no response tokens to predict
        if len(r_ids) == 0:
            raise ValueError(
                "Response tokenised to zero tokens — cannot compute LM loss. "
                f"response={repr(response[:80])}"
            )

        input_ids = p_ids + r_ids
        labels    = [-100] * len(p_ids) + list(r_ids)

        max_len   = getattr(self.model.config, "max_position_embeddings", 2048)
        input_ids = input_ids[-max_len:]
        labels    = labels[-max_len:]

        # Guard: truncation may have removed all response tokens
        if all(lbl == -100 for lbl in labels):
            raise ValueError(
                "All label tokens are masked after truncation — "
                "response is shorter than the prompt tail."
            )

        t = torch.tensor([input_ids], device=self.device)
        l = torch.tensor([labels],    device=self.device)

        loss = self.model(input_ids=t, labels=l).loss

        # Guard: nan / inf (can happen with quantized models on edge cases)
        if not torch.isfinite(loss):
            raise ValueError(
                f"Model returned non-finite loss: {loss.item():.6g}. "
                "Possibly a quantization edge case — skipping this turn."
            )

        return loss

    def _update_drift(self, loss_val: float) -> bool:
        """
        EMA-based distribution drift detection.
        """

        import math
        if not math.isfinite(loss_val):
            return False   # ignore bad values
        if self._loss_ema is None:
            self._loss_ema      = loss_val
            self._peak_loss_ema = loss_val
            return False

        self._loss_ema = (
            self._ema_alpha * loss_val
            + (1.0 - self._ema_alpha) * self._loss_ema
        )

        if self._loss_ema < self._peak_loss_ema:
            self._peak_loss_ema = self._loss_ema
            return False

        rise = (
            (self._loss_ema - self._peak_loss_ema)
            / (self._peak_loss_ema + 1e-8)
        )
        if rise > self.drift_threshold:
            print(
                f"[PCLAdapter] Drift — EMA rose {rise:.1%} "
                f"({self._peak_loss_ema:.4f} → {self._loss_ema:.4f})"
            )
            self._peak_loss_ema = self._loss_ema
            return True
        return False

    # ── Diagnostics ───────────────────────────────────────────────────

    def _find_dominant_expert(self, prompt: str) -> Optional[int]:
        """
        Identify which existing expert activates most on this prompt by
        running a no-grad forward pass and reading bank activation profiles.
        """
        if self._expert_count() == 0:
            return None

        enc = self.tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=256
        )
        input_ids = enc["input_ids"].to(self.device)

        self.model.eval()
        with torch.no_grad():
            self.model(input_ids=input_ids)

        totals: Dict[int, float] = {}
        for bank in self._banks.values():
            for idx, strength in bank._last_activation_profile.items():
                totals[idx] = totals.get(idx, 0.0) + strength

        return max(totals, key=lambda k: totals[k]) if totals else None

    def _aggregate_activation_profile(self) -> Dict[str, float]:
        """Activation averaged across all banks for the last forward pass."""
        totals:  Dict[int, float] = {}
        n_banks: int = 0
        for bank in self._banks.values():
            for idx, s in bank._last_activation_profile.items():
                totals[idx] = totals.get(idx, 0.0) + s
            n_banks += 1
        if not n_banks:
            return {}
        return {
            f"expert_{k}": round(v / n_banks, 6)
            for k, v in sorted(totals.items())
        }

    # ── Persistence ───────────────────────────────────────────────────

    def _save_experts(self):
        if not self.checkpoint_dir:
            return
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        state: Dict[str, Any] = {
            bank_name: [
                {
                    "key":   exp.key.detach().cpu().tolist(),
                    "value": exp.value.detach().cpu().tolist(),
                    "scale": exp.scale,
                }
                for exp in bank.frozen_experts
            ]
            for bank_name, bank in self._banks.items()
        }
        path = os.path.join(self.checkpoint_dir, "pcl_experts.json")
        with open(path, "w") as f:
            json.dump(state, f)
        print(f"[PCLAdapter] Experts saved to {path}")

    def load_experts(self, checkpoint_dir: Optional[str] = None):
        """Restore expert weights from a previous session's checkpoint."""
        ckpt = checkpoint_dir or self.checkpoint_dir or "."
        path = os.path.join(ckpt, "pcl_experts.json")
        if not os.path.exists(path):
            print(f"[PCLAdapter] No checkpoint at {path} — starting fresh.")
            return
        with open(path) as f:
            state = json.load(f)
        for bank_name, experts_data in state.items():
            if bank_name not in self._banks:
                continue
            bank  = self._banks[bank_name]
            dtype = self._model_dtype()
            for e in experts_data:
                exp = Rank1Expert(
                    bank.in_features, bank.out_features, scale=e["scale"]
                )
                exp.key.data   = torch.tensor(e["key"],   device=self.device, dtype=dtype)
                exp.value.data = torch.tensor(e["value"], device=self.device, dtype=dtype)
                for p in exp.parameters():
                    p.requires_grad = False
                bank.frozen_experts.append(exp)
        print(f"[PCLAdapter] Loaded experts from {path}")

    @property
    def summary(self) -> Dict[str, Any]:
        """Quick status snapshot."""
        return {
            "experts_total":   self._expert_count(),
            "experts_spawned": self._experts_spawned,
            "total_turns":     self._total_turns,
            "total_trained":   self._total_trained,
            "total_skipped":   self._total_skipped,
            "loss_ema":        round(self._loss_ema or 0.0, 6),
            "loss_ema_peak":   round(self._peak_loss_ema or 0.0, 6),
            "banks":           len(self._banks),
            "replay_buffer":   len(self._replay_buffer),
        }

    # ─────────────────────────────────────────────────────────────────
    # Main public method
    # ─────────────────────────────────────────────────────────────────

    def observe(self, prompt: str, response: str) -> PCLStats:
        """
        Parameters
        ----------
        prompt   : the full prompt string sent to the model this turn
        response : the complete text the model generated

        Returns
        -------
        PCLStats — embed .to_dict() in the prefill event JSON under "pcl"
        """
        print(f"[PCLAdapter] Observing: {prompt}")
        t0 = time.perf_counter()
        self._total_turns += 1

        stats = PCLStats(
            total_turns        = self._total_turns,
            total_skipped      = self._total_skipped,
            total_trained      = self._total_trained,
            replay_buffer_size = len(self._replay_buffer),
        )

        # ── Step 1: LM loss (no grad) ─────────────────────────────────
        # Always computed even if we skip training, so drift tracking stays
        # current even during confident turns.
        try:
            self.model.eval()
            with torch.no_grad():
                loss_val = float(self._compute_lm_loss(prompt, response).item())
        except Exception as exc:
            print(f"[PCLAdapter] Loss failed: {exc}")
            stats.skipped         = True
            stats.skip_reason     = f"loss_error: {exc}"
            self._total_skipped  += 1
            stats.total_skipped   = self._total_skipped
            stats.observe_ms      = round((time.perf_counter() - t0) * 1000.0, 3)
            return stats

        stats.lm_loss = round(loss_val, 6)

        # ── Step 2: Drift detection ───────────────────────────────────
        drift = self._update_drift(loss_val)
        stats.drift_detected = drift
        stats.loss_ema       = round(self._loss_ema or 0.0, 6)
        stats.loss_ema_peak  = round(self._peak_loss_ema or 0.0, 6)
        print(f"[PCLAdapter] Drift measured: {stats.loss_ema}")

        if drift or self._expert_count() == 0:
            self._commit_experts()
            self._add_experts()

        stats.experts_total   = self._expert_count()
        stats.experts_spawned = self._experts_spawned

        # ── Step 3: Entropy gate ──────────────────────────────────────
        try:
            entropy = self._compute_entropy(prompt)
            print(f"[PCLAdapter] Entropy measured: {entropy}")
        except Exception as exc:
            entropy = self.entropy_threshold + 0.01   # fail-open
            print(f"[PCLAdapter] Entropy failed (fail-open): {exc}")

        stats.entropy = round(entropy, 6)

        if entropy < self.entropy_threshold and not drift:
            # Model already handles this type of input confidently — skip
            self._total_skipped += 1
            stats.skipped       = True
            stats.skip_reason   = (
                f"low_entropy ({entropy:.4f} < {self.entropy_threshold})"
            )
            stats.total_skipped      = self._total_skipped
            stats.activation_profile = self._aggregate_activation_profile()
            stats.observe_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            self._replay_buffer.append((prompt, response))
            return stats

        # ── Step 4: Activation-guided expert selection ────────────────
        dominant_idx = self._find_dominant_expert(prompt)
        stats.expert_updated_idx = dominant_idx

        if dominant_idx is not None:
            n_frozen = len(
                list(self._banks.values())[0].frozen_experts
            ) if self._banks else 0
            if dominant_idx < n_frozen:
                # A frozen expert is most relevant — refine it carefully
                self._temporarily_unfreeze(dominant_idx)

        # ── Step 5: Gradient step ─────────────────────────────────────
        try:
            self.model.train()        # allow grad through active / unfrozen experts

            loss   = self._compute_lm_loss(prompt, response)
            (loss / self.accum_steps).backward()
            self._accum_count += 1

            # Optional replay to stabilise older experts
            if (
                self._replay_buffer
                and self._accum_count < self.accum_steps
                and torch.rand(1).item() < self.replay_ratio
            ):
                rp, rr = list(self._replay_buffer)[
                    torch.randint(len(self._replay_buffer), (1,)).item()
                ]
                rl = self._compute_lm_loss(rp, rr)
                (rl / self.accum_steps).backward()
                self._accum_count += 1

            if self._accum_count >= self.accum_steps:
                trainable = self._trainable_params()
                if trainable and self._optimizer is not None:
                    torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                    self._optimizer.step()
                    self._optimizer.zero_grad()
                self._accum_count = 0
                self._refreeze_temporary()

            self._total_trained += 1
            stats.trained        = True

        except Exception as exc:
            print(f"[PCLAdapter] Gradient step failed: {exc}")
            stats.skipped         = True
            stats.skip_reason     = f"grad_error: {exc}"
            self._total_skipped  += 1

        finally:
            # ── ALWAYS restore eval mode ───────────────────────────────
            self.model.eval()
            self._refreeze_temporary()    # idempotent safety call

        # ── Step 6: Bookkeeping ───────────────────────────────────────
        self._replay_buffer.append((prompt, response))

        stats.total_trained        = self._total_trained
        stats.total_skipped        = self._total_skipped
        stats.replay_buffer_size   = len(self._replay_buffer)
        stats.activation_profile   = self._aggregate_activation_profile()
        stats.observe_ms           = round(
            (time.perf_counter() - t0) * 1000.0, 3
        )
        print()
        print(stats)
        print()
        return stats
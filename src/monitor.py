# monitor.py
from __future__ import annotations

import csv
import os
import torch
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict

import psutil

# Optional GPU support via NVML
try:
    import pynvml  # pip install nvidia-ml-py
    _NVML_AVAILABLE = True
except Exception:
    pynvml = None
    _NVML_AVAILABLE = False

# Optional torch CUDA memory stats
try:
    import torch
    _TORCH_AVAILABLE = True
except Exception:
    torch = None
    _TORCH_AVAILABLE = False


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class InferenceOptions:
    temperature: float
    top_p: float
    max_tokens: int
    stream: bool


class ModelTracer:
    def __init__(self, topk: int = 12, gpu_index: int = 0, capture_resource_metrics: bool = True):
        self.topk = topk
        self.step = -1
        self.frames = []
        self._hooks = []
        self.gpu_index = gpu_index
        self.capture_resource_metrics = capture_resource_metrics

        self._layer_start_times = {}
        self._timing_enabled = False
        self._timing_phase = "decode"

    def next_step(self):
        self.step += 1
        frame = {"step": self.step, "layers": {}}

        if self.capture_resource_metrics:
            try:
                frame["resource_metrics"] = MetricsLogger.gather_resource_metrics(
                    gpu_index=self.gpu_index
                )
            except Exception:
                frame["resource_metrics"] = {}

        self.frames.append(frame)

    def _sync_cuda(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def enable_timing(self, enabled: bool = True, phase: str = "prefill"):
        self._timing_enabled = enabled
        self._timing_phase = phase

    def _layer(self, i: int):
        if not self.frames:
            self.next_step()
        L = self.frames[-1]["layers"]
        L.setdefault(i, {})
        return L[i]

    def _topk_abs_lasttok(self, t: torch.Tensor):
        v = t[:, -1, :][0]
        vals, idx = torch.topk(v.abs(), k=min(self.topk, v.shape[-1]))
        return {
            "topk_idx": idx.detach().cpu().tolist(),
            "topk_abs": vals.detach().cpu().tolist(),
            "l2": float(v.norm(p=2).detach().cpu()),
        }

    def hook_proj(self, layer_idx: int, name: str):
        def _hook(module, inp, out):
            rec = self._layer(layer_idx)
            rec[name] = self._topk_abs_lasttok(out)
        return _hook

    def hook_norm_stats(self, layer_idx: int, name: str):
        def _hook(module, inp, out):
            x = out[:, -1, :][0]
            rec = self._layer(layer_idx)
            rec[name] = {
                "l2": float(x.norm(p=2).detach().cpu()),
                "mean": float(x.mean().detach().cpu()),
                "std": float(x.std(unbiased=False).detach().cpu()),
            }
        return _hook

    def hook_layer_pre(self, layer_idx: int):
        def _hook(module, inputs):
            if not self._timing_enabled:
                return
            self._sync_cuda()
            self._layer_start_times[layer_idx] = time.perf_counter()
        return _hook

    def hook_layer_post(self, layer_idx: int):
        def _hook(module, inputs, output):
            if not self._timing_enabled:
                return

            self._sync_cuda()
            t0 = self._layer_start_times.get(layer_idx)
            if t0 is None:
                return

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            rec = self._layer(layer_idx)
            rec[f"{self._timing_phase}_time_ms"] = round(elapsed_ms, 3)
        return _hook

    def _resolve_attr_path(self, root, path: str):
        cur = root
        for part in path.split("."):
            if not hasattr(cur, part):
                return None
            cur = getattr(cur, part)
        return cur

    def _try_register(self, root, path: str, hook_fn):
        mod = self._resolve_attr_path(root, path)
        if mod is not None and hasattr(mod, "register_forward_hook"):
            self._hooks.append(mod.register_forward_hook(hook_fn))
            return True
        return False

    def _try_register_first(self, root, paths, hook_fn):
        for path in paths:
            if self._try_register(root, path, hook_fn):
                return path
        return None

    def _get_backbone(self, model):
        for attr in ("model", "transformer", "base_model", "backbone"):
            if hasattr(model, attr):
                candidate = getattr(model, attr)
                if candidate is not None:
                    return candidate
        return model

    def _get_layers(self, backbone):
        for attr in ("layers", "h", "blocks"):
            if hasattr(backbone, attr):
                layers = getattr(backbone, attr)
                if layers is not None:
                    return layers
        raise RuntimeError("Could not find transformer layers on model backbone")

    def attach(self, model):
        backbone = self._get_backbone(model)
        layers = self._get_layers(backbone)

        for i, block in enumerate(layers):
            self._hooks.append(block.register_forward_pre_hook(self.hook_layer_pre(i)))
            self._hooks.append(block.register_forward_hook(self.hook_layer_post(i)))

            # MLP projections
            self._try_register_first(
                block,
                [
                    "mlp.gate_proj",      # LLaMA/Mistral style
                    "mlp.gate_up_proj",   # fused gate+up projection
                    "mlp.fc1",            # Phi/GPT-like first FFN projection
                ],
                self.hook_proj(i, "mlp_in_proj"),
            )

            self._try_register_first(
                block,
                [
                    "mlp.up_proj",        # LLaMA style
                    "mlp.fc2",            # Phi/GPT-like second FFN projection
                    "mlp.down_proj",      # sometimes useful fallback
                ],
                self.hook_proj(i, "mlp_out_proj"),
            )

            # Attention output projection
            self._try_register_first(
                block,
                [
                    "self_attn.o_proj",
                    "self_attn.dense",
                    "attn.out_proj",
                    "attn.c_proj",
                ],
                self.hook_norm_stats(i, "attn_o_proj"),
            )

            # Input norm
            self._try_register_first(
                block,
                [
                    "input_layernorm",
                    "ln_1",
                    "self_attn_layer_norm",
                ],
                self.hook_norm_stats(i, "in_norm"),
            )

            # Post-attention norm
            self._try_register_first(
                block,
                [
                    "post_attention_layernorm",
                    "ln_2",
                    "final_layernorm",
                ],
                self.hook_norm_stats(i, "post_attn_norm"),
            )

        return self

    def detach(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


class MetricsLogger:
    """
    Appends per-request metrics to a CSV file.

    Requirements:
      - psutil (CPU/RAM)
      - optionally nvidia-ml-py (pynvml) for GPU util + VRAM
      - optionally torch for CUDA allocator stats
    """

    HEADER = [
        # request metadata
        "timestamp",
        "endpoint",
        "origin_ip",
        "model_id",
        "temperature",
        "top_p",
        "max_tokens",
        "stream",
        "status",          # "ok" / "error"
        "latency_ms",
        "thinking_ms",

        # CPU / RAM
        "cpu_percent",
        "ram_percent",
        "ram_used_mb",
        "ram_available_mb",

        # GPU (NVML)
        "gpu_name",
        "gpu_util_percent",
        "gpu_mem_used_mb",
        "gpu_mem_total_mb",

        # Torch allocator (if available)
        "torch_cuda_mem_allocated_mb",
        "torch_cuda_mem_reserved_mb",

        # Generation metrics
        "prompt_tokens",
        "response_tokens",
        "history_tokens",
        "current_message_tokens_raw",
        "current_message_tokens_templated",
    ]

    _class_nvml_inited = False

    def __init__(self, csv_path: str = "metrics.csv", gpu_index: int = 0, float_precision: int = 2):
        self.csv_path = csv_path
        self.gpu_index = gpu_index
        self.float_precision = int(float_precision)
        self._nvml_inited = False
        self._ensure_csv()

    def _ensure_csv(self) -> None:
        os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=self.HEADER,
                    delimiter=";",
                    quotechar='"',
                    quoting=csv.QUOTE_NONNUMERIC,
                )
                writer.writeheader()

    def _init_nvml(self) -> None:
        if not _NVML_AVAILABLE or self._nvml_inited:
            return
        try:
            pynvml.nvmlInit()
            self._nvml_inited = True
        except Exception:
            self._nvml_inited = False

    @classmethod
    def _init_nvml_class(cls) -> None:
        if not _NVML_AVAILABLE or cls._class_nvml_inited:
            return
        try:
            pynvml.nvmlInit()
            cls._class_nvml_inited = True
        except Exception:
            cls._class_nvml_inited = False

    def _read_gpu_metrics(self) -> Dict[str, Any]:
        """
        Returns best-effort GPU metrics (single GPU by gpu_index).
        """
        out: Dict[str, Any] = {
            "gpu_name": "",
            "gpu_util_percent": "",
            "gpu_mem_used_mb": "",
            "gpu_mem_total_mb": "",
        }

        if not _NVML_AVAILABLE:
            return out

        self._init_nvml()
        if not self._nvml_inited:
            return out

        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            name = pynvml.nvmlDeviceGetName(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)

            out["gpu_name"] = name.decode("utf-8") if isinstance(name, (bytes, bytearray)) else str(name)
            out["gpu_util_percent"] = int(util.gpu)
            out["gpu_mem_used_mb"] = round(mem.used / (1024 * 1024), 2)
            out["gpu_mem_total_mb"] = round(mem.total / (1024 * 1024), 2)
        except Exception:
            # leave blanks on failure
            pass

        return out

    def _read_torch_cuda_metrics(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "torch_cuda_mem_allocated_mb": "",
            "torch_cuda_mem_reserved_mb": "",
        }
        if not _TORCH_AVAILABLE:
            return out

        try:
            if torch.cuda.is_available():
                out["torch_cuda_mem_allocated_mb"] = round(torch.cuda.memory_allocated() / (1024 * 1024), 2)
                out["torch_cuda_mem_reserved_mb"] = round(torch.cuda.memory_reserved() / (1024 * 1024), 2)
        except Exception:
            pass
        return out

    def _read_cpu_ram_metrics(self) -> Dict[str, Any]:
        vm = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=None)

        return {
            "cpu_percent": cpu,
            "ram_percent": vm.percent,
            "ram_used_mb": round(vm.used / (1024 * 1024), 2),
            "ram_available_mb": round(vm.available / (1024 * 1024), 2),
        }

    @staticmethod
    def _read_cpu_ram_metrics_static() -> Dict[str, Any]:
        vm = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=None)

        return {
            "cpu_percent": cpu,
            "ram_percent": vm.percent,
            "ram_used_mb": round(vm.used / (1024 * 1024), 2),
            "ram_available_mb": round(vm.available / (1024 * 1024), 2),
        }

    @classmethod
    def _read_gpu_metrics_static(cls, gpu_index: int = 0) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "gpu_name": "",
            "gpu_util_percent": "",
            "gpu_mem_used_mb": "",
            "gpu_mem_total_mb": "",
        }

        if not _NVML_AVAILABLE:
            return out

        cls._init_nvml_class()
        if not cls._class_nvml_inited:
            return out

        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            name = pynvml.nvmlDeviceGetName(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)

            out["gpu_name"] = name.decode("utf-8") if isinstance(name, (bytes, bytearray)) else str(name)
            out["gpu_util_percent"] = int(util.gpu)
            out["gpu_mem_used_mb"] = round(mem.used / (1024 * 1024), 2)
            out["gpu_mem_total_mb"] = round(mem.total / (1024 * 1024), 2)
        except Exception:
            # leave blanks on failure
            pass

        return out

    @staticmethod
    def _read_torch_cuda_metrics_static() -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "torch_cuda_mem_allocated_mb": "",
            "torch_cuda_mem_reserved_mb": "",
        }
        if not _TORCH_AVAILABLE:
            return out

        try:
            if torch.cuda.is_available():
                out["torch_cuda_mem_allocated_mb"] = round(torch.cuda.memory_allocated() / (1024 * 1024), 2)
                out["torch_cuda_mem_reserved_mb"] = round(torch.cuda.memory_reserved() / (1024 * 1024), 2)
        except Exception:
            pass
        return out

    @classmethod
    def gather_resource_metrics(cls, gpu_index: int = 0) -> Dict[str, Any]:
        """
        Gather best-effort CPU/RAM, GPU (NVML), and Torch CUDA allocator metrics
        without instantiating MetricsLogger and without writing to disk.
        """
        out: Dict[str, Any] = {
            "timestamp": int(datetime.now().timestamp()),
        }
        out.update(cls._read_cpu_ram_metrics_static())
        out.update(cls._read_gpu_metrics_static(gpu_index=gpu_index))
        out.update(cls._read_torch_cuda_metrics_static())
        return out

    def _format_decimal(self, value: Any) -> Any:
        """
        Convert float -> string with comma decimal separator for Excel (EU locales).
        """
        #if isinstance(value, float):
        #    # fixed precision + comma decimal separator
        #    s = format(value, f".{self.float_precision}f")
        #    return s.replace(".", ",")
        return value

    def log_request(
        self,
        *,
        endpoint: str,
        origin_ip: str,
        model_id: str,
        options: InferenceOptions,
        status: str,
        latency_ms: int,
        thinking_ms: int,
        prompt_tokens: int,
        response_tokens: int,
        history_tokens: int,
        message_tokens: int,
        full_message_tokens: int
    ) -> None:
        """
        Append a single record.
        """
        record: Dict[str, Any] = {
            "timestamp": int(datetime.now().timestamp()),
            "endpoint": endpoint,
            "origin_ip": origin_ip,
            "model_id": model_id,
            "temperature": float(options.temperature),
            "top_p": float(options.top_p),
            "max_tokens": int(options.max_tokens),
            "stream": bool(options.stream),
            "status": status,
            "latency_ms": int(latency_ms),
            "thinking_ms": int(thinking_ms),
            "prompt_tokens": int(prompt_tokens),
            "response_tokens": int(response_tokens),
            "history_tokens": int(history_tokens),
            "current_message_tokens_raw": int(message_tokens),
            "current_message_tokens_templated": int(full_message_tokens),
        }

        record.update(self._read_cpu_ram_metrics())
        record.update(self._read_gpu_metrics())
        record.update(self._read_torch_cuda_metrics())

        # Format floats for Excel (comma decimal separator) right before writing
        formatted_record: Dict[str, Any] = {k: self._format_decimal(v) for k, v in record.items()}

        # Ensure CSV exists & append
        self._ensure_csv()
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=self.HEADER,
                delimiter=";",
                quotechar='"',
                quoting=csv.QUOTE_NONNUMERIC,
            )
            writer.writerow(formatted_record)
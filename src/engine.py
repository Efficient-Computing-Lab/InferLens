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
import os
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Tuple

import torch
import torch.nn.functional as F
from fastapi import HTTPException
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
from monitor import ModelTracer
from pcl import PCLAdapter

def _greedy_from_logits(logits):
    return torch.argmax(logits, dim=-1)

def _get_early_exit_heads(model):
    """
    Returns the final normalization module (if any) and lm_head needed to
    convert an intermediate layer hidden state into vocabulary logits.
    """
    backbone = getattr(model, "model", None)

    final_norm = None
    if backbone is not None:
        final_norm = getattr(backbone, "norm", None)
        if final_norm is None:
            final_norm = getattr(backbone, "final_layernorm", None)

    if final_norm is None:
        final_norm = getattr(model, "norm", None)
    if final_norm is None:
        final_norm = getattr(model, "final_layernorm", None)

    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        raise RuntimeError("Model does not expose lm_head, cannot compute early-exit tokens")

    return final_norm, lm_head


def _compute_layer_exit_tokens(model, tokenizer, hidden_states):
    """
    hidden_states is expected to be the tuple returned by Hugging Face when
    output_hidden_states=True:
      hidden_states[0] = embeddings output
      hidden_states[i+1] = output of decoder layer i

    Returns:
      {
        0: {"exit_token_id": ..., "exit_token_str": ..., "exit_token_text": ...},
        1: {...},
        ...
      }
    """
    if not hidden_states:
        return {}

    final_norm, lm_head = _get_early_exit_heads(model)
    layer_exit = {}

    # Skip hidden_states[0] because that is the embedding output, not a transformer layer
    for layer_idx, hs in enumerate(hidden_states[1:]):
        last_hidden = hs[:, -1, :]  # [batch, hidden_dim]

        if final_norm is not None:
            last_hidden = final_norm(last_hidden)

        layer_logits = lm_head(last_hidden)
        layer_token = _greedy_from_logits(layer_logits)

        tok_id = int(layer_token.item())
        tok_str = tokenizer.convert_ids_to_tokens([tok_id])[0]
        tok_text = tokenizer.decode([tok_id], skip_special_tokens=True)

        layer_exit[layer_idx] = {
            "exit_token_id": tok_id,
            "exit_token_str": tok_str,
            "exit_token_text": tok_text,
        }

    return layer_exit


def _attach_layer_exit_tokens_to_trace(trace, layer_exit_tokens):
    """
    Merges the per-layer exit-token info into the existing tracer layer entries.
    Works whether tracer layer keys are ints or strings.
    """
    layers = trace.get("layers")
    if not isinstance(layers, dict):
        return

    for layer_idx, info in layer_exit_tokens.items():
        if layer_idx in layers:
            key = layer_idx
        elif str(layer_idx) in layers:
            key = str(layer_idx)
        else:
            key = str(layer_idx)

        if key not in layers or not isinstance(layers[key], dict):
            layers[key] = {}

        layers[key].update(info)

def safe_model_dir(base_dir: str, model_id: str) -> str:
    safe_name = model_id.replace("/", "__")
    return str(Path(base_dir) / safe_name)

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

def _sample_top_p(logits, temperature: float, top_p: float):
    if temperature <= 0:
        return torch.argmax(logits, dim=-1)

    logits = logits / max(temperature, 1e-6)
    probs = F.softmax(logits, dim=-1)

    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        mask = cum > top_p
        # keep at least 1
        mask[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(mask, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        next_idx = torch.multinomial(sorted_probs, num_samples=1).squeeze(-1)
        return sorted_idx.gather(-1, next_idx.unsqueeze(-1)).squeeze(-1)

    return torch.multinomial(probs, num_samples=1).squeeze(-1)

@dataclass
class EngineConfig:
    model_id: str
    local_dir: str
    hf_token: Optional[str]
    load_in_4bit: bool
    dtype: str  # "auto"|"fp16"|"bf16"|"fp32"
    want_hid: bool


@dataclass
class EngineState:
    model_id: str
    local_dir: str
    device: str
    load_in_4bit: bool
    dtype: str
    tokenizer: Any
    model: Any
    pcl: Optional[Any] = None   # PCLAdapter instance, set after init


def download_model(model_id: str, local_dir: str, token: Optional[str]) -> str:
    model_dir = safe_model_dir(local_dir, model_id)
    os.makedirs(model_dir, exist_ok=True)
    path = snapshot_download(
        repo_id=model_id,
        local_dir=model_dir,
        token=token,
    )
    return path


def _pick_torch_dtype(dtype: str):
    d = dtype.lower()
    if d == "auto":
        return "auto"
    if d in ("fp16", "float16"):
        return torch.float16
    if d in ("bf16", "bfloat16"):
        return torch.bfloat16
    if d in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def load_model(model_path: str, load_in_4bit: bool, dtype: str) -> Tuple[Any, Any, str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    tokenizer_properties = {
        'name_or_path': tokenizer.name_or_path,
        'vocab_size': tokenizer.vocab_size,
        'model_max_length': tokenizer.model_max_length,
        'is_fast': tokenizer.is_fast,
        'padding_side': tokenizer.padding_side,
        'truncation_side': tokenizer.truncation_side,
        'clean_up_tokenization_spaces': tokenizer.clean_up_tokenization_spaces,
        'bos_token': tokenizer.bos_token,
        'eos_token': tokenizer.eos_token,
        'unk_token': tokenizer.unk_token,
        'sep_token': tokenizer.sep_token,
        'pad_token': tokenizer.pad_token,
        'mask_token': tokenizer.mask_token,
    }
    print("======================= TOKENIZER PROPS =======================")
    for key, value in tokenizer_properties.items():
        print(f"{key.rjust(30)}:  {value}")
    print("===============================================================")

    if device == "cuda":
        gpu_stats = torch.cuda.get_device_properties(0)
        gpu_properties = {
            'name': gpu_stats.name,
            'memory (GB)': round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3),
        }
        print("========================== GPU PROPS ==========================")
        for key, value in gpu_properties.items():
            print(f"{key.rjust(30)}:  {value}")
        print("===============================================================")

    quant_config = None
    if load_in_4bit and device == "cuda":
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=_pick_torch_dtype(dtype),
        quantization_config=quant_config,
        low_cpu_mem_usage=True,
        attn_implementation="eager",  # <-- key for real attention weights
    )

    if device == "cpu":
        model.to(device)

    model.eval()
    return tokenizer, model, device


class LLMEngine:
    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        self.state: Optional[EngineState] = None

    def init(self) -> EngineState:
        print(f"[{now_iso()}] Downloading model: {self.cfg.model_id}")
        model_path = download_model(self.cfg.model_id, self.cfg.local_dir, self.cfg.hf_token)
        print(f"[{now_iso()}] Model snapshot at: {model_path}")

        print(f"[{now_iso()}] Loading model (cuda={torch.cuda.is_available()}, 4bit={self.cfg.load_in_4bit}, dtype={self.cfg.dtype})")
        tokenizer, model, device = load_model(model_path, self.cfg.load_in_4bit, self.cfg.dtype)

        self.state = EngineState(
            model_id=self.cfg.model_id,
            local_dir=self.cfg.local_dir,
            device=device,
            load_in_4bit=self.cfg.load_in_4bit,
            dtype=self.cfg.dtype,
            tokenizer=tokenizer,
            model=model,
            pcl=PCLAdapter(
                model,
                tokenizer,
                device=device,
                checkpoint_dir="./pcl_checkpoints",
                lr=5e-3,
                accum_steps=1,
                drift_threshold=0.50,
                entropy_threshold=0.3,
                loss_ema_alpha=0.2,
                replay_ratio=0.0,      # disable replay until routing works
            )
        )
        return self.state

    # ---------- History / prompt building ----------

    @staticmethod
    def parse_history(history_json: str) -> List[Dict[str, str]]:
        try:
            raw = json.loads(history_json or "[]")
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid history JSON: {e}")

        if not isinstance(raw, list):
            raise HTTPException(status_code=400, detail="History must be a JSON array")

        msgs: List[Dict[str, str]] = []
        for m in raw:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            content = m.get("content")
            if role not in ("user", "assistant", "system"):
                continue
            if not isinstance(content, str):
                continue
            msgs.append({"role": role, "content": content})
        return msgs

    @staticmethod
    def build_chat_prompt(tokenizer, history_msgs: List[Dict[str, str]], new_user_message: str) -> str:
        messages = list(history_msgs) or []
        if new_user_message:
            messages.append({"role": "user", "content": new_user_message})

        if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        out = []
        for m in messages:
            out.append(f"{m['role'].upper()}: {m['content']}")
        out.append("ASSISTANT:")
        return "\n".join(out)

    # ---------- Generation ----------

    def generate_one_shot(self, prompt: str, max_tokens: int, temperature: float, top_p: float) -> str:
        if self.state is None:
            raise RuntimeError("Engine not initialized")

        tokenizer = self.state.tokenizer
        model = self.state.model
        device = self.state.device

        inputs = tokenizer(prompt, return_tensors="pt")
        if device == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}

        gen_kwargs: Dict[str, Any] = dict(**inputs, max_new_tokens=max_tokens)
        if temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
        else:
            gen_kwargs.update(do_sample=False)

        with torch.no_grad():
            out = model.generate(**gen_kwargs)

        text = tokenizer.decode(out[0], skip_special_tokens=True)
        if text.startswith(prompt):
            text = text[len(prompt):]
        return text

    def generate_stream(self, prompt: str, max_tokens: int, temperature: float, top_p: float) -> Generator[str, None, None]:
        """
        Streams NDJSON lines like Ollama.
        """
        if self.state is None:
            raise RuntimeError("Engine not initialized")

        tokenizer = self.state.tokenizer
        model = self.state.model
        device = self.state.device

        inputs = tokenizer(prompt, return_tensors="pt")
        tokens = tokenizer.tokenize(prompt)
        if device == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        #print("-------------- request --------------")
        #print(prompt)
        #print()
        #print(tokens)
        #print()
        #print(tokenizer.convert_tokens_to_ids(tokens))
        #print("-------------------------------------")

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        gen_kwargs: Dict[str, Any] = dict(**inputs, max_new_tokens=max_tokens, streamer=streamer)
        if temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
        else:
            gen_kwargs.update(do_sample=False)

        import threading
        thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        for chunk in streamer:
            obj = {
                "model": self.state.model_id,
                "created_at": now_iso(),
                "response": chunk,
                "done": False,
            }
            yield json.dumps(obj, ensure_ascii=False) + "\n"

        thread.join()

        final_obj = {
            "model": self.state.model_id,
            "created_at": now_iso(),
            "response": "",
            "done": True,
        }
        yield json.dumps(final_obj, ensure_ascii=False) + "\n"

    def generate_stream_traced(self, prompt: str, max_tokens: int, temperature: float, top_p: float, enable_pcl: bool = True):
        if self.state is None:
            raise RuntimeError("Engine not initialized")

        tokenizer = self.state.tokenizer
        model = self.state.model
        device = self.state.device
        pcl_stats_dict = {}

        def sync_cuda():
            if device == "cuda":
                torch.cuda.synchronize()

        def gpu_mem_stats():
            if device != "cuda":
                return {
                    "device": device,
                    "torch_allocated_mb": 0.0,
                    "torch_reserved_mb": 0.0,
                    "torch_peak_allocated_mb": 0.0,
                    "torch_peak_reserved_mb": 0.0,
                    "gpu_total_mb": 0.0,
                }

            props = torch.cuda.get_device_properties(0)
            total_mb = props.total_memory / 1024 / 1024

            return {
                "device": device,
                "torch_allocated_mb": torch.cuda.memory_allocated(0) / 1024 / 1024,
                "torch_reserved_mb": torch.cuda.memory_reserved(0) / 1024 / 1024,
                "torch_peak_allocated_mb": torch.cuda.max_memory_allocated(0) / 1024 / 1024,
                "torch_peak_reserved_mb": torch.cuda.max_memory_reserved(0) / 1024 / 1024,
                "gpu_total_mb": total_mb,
            }

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats(0)

        tracer = ModelTracer(topk=12).attach(model)
        want_attn = False
        want_hid = self.cfg.want_hid

        try:
            # -----------------------------
            # 1) Tokenization on CPU
            # -----------------------------
            t_total0 = time.perf_counter()

            t0 = time.perf_counter()
            enc = tokenizer(prompt, return_tensors="pt")
            sync_cuda()
            tokenize_ms = (time.perf_counter() - t0) * 1000.0

            input_ids = enc["input_ids"]
            attention_mask = enc.get("attention_mask")

            prompt_token_count = int(input_ids.shape[1])

            # -----------------------------
            # 2) Host -> device transfer
            # -----------------------------
            mem_before_transfer = gpu_mem_stats()

            t1 = time.perf_counter()
            input_ids = input_ids.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            sync_cuda()
            transfer_ms = (time.perf_counter() - t1) * 1000.0

            mem_before_prefill = gpu_mem_stats()

            # -----------------------------
            # 3) Prefill forward pass
            # -----------------------------
            with torch.inference_mode():
                tracer.next_step()
                tracer.enable_timing(True, phase="prefill")

                t2 = time.perf_counter()
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    output_attentions=want_attn,
                    output_hidden_states=want_hid,
                    return_dict=True,
                )
                layer_exit_tokens = _compute_layer_exit_tokens(model, tokenizer, out.hidden_states)
                sync_cuda()
                prefill_ms = (time.perf_counter() - t2) * 1000.0

                tracer.enable_timing(False)
                past = out.past_key_values
                mem_after_prefill = gpu_mem_stats()

                kv_seq_len = None
                if past is not None:
                    if hasattr(past, "get_seq_length"):
                        try:
                            kv_seq_len = int(past.get_seq_length())
                        except TypeError:
                            kv_seq_len = int(past.get_seq_length(0))
                    elif isinstance(past, (list, tuple)) and len(past) > 0:
                        k0 = past[0][0]
                        if k0 is not None and k0.dim() >= 3:
                            kv_seq_len = int(k0.shape[-2])

                # -----------------------------
                # 4) First token sampling
                # -----------------------------
                t3 = time.perf_counter()
                logits = out.logits[:, -1, :]
                next_token = _sample_top_p(logits, temperature, top_p)
                sync_cuda()
                first_token_sample_ms = (time.perf_counter() - t3) * 1000.0

                first_token_latency_ms = (time.perf_counter() - t_total0) * 1000.0

                tok_id = int(next_token.item())
                tok_str = tokenizer.convert_ids_to_tokens([tok_id])[0]
                chunk = tokenizer.decode([tok_id], skip_special_tokens=True)
                full_response = chunk

                trace = tracer.frames[-1]
                trace["kv_seq_len"] = kv_seq_len
                trace["next_token_id"] = tok_id
                trace["next_token_str"] = tok_str
                _attach_layer_exit_tokens_to_trace(trace, layer_exit_tokens)

                top_vals, top_idx = torch.topk(logits[0], k=5)
                trace["logit_top5"] = [
                    {
                        "id": int(top_id),
                        "token": tokenizer.convert_ids_to_tokens([int(top_id)])[0],
                        "logit": float(val),
                    }
                    for val, top_id in zip(top_vals, top_idx)
                ]

                # Emit a special "prefill" event BEFORE the first token trace
                yield json.dumps({
                    "type": "prefill",
                    "model": self.state.model_id,
                    "created_at": now_iso(),
                    "done": False,
                    "prefill": {
                        "prompt_tokens": prompt_token_count,
                        "tokenize_ms": round(tokenize_ms, 3),
                        "transfer_ms": round(transfer_ms, 3),
                        "prefill_ms": round(prefill_ms, 3),
                        "first_token_sample_ms": round(first_token_sample_ms, 3),
                        "first_token_latency_ms": round(first_token_latency_ms, 3),
                        "kv_seq_len_after_prefill": kv_seq_len,
                        "memory_before_transfer": mem_before_transfer,
                        "memory_before_prefill": mem_before_prefill,
                        "memory_after_prefill": mem_after_prefill,
                        "layers": trace.get("layers", {}),
                    }
                }, ensure_ascii=False) + "\n"

                # Emit first generated token trace
                yield json.dumps({
                    "model": self.state.model_id,
                    "created_at": now_iso(),
                    "response": chunk,
                    "done": False,
                    "trace": trace,
                }, ensure_ascii=False) + "\n"

                if tokenizer.eos_token_id is not None and tok_id == tokenizer.eos_token_id:
                    # PCL observe even for single-token responses
                    if enable_pcl and self.state.pcl is not None and full_response.strip():
                        try:
                            pcl_result = self.state.pcl.observe(prompt, full_response)
                            pcl_stats_dict = pcl_result.to_dict()
                        except Exception as pcl_exc:
                            pcl_stats_dict = {"error": str(pcl_exc)}
                    yield json.dumps({
                        "model": self.state.model_id,
                        "created_at": now_iso(),
                        "response": "",
                        "pcl": pcl_stats_dict,
                        "done": True,
                    }, ensure_ascii=False) + "\n"
                    return

                # -----------------------------
                # 5) Normal decode loop
                # -----------------------------
                for step in range(1, max_tokens):
                    tracer.next_step()

                    t_decode = time.perf_counter()
                    out = model(
                        input_ids=next_token.view(1, 1),
                        past_key_values=past,
                        use_cache=True,
                        output_attentions=want_attn,
                        output_hidden_states=want_hid,
                        return_dict=True,
                    )
                    layer_exit_tokens = _compute_layer_exit_tokens(model, tokenizer, out.hidden_states)
                    sync_cuda()
                    decode_forward_ms = (time.perf_counter() - t_decode) * 1000.0

                    past = out.past_key_values

                    logits = out.logits[:, -1, :]
                    next_token = _sample_top_p(logits, temperature, top_p)

                    tok_id = int(next_token.item())
                    tok_str = tokenizer.convert_ids_to_tokens([tok_id])[0]
                    chunk = tokenizer.decode([tok_id], skip_special_tokens=True)
                    full_response += chunk

                    kv_seq_len = None
                    if past is not None:
                        if hasattr(past, "get_seq_length"):
                            try:
                                kv_seq_len = int(past.get_seq_length())
                            except TypeError:
                                kv_seq_len = int(past.get_seq_length(0))
                        elif isinstance(past, (list, tuple)) and len(past) > 0:
                            k0 = past[0][0]
                            if k0 is not None and k0.dim() >= 3:
                                kv_seq_len = int(k0.shape[-2])

                    trace = tracer.frames[-1]
                    trace["kv_seq_len"] = kv_seq_len
                    trace["next_token_id"] = tok_id
                    trace["next_token_str"] = tok_str
                    trace["decode_forward_ms"] = round(decode_forward_ms, 3)
                    _attach_layer_exit_tokens_to_trace(trace, layer_exit_tokens)

                    top_vals, top_idx = torch.topk(logits[0], k=5)
                    trace["logit_top5"] = [
                        {
                            "id": int(top_id),
                            "token": tokenizer.convert_ids_to_tokens([int(top_id)])[0],
                            "logit": float(val),
                        }
                        for val, top_id in zip(top_vals, top_idx)
                    ]

                    yield json.dumps({
                        "model": self.state.model_id,
                        "created_at": now_iso(),
                        "response": chunk,
                        "done": False,
                        "trace": trace,
                    }, ensure_ascii=False) + "\n"

                    if tokenizer.eos_token_id is not None and tok_id == tokenizer.eos_token_id:
                        break

            # ── PCL: observe this turn ───────────────────────────────────────
            if enable_pcl and self.state.pcl is not None:
                try:
                    pcl_result = self.state.pcl.observe(prompt, full_response)
                    pcl_stats_dict = pcl_result.to_dict()
                except Exception as pcl_exc:
                    print({"error": str(pcl_exc)})
                    pcl_stats_dict = {"error": str(pcl_exc)}

            yield json.dumps({
                "model": self.state.model_id,
                "created_at": now_iso(),
                "response": "",
                "pcl": pcl_stats_dict,
                "done": True,
            }, ensure_ascii=False) + "\n"

        finally:
            tracer.detach()
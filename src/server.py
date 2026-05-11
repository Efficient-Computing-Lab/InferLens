from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional
from pydantic import BaseModel

import torch
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from engine import LLMEngine
from monitor import MetricsLogger, InferenceOptions

METRICS = MetricsLogger(csv_path="../metrics/metrics.csv", gpu_index=0)

class MonitorRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9

def sse_event(obj: dict) -> str:
    return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"

def create_app(engine: LLMEngine, interface_dir: Optional[Path] = None) -> FastAPI:
    app = FastAPI(title="Local Llama Ollama-like API", version="0.1")

    # Local dev: permissive. Could be tighten later if needed.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if interface_dir is not None and interface_dir.exists():
        app.mount("/interface", StaticFiles(directory=str(interface_dir)), name="interface")

        @app.get("/")
        async def ui_index():
            return FileResponse(str(interface_dir / "monitor.html"))

    @app.get("/monitor")
    async def ui_monitor():
        if interface_dir is None:
            return JSONResponse({"error": "Monitor UI not available"}, status_code=404)
        return FileResponse(str(interface_dir / "monitor.html"))

    @app.post("/api/generate")
    async def api_generate(req: Request):
        if engine.state is None:
            return JSONResponse({"error": "Model not loaded"}, status_code=503)

        body = await req.json()
        prompt = body.get("prompt", "")
        stream = bool(body.get("stream", True))

        options = body.get("options", {}) or {}
        max_tokens = int(options.get("num_predict", options.get("max_tokens", 256)))
        temperature = float(options.get("temperature", 0.7))
        top_p = float(options.get("top_p", 0.9))

        if not isinstance(prompt, str) or not prompt.strip():
            return JSONResponse({"error": "Missing 'prompt' (string)"}, status_code=400)

        origin_ip = req.client.host if req.client else ""
        model_id = engine.state.model_id
        opts = InferenceOptions(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=stream,
        )

        t0 = time.time()
        status = "ok"

        try:
            if stream:
                # Wrap the generator so we can log after it finishes
                def wrapped_gen():
                    nonlocal status
                    try:
                        for line in engine.generate_stream(prompt, max_tokens, temperature, top_p):
                            yield line
                    except Exception:
                        status = "error"
                        raise
                    finally:
                        latency_ms = int((time.time() - t0) * 1000)
                        METRICS.log_request(
                            endpoint="/api/generate",
                            origin_ip=origin_ip,
                            model_id=model_id,
                            options=opts,
                            status=status,
                            latency_ms=latency_ms,
                        )

                return StreamingResponse(wrapped_gen(), media_type="application/x-ndjson")

            # Non-stream inference
            text = engine.generate_one_shot(prompt, max_tokens, temperature, top_p)

            latency_ms = int((time.time() - t0) * 1000)
            METRICS.log_request(
                endpoint="/api/generate",
                origin_ip=origin_ip,
                model_id=model_id,
                options=opts,
                status=status,
                latency_ms=latency_ms,
            )

            return JSONResponse({
                "model": model_id,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
                "response": text,
                "done": True,
            })

        except Exception:
            status = "error"
            # Log errors for non-stream requests too
            if not stream:
                latency_ms = int((time.time() - t0) * 1000)
                METRICS.log_request(
                    endpoint="/api/generate",
                    origin_ip=origin_ip,
                    model_id=model_id,
                    options=opts,
                    status=status,
                    latency_ms=latency_ms,
                )
            raise

    @app.get("/health")
    async def health():
        ok = engine.state is not None
        return JSONResponse({
            "ok": ok,
            "model": engine.state.model_id if engine.state else None,
            "device": engine.state.device if engine.state else None,
            "cuda_available": torch.cuda.is_available(),
        })

    @app.post("/api/chat")
    async def api_chat(
        request: Request,
        message: str = Form(""),
        history: str = Form("[]"),
        file: UploadFile | None = File(None),
    ):
        if engine.state is None:
            return JSONResponse({"error": "Model not loaded"}, status_code=503)

        # Options used by this endpoint (currently hard-coded)
        temperature = 0.7
        top_p = 0.9
        max_tokens = 1024
        stream = False

        origin_ip = request.client.host if request.client else ""
        model_id = engine.state.model_id
        opts = InferenceOptions(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=stream,
        )

        t0 = time.time()
        status = "ok"

        completion_token_count = 0
        prompt_token_count = 0
        history_token_count = 0
        current_message_tokens_raw = 0
        current_message_tokens_templated = 0

        try:
            history_msgs = engine.parse_history(history)
            user_msg = message or ("(file upload)" if file else "")
            prompt = engine.build_chat_prompt(engine.state.tokenizer, history_msgs, user_msg)
            prompt_inputs = engine.state.tokenizer(prompt, return_tensors="pt")
            prompt_token_count = prompt_inputs["input_ids"].shape[1]

            current_message_tokens_raw = len(
                engine.state.tokenizer.encode(user_msg, add_special_tokens=False)
            )

            prompt_current_only = engine.build_chat_prompt(engine.state.tokenizer, [], user_msg)
            current_message_tokens_templated = engine.state.tokenizer(
                prompt_current_only, return_tensors="pt"
            )["input_ids"].shape[1]

            if history_msgs and len(history_msgs) > 0:
                prompt_history_only = engine.build_chat_prompt(engine.state.tokenizer, history_msgs, "")
                history_token_count = engine.state.tokenizer(
                    prompt_history_only, return_tensors="pt"
                )["input_ids"].shape[1]

            text = engine.generate_one_shot(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            latency_ms = int((time.time() - t0) * 1000)
            if text:
                completion_inputs = engine.state.tokenizer(
                    text,
                    return_tensors="pt"
                )
                completion_token_count = completion_inputs["input_ids"].shape[1]

            METRICS.log_request(
                endpoint="/api/chat",
                origin_ip=origin_ip,
                model_id=model_id,
                options=opts,
                status=status,
                latency_ms=latency_ms,
                thinking_ms = latency_ms,
                prompt_tokens = prompt_token_count,
                response_tokens = completion_token_count,
                history_tokens = history_token_count,
                message_tokens = current_message_tokens_raw,
                full_message_tokens = current_message_tokens_templated
            )

            return JSONResponse({"reply": text, "model": model_id, "t_ms": latency_ms})

        except Exception:
            status = "error"
            latency_ms = int((time.time() - t0) * 1000)

            METRICS.log_request(
                endpoint="/api/chat",
                origin_ip=origin_ip,
                model_id=model_id,
                options=opts,
                status=status,
                latency_ms=latency_ms,
                thinking_ms = latency_ms,
                prompt_tokens = prompt_token_count,
                response_tokens = completion_token_count,
                history_tokens = history_token_count,
                message_tokens = current_message_tokens_raw,
                full_message_tokens = current_message_tokens_templated
            )
            raise

    @app.post("/api/chat_stream")
    async def api_chat_stream(
        request: Request,
        message: str = Form(""),
        history: str = Form("[]"),
        file: UploadFile | None = File(None),
    ):
        if engine.state is None:
            return JSONResponse({"error": "Model not loaded"}, status_code=503)

        # Options used by this endpoint (currently hard-coded)
        temperature = 0.7
        top_p = 0.9
        max_tokens = 1024
        stream = True

        origin_ip = request.client.host if request.client else ""
        model_id = engine.state.model_id
        opts = InferenceOptions(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=stream,
        )

        # Build prompt (history-aware)
        history_msgs = engine.parse_history(history)
        user_msg = message or ("(file upload)" if file else "")
        prompt = engine.build_chat_prompt(engine.state.tokenizer, history_msgs, user_msg)
        prompt_inputs = engine.state.tokenizer(prompt, return_tensors="pt")
        prompt_token_count = prompt_inputs["input_ids"].shape[1]
        current_message_tokens_raw = len(
            engine.state.tokenizer.encode(user_msg, add_special_tokens=False)
        )

        prompt_current_only = engine.build_chat_prompt(engine.state.tokenizer, [], user_msg)
        current_message_tokens_templated = engine.state.tokenizer(
            prompt_current_only, return_tensors="pt"
        )["input_ids"].shape[1]

        history_token_count = 0
        if history_msgs and len(history_msgs) > 0:
            prompt_history_only = engine.build_chat_prompt(engine.state.tokenizer, history_msgs, "")
            history_token_count = engine.state.tokenizer(
                prompt_history_only, return_tensors="pt"
            )["input_ids"].shape[1]

        t0 = time.time()
        status = "ok"

        def wrapped_gen():
            nonlocal status
            full_response = ""
            thinking_ms = 0
            try:
                for line in engine.generate_stream(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                ):
                    if(thinking_ms == 0):
                        thinking_ms = int((time.time() - t0) * 1000)
                    evt = json.loads(line)
                    chunk = evt.get("response", "")
                    done = evt.get("done", False)

                    if chunk:
                        full_response += chunk
                        yield sse_event({"type": "delta", "text": chunk})

                    if done:
                        yield sse_event({"type": "done"})
                        break

            except Exception as e:
                status = "error"
                # Still emit an SSE error event to the client
                yield sse_event({"type": "error", "error": str(e)})
            finally:
                latency_ms = int((time.time() - t0) * 1000)
                completion_token_count = 0
                if full_response:
                    completion_inputs = engine.state.tokenizer(
                        full_response,
                        return_tensors="pt"
                    )
                    completion_token_count = completion_inputs["input_ids"].shape[1]

                METRICS.log_request(
                    endpoint="/api/chat_stream",
                    origin_ip=origin_ip,
                    model_id=model_id,
                    options=opts,
                    status=status,
                    latency_ms=latency_ms,
                    thinking_ms = thinking_ms,
                    prompt_tokens = prompt_token_count,
                    response_tokens = completion_token_count,
                    history_tokens = history_token_count,
                    message_tokens = current_message_tokens_raw,
                    full_message_tokens = current_message_tokens_templated
                )

        return StreamingResponse(
            wrapped_gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    
    @app.post("/api/chat_monitor_stream")
    async def api_chat_monitor_stream(
        request: Request,
        message: str = Form(""),
        history: str = Form("[]"),
        file: UploadFile | None = File(None),
        max_tokens: int = Form(256),
        temperature: float = Form(0.7),
        top_p: float = Form(0.9),
        enable_pcl: bool = Form(False),
    ):
        if engine.state is None:
            return JSONResponse({"error": "Model not loaded"}, status_code=503)

        history_msgs = engine.parse_history(history)
        user_msg = message or ("(file upload)" if file else "")
        user_msg = user_msg.strip()

        if not user_msg:
            return JSONResponse({"error": "Missing prompt"}, status_code=400)

        prompt = engine.build_chat_prompt(
            engine.state.tokenizer,
            history_msgs,
            user_msg,
        )

        prompt_inputs = engine.state.tokenizer(prompt, return_tensors="pt")
        prompt_token_count = prompt_inputs["input_ids"].shape[1]
        current_message_tokens_raw = len(
            engine.state.tokenizer.encode(user_msg, add_special_tokens=False)
        )

        prompt_current_only = engine.build_chat_prompt(engine.state.tokenizer, [], user_msg)
        current_message_tokens_templated = engine.state.tokenizer(
            prompt_current_only, return_tensors="pt"
        )["input_ids"].shape[1]

        history_token_count = 0
        if history_msgs and len(history_msgs) > 0:
            prompt_history_only = engine.build_chat_prompt(engine.state.tokenizer, history_msgs, "")
            history_token_count = engine.state.tokenizer(
                prompt_history_only, return_tensors="pt"
            )["input_ids"].shape[1]

        t0 = time.time()
        status = "ok"
        origin_ip = request.client.host if request.client else ""
        model_id = engine.state.model_id
        opts = InferenceOptions(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=True,
        )

        def wrapped_gen():
            nonlocal status
            full_response = ""
            thinking_ms = 0

            try:
                for line in engine.generate_stream_traced(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    enable_pcl=enable_pcl,
                ):
                    if(thinking_ms == 0):
                        thinking_ms = int((time.time() - t0) * 1000)
                    evt = json.loads(line)
                    chunk = evt.get("response", "")

                    if chunk:
                        full_response += chunk

                    yield line
            except Exception as e:
                status = "error"

                yield json.dumps({
                    "error": str(e),
                    "done": True,
                }, ensure_ascii=False) + "\n"
            
            finally:
                latency_ms = int((time.time() - t0) * 1000)
                completion_token_count = 0
                if full_response:
                    completion_inputs = engine.state.tokenizer(
                        full_response,
                        return_tensors="pt"
                    )
                    completion_token_count = completion_inputs["input_ids"].shape[1]

                METRICS.log_request(
                    endpoint="/api/chat_monitor_stream",
                    origin_ip=origin_ip,
                    model_id=model_id,
                    options=opts,
                    status=status,
                    latency_ms=latency_ms,
                    thinking_ms = thinking_ms,
                    prompt_tokens = prompt_token_count,
                    response_tokens = completion_token_count,
                    history_tokens = history_token_count,
                    message_tokens = current_message_tokens_raw,
                    full_message_tokens = current_message_tokens_templated
                )

        return StreamingResponse(
            wrapped_gen(),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app
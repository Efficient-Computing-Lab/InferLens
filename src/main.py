#!/usr/bin/env python3
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

import argparse
from pathlib import Path

import uvicorn

from hugface_token import TOKEN
from engine import EngineConfig, LLMEngine
from server import create_app

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="meta-llama/Meta-Llama-3-8B-Instruct",
                   help="HF model repo id (may be gated).")
    p.add_argument("--local_dir", default="../hf_models",
                   help="Where to store downloaded snapshots.")
    p.add_argument("--load_in_4bit", action="store_true",
                   help="Use bitsandbytes 4-bit quantization (CUDA only).")
    p.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16", "fp32"],
                   help="Weights dtype (auto recommended).")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=80,
                   help="Server port.")
    p.add_argument("--hidden_state", action="store_true",
                   help="Calculate tokens per layer.")
    args = p.parse_args()

    cfg = EngineConfig(
        model_id=args.model_id,
        local_dir=args.local_dir,
        hf_token=TOKEN,
        load_in_4bit=args.load_in_4bit,
        dtype=args.dtype,
        want_hid=args.hidden_state,
    )

    engine = LLMEngine(cfg)
    engine.init()

    base_dir = Path(__file__).resolve().parent
    interface_dir = base_dir / "interface"

    app = create_app(engine, interface_dir=interface_dir)

    print(f"Ready. Starting API at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

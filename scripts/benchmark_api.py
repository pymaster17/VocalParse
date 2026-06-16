#!/usr/bin/env python3
"""End-to-end benchmark for ``VocalParseTranscriber`` on Opencpop.

Mirrors ``scripts/benchmark_inference.py`` but exercises the public
``vocalparse.api.VocalParseTranscriber`` path: float32 numpy arrays in,
decoded text out. mel extraction + audio encode + decode all happen
inside ``transcribe()``.

Audio I/O is excluded from the timed window — every rank pre-loads all
wavs into RAM, ranks barrier, then we time the single ``transcribe()``
call (which internally shards across ranks via ``batches[rank::world_size]``).

Launch:

    torchrun --nproc_per_node=4 scripts/benchmark_api.py \\
        --checkpoint /path/to/vocalparse \\
        --json data/Opencpop.json \\
        --audio_root /dataset/cyk/Opencpop
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from vocalparse import VocalParseTranscriber
from vocalparse.distributed import init_distributed
from vocalparse.model import load_audio


def _load_one(arg):
    idx, path = arg
    return idx, load_audio(path, sr=16000).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--json", default="data/Opencpop.json")
    parser.add_argument("--audio_root", default="/dataset/cyk/Opencpop")
    parser.add_argument("--num", type=int, default=-1,
                        help="Limit samples (-1 = all).")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--batch_mel_tokens", type=int, default=48000)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--load_workers", type=int, default=16,
                        help="Threads for librosa loading.")
    args = parser.parse_args()

    rank, world = init_distributed()
    is_rank0 = rank == 0

    with open(args.json) as f:
        items = json.load(f)
    if args.num > 0:
        items = items[: args.num]
    paths = [os.path.join(args.audio_root, it["wav_fn"]) for it in items]
    if is_rank0:
        print(f"[rank0] samples={len(paths)} world={world}", flush=True)

    # ── Pre-load audio (each rank loads same list — page-cache amortises) ──
    t0 = time.perf_counter()
    audios = [None] * len(paths)
    with ThreadPoolExecutor(max_workers=args.load_workers) as pool:
        for idx, wav in pool.map(_load_one, list(enumerate(paths))):
            audios[idx] = wav
    t_load = time.perf_counter() - t0
    total_audio_sec = sum(len(w) for w in audios) / 16000.0
    if is_rank0:
        print(f"[rank0] loaded {len(audios)} wavs in {t_load:.1f}s, "
              f"audio_dur={total_audio_sec:.1f}s", flush=True)

    # ── Build transcriber ─────────────────────────────────────────────
    trx = VocalParseTranscriber(checkpoint=args.checkpoint)

    # ── Warmup pass on a tiny slice to amortise CUDA / kernel JIT ──────
    if is_rank0:
        print("[rank0] warmup...", flush=True)
    _ = trx.transcribe(
        audios[: min(8, len(audios))],
        max_new_tokens=64,
        batch_size=args.batch_size,
        batch_mel_tokens=args.batch_mel_tokens,
    )

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    # ── Timed e2e call ────────────────────────────────────────────────
    t_start = time.perf_counter()
    results = trx.transcribe(
        audios,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        batch_mel_tokens=args.batch_mel_tokens,
    )
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    wall = time.perf_counter() - t_start

    if is_rank0:
        rtf = wall / total_audio_sec
        thr = len(audios) / wall
        print()
        print("=" * 60)
        print("e2e VocalParseTranscriber benchmark (Opencpop)")
        print("=" * 60)
        print(f"  samples            : {len(audios)}")
        print(f"  world_size         : {world}")
        print(f"  batch_size         : {args.batch_size}")
        print(f"  batch_mel_tokens   : {args.batch_mel_tokens}")
        print(f"  max_new_tokens     : {args.max_new_tokens}")
        print(f"  audio total (sec)  : {total_audio_sec:.2f}")
        print(f"  wall (transcribe)  : {wall:.2f}s")
        print(f"  RTF                : {rtf:.4f}")
        print(f"  realtime factor    : {1.0 / rtf:.1f}x")
        print(f"  throughput         : {len(audios) / wall:.2f} samples/s")
        print()
        print(f"  sample[0][:200]    : {results[0][:200] if results else '<none>'}")
        print(f"  sample[-1][:200]   : {results[-1][:200] if results else '<none>'}")

    trx.close()


if __name__ == "__main__":
    main()

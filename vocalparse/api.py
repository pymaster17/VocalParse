# coding=utf-8
# VocalParse end-to-end transcribe API.
#
# Single class entry-point for downstream programs (e.g. SVS validation):
#   trx = VocalParseTranscriber(checkpoint="...")
#   results = trx.transcribe([wav_np_a, wav_np_b, ...])  # list[str] on rank 0
#
# Multi-GPU == launch the caller via torchrun (the constructor reads RANK /
# WORLD_SIZE from env). Single-process invocation reduces to single-GPU.

import fcntl
import os
from concurrent.futures import ThreadPoolExecutor

import torch

from vocalparse.distributed import (
    init_distributed,
    cleanup_distributed,
    pack_batches,
    left_pad_input_ids,
    pre_encode_audio_features,
    gather_results_via_shm,
)


_MEL_FRAMES_PER_SEC = 100  # Whisper feat-ext: 16 kHz / 160 hop


def _claim_next(counter_path: str) -> int:
    """Atomically read-and-increment a shared int on /dev/shm.

    All ranks on the node share one tiny file; each call locks, reads,
    writes value+1, unlocks. /dev/shm is tmpfs so there's no real I/O.
    """
    fd = os.open(counter_path, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        os.lseek(fd, 0, os.SEEK_SET)
        raw = os.read(fd, 64).decode().strip()
        v = int(raw) if raw else 0
        os.lseek(fd, 0, os.SEEK_SET)
        os.ftruncate(fd, 0)
        os.write(fd, str(v + 1).encode())
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
    return v


class VocalParseTranscriber:
    """End-to-end batch transcription: float32 audio arrays → raw decoded text.

    The returned strings are exactly what ``tokenizer.decode(..., skip_special_tokens=False)``
    produces — caller-side parsing (lyrics / pitch / note / bpm) lives in the
    caller's own library (e.g. ``vocalparse.evaluation.parse_transcription_text``).

    Multi-GPU: launch the caller via ``torchrun --nproc_per_node=N``. Each rank
    holds its own model copy; ``transcribe()`` shards batches across ranks and
    gathers results to rank 0. World size 1 (no torchrun) is supported as a
    no-op.
    """

    def __init__(
        self,
        checkpoint: str,
        attn_implementation: str = "flash_attention_2",
    ):
        from vocalparse.model import load_model

        was_distributed = (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )
        self.rank, self.world_size = init_distributed()
        self._owns_distributed = (
            not was_distributed
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )

        cfg = {"checkpoint": checkpoint, "attn_implementation": attn_implementation}
        self.model, self.processor, self.device = load_model(cfg)
        self.tokenizer = self.processor.tokenizer
        self._pad_token_id = self.tokenizer.pad_token_id or 151643

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if self._owns_distributed:
            cleanup_distributed()

    @torch.no_grad()
    def transcribe(
        self,
        audios,
        sr: int = 16000,
        max_new_tokens: int = 512,
        batch_size: int = 32,
        batch_mel_tokens: int = 24000,
        lyrics=None,
        inference_mode: str = "audio-only",
    ):
        """Transcribe a batch of mono float32 audio arrays.

        Args:
            audios: list of 1D ``np.float32`` arrays (mono PCM at ``sr``).
            sr: sample rate of every input array. Must match the model's
                feature extractor (16000); no resampling is done here.
            max_new_tokens: decoder generation cap.
            batch_size: max samples per packed batch.
            batch_mel_tokens: max total mel frames per packed batch.
            lyrics: optional list of per-sample GT lyric strings. Required
                (and every entry must be non-empty) when
                ``inference_mode="audio-lyric"``; ignored otherwise.
            inference_mode: ``"audio-only"`` (default; the model decodes
                lyrics + interleaved AST from audio alone) or
                ``"audio-lyric"`` (caller supplies GT lyrics as the prompt
                prefix; the model only predicts the interleaved AST).

        Returns:
            On rank 0: ``list[str]`` of decoded text, in the same order as
            ``audios``. On other ranks: ``None``.
        """
        from vocalparse.prompts import build_prefix_text

        if inference_mode not in ("audio-only", "audio-lyric"):
            raise ValueError(
                f"inference_mode must be 'audio-only' or 'audio-lyric', "
                f"got {inference_mode!r}"
            )
        if inference_mode == "audio-lyric":
            if lyrics is None or len(lyrics) != len(audios) or any(
                not l for l in lyrics
            ):
                raise ValueError(
                    "inference_mode='audio-lyric' requires `lyrics` to be a "
                    "list of non-empty strings, one per audio"
                )
            prefixes = [
                build_prefix_text(self.processor, lyrics_text=l) for l in lyrics
            ]
        else:
            shared = build_prefix_text(self.processor)
            prefixes = [shared] * len(audios)

        indexed_samples = [
            (i, {
                "audio": wav,
                "prefix": pfx,
                "mel_frames": (len(wav) * _MEL_FRAMES_PER_SEC) // sr,
            })
            for i, (wav, pfx) in enumerate(zip(audios, prefixes))
        ]
        batches = pack_batches(indexed_samples, batch_mel_tokens, batch_size)

        # ── Shared work-steal counter on /dev/shm ──────────────────────
        # All ranks claim batches one-at-a-time from a single counter, so
        # a slow batch on one GPU doesn't block the others.
        counter_path = (
            f"/dev/shm/_vp_api_counter_"
            f"{os.environ.get('MASTER_PORT', '0')}_{os.getppid()}"
        )
        if self.rank == 0:
            with open(counter_path, "w") as f:
                f.write("0")
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        my_results = []
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="api_prep") as pool:
            # Bootstrap: claim first batch, kick off its CPU prep.
            first_idx = _claim_next(counter_path)
            if first_idx < len(batches):
                prep_idx = first_idx
                first_audios = [s["audio"] for _, s in batches[first_idx]]
                first_prefixes = [s["prefix"] for _, s in batches[first_idx]]
                prep_future = pool.submit(
                    self._prepare_audio_batch, first_audios, first_prefixes, sr,
                )
            else:
                prep_future = None

            while prep_future is not None:
                cur_idx = prep_idx
                cur_batch = batches[cur_idx]
                prepared = prep_future.result()

                # Claim next BEFORE running GPU on current → CPU prep of
                # batch n+1 overlaps with GPU generate of batch n.
                next_idx = _claim_next(counter_path)
                if next_idx < len(batches):
                    prep_idx = next_idx
                    next_audios = [s["audio"] for _, s in batches[next_idx]]
                    next_prefixes = [s["prefix"] for _, s in batches[next_idx]]
                    prep_future = pool.submit(
                        self._prepare_audio_batch, next_audios, next_prefixes, sr,
                    )
                else:
                    prep_future = None

                texts = self._run_generate(prepared, max_new_tokens)
                for (idx, _), text in zip(cur_batch, texts):
                    my_results.append((idx, text))

        # All ranks have drained the counter — safe to unlink.
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        if self.rank == 0:
            try:
                os.unlink(counter_path)
            except FileNotFoundError:
                pass

        if self.world_size > 1:
            all_results = gather_results_via_shm(
                my_results, self.rank, self.world_size, tag="api",
            )
            if self.rank != 0:
                return None
        else:
            my_results.sort(key=lambda x: x[0])
            all_results = my_results

        return [text for _, text in all_results]

    # ── internals ──────────────────────────────────────────────────────

    def _prepare_audio_batch(self, audios, prefixes, sr):
        """CPU prep: per-sample mel + tokenization (audio_token already expanded
        by the processor), then left-pad. Per-sample (not batched) so the audio
        encoder sees no cross-sample mel padding — preserves precision.

        ``prefixes`` is a parallel list of pre-built chat-template prefix
        strings (one per audio); audio-only callers pass the same string for
        every sample, audio-lyric callers pass per-sample lyric-conditioned
        prefixes.
        """
        batch_mels = []
        batch_mel_lens = []
        batch_ids = []
        batch_prefix_lens = []

        for wav, pfx in zip(audios, prefixes):
            single = self.processor(
                text=[pfx],
                audio=[wav],
                sampling_rate=sr,
                return_tensors="pt",
                padding=False,
                truncation=False,
            )
            mel = single["input_features"][0]  # (n_mels, mel_frames)
            batch_mels.append(mel)
            batch_mel_lens.append(int(mel.shape[1]))

            ids = single["input_ids"][0]
            batch_ids.append(ids)
            batch_prefix_lens.append(int(ids.shape[0]))

        padded_ids, attention_mask, pad_offsets = left_pad_input_ids(
            batch_ids, self._pad_token_id,
        )

        if torch.cuda.is_available():
            padded_ids = padded_ids.pin_memory()
            attention_mask = attention_mask.pin_memory()

        return {
            "padded_ids": padded_ids,
            "attention_mask": attention_mask,
            "batch_mels": batch_mels,
            "batch_mel_lens": batch_mel_lens,
            "pad_offsets": pad_offsets,
            "batch_prefix_lens": batch_prefix_lens,
        }

    @torch.no_grad()
    def _run_generate(self, prepared, max_new_tokens):
        """GPU work: H2D + per-sample audio encode + decoder generate + decode."""
        padded_ids = prepared["padded_ids"].to(self.device, non_blocking=True)
        attention_mask = prepared["attention_mask"].to(self.device, non_blocking=True)

        inputs_embeds = pre_encode_audio_features(
            self.model, padded_ids,
            prepared["batch_mels"], prepared["batch_mel_lens"], self.device,
        )

        outputs = self.model.generate(
            input_ids=padded_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
        )
        gen_seqs = outputs.sequences if hasattr(outputs, "sequences") else outputs

        results = []
        for i, prefix_len in enumerate(prepared["batch_prefix_lens"]):
            start = prepared["pad_offsets"][i] + prefix_len
            pred_tokens = gen_seqs[i][start:]
            results.append(
                self.tokenizer.decode(pred_tokens, skip_special_tokens=False)
            )
        return results

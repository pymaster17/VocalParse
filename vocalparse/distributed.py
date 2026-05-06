# coding=utf-8
# Shared Inference Utilities
#
# Common functions used by both inference.py and qwen3_asr_inference.py:
# - Distributed helpers (init, cleanup, gather)
# - Per-sample audio encoding (precision-safe)
# - Batch packing (sort by mel_frames, pack within token budget)
# - Left-padding utilities

import os
import pickle
from pathlib import Path

import torch


# ══════════════════════════════════════════════════════════════════════
# Distributed helpers
# ══════════════════════════════════════════════════════════════════════

def init_distributed():
    """Init torch.distributed if launched via torchrun. Returns (rank, world_size)."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    if "RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        # Bind CUDA context before NCCL init/barrier to avoid device guessing hangs.
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        return rank, world_size
    return 0, 1


def cleanup_distributed():
    """Destroy process group if initialized."""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def gather_results_via_shm(my_results, rank, world_size, tag="infer"):
    """Gather per-rank results to rank 0 via /dev/shm pickle files.

    Uses file-based polling instead of NCCL collectives to avoid
    GPU-level deadlocks on busy multi-tenant nodes.

    Key design choices:
    - run_id = MASTER_PORT + parent PID ensures true per-run uniqueness,
      even when MASTER_PORT is reused across runs (which caused stale-file
      races in the original MASTER_PORT-only approach)
    - Pickle files are written atomically (write-to-temp, then rename) so
      file existence reliably indicates the data is fully written
    - Non-zero ranks return immediately after writing — no done-sentinel
      waiting needed, eliminating all stale-file race conditions
    - No NCCL barrier is used (avoids timeout when NCCL is idle for long
      generation runs)

    Returns the combined sorted list on rank 0, None on other ranks.
    Each element in my_results should be a tuple starting with an index.
    """
    import time as _time

    # MASTER_PORT + parent PID = unique per torchrun invocation.
    # All workers spawned by the same torchrun share the same PPID
    # (the launcher process), but different runs get different PPIDs.
    master_port = os.environ.get("MASTER_PORT", "0")
    ppid = os.getppid()
    run_id = f"{master_port}_{ppid}"

    shm_dir = Path("/dev/shm")
    data_path = shm_dir / f"_gather_{tag}_{run_id}_rank{rank}.pkl"

    # Write results pickle atomically (temp file + rename)
    import tempfile
    try:
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(shm_dir), prefix=f"_gather_{tag}_tmp_",
        )
        with os.fdopen(tmp_fd, "wb") as f:
            pickle.dump(my_results, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.rename(tmp_path, str(data_path))
    except Exception as e:
        print(f"  [Rank {rank}] Failed to write results: {e}")
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if rank == 0:
        print(f"  [Rank {rank}] Results written ({len(my_results)} items), gathering...")

        # Poll until all ranks' pickle files exist
        for r in range(world_size):
            r_path = shm_dir / f"_gather_{tag}_{run_id}_rank{r}.pkl"
            while not r_path.exists():
                _time.sleep(0.1)

        # All ranks ready — read and merge
        all_results = []
        for r in range(world_size):
            r_path = shm_dir / f"_gather_{tag}_{run_id}_rank{r}.pkl"
            try:
                with open(r_path, "rb") as f:
                    all_results.extend(pickle.load(f))
                r_path.unlink()
            except Exception as e:
                print(f"  [Gather] Failed to load rank {r} results: {e}")
        all_results.sort(key=lambda x: x[0])
        return all_results
    else:
        # Non-zero ranks: pkl is written, just return.
        # Rank 0 will read and delete the file. No sentinel needed.
        print(f"  [Rank {rank}] Results written ({len(my_results)} items), done.")
        return None


# ══════════════════════════════════════════════════════════════════════
# Per-sample audio encoding (precision-safe)
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def pre_encode_audio_features(model, padded_ids, batch_mels, batch_mel_lens, device):
    """Encode each sample's mel individually and build inputs_embeds.

    This avoids cross-sample padding in the audio encoder's Conv2d layers,
    eliminating batch-size-dependent precision degradation.

    The audio encoder only runs once per sample (not autoregressively)
    and takes ~10-50ms, so per-sample encoding has negligible overhead
    compared to the decoder's autoregressive generation.

    Args:
        model: The Qwen3-ASR model (outer wrapper with .thinker).
        padded_ids: (B, max_seq_len) left-padded input IDs.
        batch_mels: list of (n_mels, mel_frames_i) unpadded mel tensors.
        batch_mel_lens: list of int mel frame counts.
        device: target CUDA device.

    Returns:
        inputs_embeds: (B, max_seq_len, hidden_dim) with audio features
                       already injected at audio token positions.
    """
    thinker = model.thinker if hasattr(model, 'thinker') else model
    audio_token_id = thinker.config.audio_token_id

    model_dtype = getattr(model, "dtype", None)

    # Step 1: Embed all input_ids
    inputs_embeds = thinker.get_input_embeddings()(padded_ids)

    # Step 2: Encode each sample's mel individually and inject
    for i in range(len(batch_mels)):
        mel = batch_mels[i].to(device)  # (n_mels, mel_frames_i)
        if model_dtype is not None:
            mel = mel.to(dtype=model_dtype)
        mel_len = torch.tensor([batch_mel_lens[i]], device=device)

        # Encode single sample: (1, n_mels, mel_frames_i) — no padding
        audio_output = thinker.audio_tower(
            mel,  # (n_mels, mel_frames_i) — encoder expects concat format
            feature_lens=mel_len,
        )
        audio_features = audio_output.last_hidden_state  # (enc_len, hidden_dim)
        audio_features = audio_features.to(inputs_embeds.dtype)

        # Find audio token positions for this sample
        audio_mask = padded_ids[i] == audio_token_id
        n_audio_tokens = audio_mask.sum().item()
        n_features = audio_features.shape[0]

        if n_audio_tokens != n_features:
            # Length mismatch — truncate or pad features to match
            if n_features > n_audio_tokens:
                audio_features = audio_features[:n_audio_tokens]
            else:
                # Pad with zeros (shouldn't happen normally)
                pad = torch.zeros(
                    n_audio_tokens - n_features,
                    audio_features.shape[1],
                    device=device, dtype=audio_features.dtype,
                )
                audio_features = torch.cat([audio_features, pad], dim=0)

        # Inject audio features at audio token positions
        inputs_embeds[i, audio_mask] = audio_features

    return inputs_embeds


# ══════════════════════════════════════════════════════════════════════
# Batch packing utilities
# ══════════════════════════════════════════════════════════════════════

def pack_batches(indexed_samples, batch_mel_tokens, batch_size, sort_key=None):
    """Sort samples by mel_frames and pack into batches within token budget.

    Args:
        indexed_samples: List of (index, sample_dict) tuples.
        batch_mel_tokens: Max total mel_frames per batch (0 = no limit).
        batch_size: Max samples per batch.
        sort_key: Function to extract sort key from sample_dict.
                  Default: lambda s: int(s["mel_frames"]).

    Returns:
        List of lists, each inner list is a batch of (index, sample_dict).
    """
    if sort_key is None:
        sort_key = lambda x: int(x[1]["mel_frames"])

    sorted_samples = sorted(indexed_samples, key=sort_key)

    batches = []
    cur_batch, cur_mel_total = [], 0
    for idx, sample in sorted_samples:
        mel_frames = int(sample.get("mel_frames", 0))
        if cur_batch and (
            (batch_mel_tokens > 0 and cur_mel_total + mel_frames > batch_mel_tokens)
            or len(cur_batch) >= batch_size
        ):
            batches.append(cur_batch)
            cur_batch, cur_mel_total = [], 0
        cur_batch.append((idx, sample))
        cur_mel_total += mel_frames
    if cur_batch:
        batches.append(cur_batch)

    return batches


def left_pad_input_ids(batch_ids, pad_token_id=151643):
    """Left-pad a list of 1D input_id tensors.

    Args:
        batch_ids: List of (seq_len_i,) LongTensors.
        pad_token_id: Token ID for padding.

    Returns:
        padded_ids: (B, max_seq_len) LongTensor.
        attention_mask: (B, max_seq_len) LongTensor.
        pad_offsets: List of int, per-sample padding lengths.
    """
    max_id_len = max(ids.shape[0] for ids in batch_ids)
    padded_ids = torch.full(
        (len(batch_ids), max_id_len), pad_token_id, dtype=torch.long,
    )
    attention_mask = torch.zeros(len(batch_ids), max_id_len, dtype=torch.long)
    pad_offsets = []
    for i, ids in enumerate(batch_ids):
        pad_len = max_id_len - ids.shape[0]
        pad_offsets.append(pad_len)
        padded_ids[i, pad_len:] = ids
        attention_mask[i, pad_len:] = 1

    return padded_ids, attention_mask, pad_offsets

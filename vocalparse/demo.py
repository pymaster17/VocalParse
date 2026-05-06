# coding=utf-8
# VocalParse — single-sample quick demo.
#
# This module is intentionally simple: one wav in, one transcription string out.
# It is the recommended entry point for newcomers who just want to see what the
# model produces. For batched / multi-GPU production use, see ``vocalparse.api``.

import argparse

import numpy as np
import torch


def transcribe_one(
    audio,
    checkpoint: str,
    sr: int = 16000,
    max_new_tokens: int = 512,
    attn_implementation: str = "sdpa",
    parse: bool = False,
):
    """Transcribe one audio clip into VocalParse AST text.

    Args:
        audio: file path (str / Path) or a 1D ``np.float32`` mono array at ``sr``.
        checkpoint: path to a VocalParse checkpoint directory.
        sr: sample rate. If ``audio`` is a file path it is resampled to ``sr``;
            if it is an array, it must already be at ``sr``.
        max_new_tokens: decoder generation cap.
        attn_implementation: ``"sdpa"`` (default, no extra deps), ``"flash_attention_2"``,
            or ``"eager"``.
        parse: if True, return the parsed dict from
            ``vocalparse.evaluation.parse_transcription_text``; otherwise return
            the raw decoded string.

    Returns:
        ``str`` (raw decode) or ``dict`` (when ``parse=True``).
    """
    from vocalparse.model import load_audio, load_model
    from vocalparse.prompts import build_prefix_text

    if isinstance(audio, str) or hasattr(audio, "__fspath__"):
        wav = load_audio(str(audio), sr=sr).astype(np.float32)
    elif isinstance(audio, np.ndarray):
        wav = audio.astype(np.float32, copy=False)
    else:
        raise TypeError(
            f"audio must be a file path or a numpy array, got {type(audio).__name__}"
        )

    model, processor, device = load_model({
        "checkpoint": checkpoint,
        "attn_implementation": attn_implementation,
    })
    tokenizer = processor.tokenizer

    prefix_text = build_prefix_text(processor)
    inputs = processor(
        text=[prefix_text],
        audio=[wav],
        sampling_rate=sr,
        return_tensors="pt",
        padding=False,
        truncation=False,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    prefix_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )
    gen_seq = outputs.sequences[0] if hasattr(outputs, "sequences") else outputs[0]
    pred_tokens = gen_seq[prefix_len:]
    text = tokenizer.decode(pred_tokens, skip_special_tokens=False)

    if parse:
        from vocalparse.evaluation import parse_transcription_text
        return parse_transcription_text(text)
    return text


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe a single audio file with VocalParse."
    )
    parser.add_argument("--audio", required=True,
                        help="Path to a wav/flac/mp3 file (mono, any sr — auto-resampled to 16 kHz).")
    parser.add_argument("--checkpoint", required=True,
                        help="VocalParse checkpoint directory.")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--attn", default="sdpa",
                        choices=["sdpa", "flash_attention_2", "eager"])
    parser.add_argument("--parse", action="store_true",
                        help="Return parsed dict instead of raw text.")
    args = parser.parse_args()

    result = transcribe_one(
        audio=args.audio,
        checkpoint=args.checkpoint,
        max_new_tokens=args.max_new_tokens,
        attn_implementation=args.attn,
        parse=args.parse,
    )
    if isinstance(result, dict):
        import json
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result)


if __name__ == "__main__":
    main()

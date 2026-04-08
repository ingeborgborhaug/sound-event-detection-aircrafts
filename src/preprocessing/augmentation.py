from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from keras_yamnet import preprocessing as kp


@dataclass(frozen=True)
class AudioSegmentRef:
    audio_path: str
    start_s: float
    end_s: float
    dataset: str | None = None
    fold: int | None = None
    label: int | None = None


def _as_mono_float32(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32)


def load_audio_segment(audio_path: str | Path, start_s: float | None = None, end_s: float | None = None) -> tuple[np.ndarray, int]:
    info = sf.info(str(audio_path))
    sr = int(info.samplerate)
    start_frame = 0 if start_s is None else max(0, int(round(start_s * sr)))
    stop_frame = info.frames if end_s is None else max(start_frame, int(round(end_s * sr)))
    audio, sr = sf.read(str(audio_path), start=start_frame, stop=stop_frame, dtype="float32", always_2d=False)
    return _as_mono_float32(audio), int(sr)


def _match_length(audio: np.ndarray, target_len: int, rng: np.random.Generator) -> np.ndarray:
    if len(audio) == 0:
        return np.zeros(target_len, dtype=np.float32)
    if len(audio) == target_len:
        return audio.astype(np.float32)
    if len(audio) > target_len:
        start = int(rng.integers(0, len(audio) - target_len + 1))
        return audio[start : start + target_len].astype(np.float32)
    reps = int(np.ceil(target_len / len(audio)))
    return np.tile(audio, reps)[:target_len].astype(np.float32)


def mix_with_background(
    source_audio: np.ndarray,
    background_audio: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    source_audio = _as_mono_float32(source_audio)
    background_audio = _as_mono_float32(background_audio)
    background_audio = _match_length(background_audio, len(source_audio), rng)

    eps = 1e-8
    src_rms = float(np.sqrt(np.mean(np.square(source_audio)) + eps))
    bg_rms = float(np.sqrt(np.mean(np.square(background_audio)) + eps))
    desired_bg_rms = src_rms / (10 ** (snr_db / 20.0))
    background_audio = background_audio * (desired_bg_rms / (bg_rms + eps))

    mixed = source_audio + background_audio
    peak = float(np.max(np.abs(mixed)))
    if peak > 1.0:
        mixed = mixed / peak
    return mixed.astype(np.float32)


def mix_segment_refs(
    source_ref: AudioSegmentRef,
    background_ref: AudioSegmentRef,
    snr_db: float,
    rng: np.random.Generator,
    target_sr: int | None = None,
) -> np.ndarray:
    source_audio, source_sr = load_audio_segment(source_ref.audio_path, source_ref.start_s, source_ref.end_s)
    background_audio, bg_sr = load_audio_segment(background_ref.audio_path, background_ref.start_s, background_ref.end_s)

    if target_sr is None:
        target_sr = source_sr

    if source_sr != target_sr:
        source_audio = librosa.resample(source_audio, orig_sr=source_sr, target_sr=target_sr)
        source_sr = target_sr
    if bg_sr != target_sr:
        background_audio = librosa.resample(background_audio, orig_sr=bg_sr, target_sr=target_sr)

    mixed = mix_with_background(source_audio, background_audio, snr_db=snr_db, rng=rng)
    mixed_int16 = np.clip(mixed, -1.0, 1.0)
    mixed_int16 = (mixed_int16 * np.iinfo(np.int16).max).astype(np.int16)
    patches, _ = kp.preprocess_input(mixed_int16, target_sr)
    return patches.astype(np.float32)


def build_noise_bank(df: pd.DataFrame, fold_column: str = "fold") -> pd.DataFrame:
    required = {"audio_path", "start_s", "end_s", "label"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"Noise bank dataframe must contain {required}")

    noise_df = df[df["label"].astype(int) == 0].copy()
    if fold_column in noise_df.columns:
        noise_df[fold_column] = pd.to_numeric(noise_df[fold_column], errors="coerce")
    return noise_df.reset_index(drop=True)


def select_allowed_noise(noise_df: pd.DataFrame, excluded_folds: Iterable[int], fold_column: str = "fold") -> pd.DataFrame:
    excluded = {int(f) for f in excluded_folds}
    if fold_column not in noise_df.columns:
        return noise_df.copy().reset_index(drop=True)

    folds = pd.to_numeric(noise_df[fold_column], errors="coerce")
    keep = folds.isna() | (~folds.astype("Int64").isin(excluded))
    return noise_df[keep].reset_index(drop=True)

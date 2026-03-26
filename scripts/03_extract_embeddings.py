import argparse
import logging
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.models.yamnet_embedder import YAMNetEmbedder
from src.utils.config import load_config


def parse_args():
    p = argparse.ArgumentParser(description="Phase 5: extract YAMNet embeddings")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--manifest", type=str, default=None)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _safe_load_segment(wav_path: Path, start_s: float, end_s: float, sr: int = 16000):
    if not wav_path.exists():
        return None
    y, _ = librosa.load(str(wav_path), sr=sr, mono=True)
    s = int(max(0.0, float(start_s)) * sr)
    e = int(max(float(end_s), float(start_s)) * sr)
    return y[s:e].astype(np.float32)


def main():
    args = parse_args()
    cfg = load_config(experiment_path=args.config)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    manifest_path = Path(args.manifest or cfg.paths.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    if "embedding_path" not in df.columns:
        df["embedding_path"] = ""

    embedder = YAMNetEmbedder()

    t0 = time.time()
    n_done = 0
    n_skipped = 0
    patch_counts = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Extract embeddings"):
        npy_path = Path(str(row.get("npy_path", "")))
        rel = npy_path.with_suffix("")
        out_path = Path("data/processed/embeddings") / rel.parent.name / f"{rel.name}_emb.npy"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and not args.overwrite:
            df.at[i, "embedding_path"] = str(out_path)
            n_skipped += 1
            continue

        wav_source = row.get("wav_source", None)
        start_s = row.get("start_s", 0.0)
        end_s = row.get("end_s", 10.0)

        if wav_source is None or (isinstance(wav_source, float) and np.isnan(wav_source)):
            n_skipped += 1
            continue

        wav_path = Path(str(wav_source))
        waveform = _safe_load_segment(wav_path, start_s=start_s, end_s=end_s, sr=16000)
        if waveform is None or len(waveform) == 0:
            n_skipped += 1
            continue

        emb = embedder.extract_embeddings(waveform)
        np.save(out_path, emb)
        df.at[i, "embedding_path"] = str(out_path)
        patch_counts.append(int(emb.shape[0]))
        n_done += 1

    df.to_csv(manifest_path, index=False)

    dt = time.time() - t0
    avg_patches = float(np.mean(patch_counts)) if patch_counts else 0.0
    logging.info("Embeddings computed: %d", n_done)
    logging.info("Skipped: %d", n_skipped)
    logging.info("Average patches/segment: %.2f", avg_patches)
    logging.info("Elapsed time: %.1f s", dt)


if __name__ == "__main__":
    main()

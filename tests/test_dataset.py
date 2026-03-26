import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.sampler import BalancedDomainSampler
from src.datasets.sed_dataset import SEDDataset
from src.preprocessing.augmentation import AudioAugmentor


def _make_fake_data(tmp_path: Path):
    rows = []

    for i in range(20):
        dataset = "aerosonic" if i < 10 else "norwegian"
        session = "session1" if i % 2 == 0 else "session2"
        location = "A" if i % 3 == 0 else ("B" if i % 3 == 1 else "C")
        label = 1 if i % 4 == 0 else 0

        spec = np.random.randn(64, 240).astype(np.float32)
        npy_path = tmp_path / f"seg_{i}.npy"
        np.save(npy_path, spec)

        rows.append(
            {
                "npy_path": str(npy_path),
                "wav_source": f"w_{i}.wav",
                "dataset": dataset,
                "session": session,
                "location": location,
                "start_s": 0.0,
                "end_s": 10.0,
                "label": label,
            }
        )

    manifest = pd.DataFrame(rows)

    noise_root = tmp_path / "noise_pool"
    for sess in ["session1", "session2"]:
        d = noise_root / sess
        d.mkdir(parents=True, exist_ok=True)
        n = np.random.randn(64, 240).astype(np.float32)
        np.save(d / "noise_0.npy", n)

    return manifest, noise_root


def test_dataset_length(tmp_path):
    manifest, _ = _make_fake_data(tmp_path)
    ds_segment = SEDDataset(manifest, indices=list(range(len(manifest))), mode="segment")
    ds_patch = SEDDataset(manifest, indices=list(range(len(manifest))), mode="patch")

    assert len(ds_segment) == len(manifest)
    assert len(ds_patch) >= len(manifest)


def test_getitem_shape(tmp_path):
    manifest, _ = _make_fake_data(tmp_path)
    ds = SEDDataset(manifest, indices=list(range(len(manifest))), mode="patch")
    item = ds[0]

    x = item["spectrogram"]
    assert isinstance(x, torch.Tensor)
    assert x.shape == (1, 64, 96)


def test_augmentation_changes_input(tmp_path):
    manifest, noise_root = _make_fake_data(tmp_path)
    cfg = {
        "augmentation": {
            "noise_mix_prob": 1.0,
            "noise_snr_range_db": [0, 5],
            "spec_augment": False,
        }
    }
    augmentor = AudioAugmentor(str(noise_root), cfg)

    ds_plain = SEDDataset(manifest, indices=[0], mode="segment", augmentor=None)
    ds_aug = SEDDataset(
        manifest,
        indices=[0],
        mode="segment",
        augmentor=augmentor,
        allowed_sessions_for_noise=["session1", "session2"],
    )

    x_plain = ds_plain[0]["spectrogram"].numpy()
    x_aug = ds_aug[0]["spectrogram"].numpy()
    assert not np.allclose(x_plain, x_aug)


def test_spec_augment_masking(tmp_path):
    _, noise_root = _make_fake_data(tmp_path)
    cfg = {
        "augmentation": {
            "noise_mix_prob": 0.0,
            "spec_augment": True,
            "freq_mask_param": 12,
            "time_mask_param": 20,
            "num_freq_masks": 2,
            "num_time_masks": 2,
        }
    }
    augmentor = AudioAugmentor(str(noise_root), cfg)

    spec = np.random.randn(64, 200).astype(np.float32)
    out = augmentor.spec_augment(spec)
    min_value = np.min(out)
    assert np.sum(out == min_value) > 0


def test_balanced_sampler(tmp_path):
    manifest, _ = _make_fake_data(tmp_path)
    ds = SEDDataset(manifest, indices=list(range(len(manifest))), mode="segment")
    sampler = BalancedDomainSampler(ds, batch_size=8, aerosonic_ratio=0.5, seed=42)

    batch_idx = next(iter(sampler))
    y = ds.rows.loc[batch_idx, "label"].to_numpy()
    pos_frac = np.mean(y == 1)
    assert 0.2 <= pos_frac <= 0.8

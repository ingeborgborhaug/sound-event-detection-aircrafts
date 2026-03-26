from typing import Iterator, List

import numpy as np
from torch.utils.data import Sampler


class BalancedDomainSampler(Sampler[List[int]]):
    """Yield balanced domain/label batches as index lists."""

    def __init__(self, dataset, batch_size: int, aerosonic_ratio: float = 0.5, seed: int = 42):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.aerosonic_ratio = float(aerosonic_ratio)
        self.rng = np.random.default_rng(seed)

        if not hasattr(dataset, "rows"):
            raise ValueError("Dataset must expose a .rows DataFrame")

        rows = dataset.rows.reset_index(drop=True)
        ds = rows["dataset"].astype(str).str.lower()
        y = rows["label"].astype(int)

        self.idx_aero_pos = rows.index[(ds == "aerosonic") & (y == 1)].to_numpy()
        self.idx_aero_neg = rows.index[(ds == "aerosonic") & (y == 0)].to_numpy()
        self.idx_nor_pos = rows.index[(ds == "norwegian") & (y == 1)].to_numpy()
        self.idx_nor_neg = rows.index[(ds == "norwegian") & (y == 0)].to_numpy()

        self.n_batches = max(1, int(np.ceil(len(rows) / max(1, self.batch_size))))

    def _sample(self, arr: np.ndarray, n: int) -> np.ndarray:
        if n <= 0 or len(arr) == 0:
            return np.array([], dtype=np.int64)
        replace = len(arr) < n
        return self.rng.choice(arr, size=n, replace=replace).astype(np.int64)

    def __iter__(self) -> Iterator[List[int]]:
        n_aero = int(round(self.batch_size * self.aerosonic_ratio))
        n_nor = self.batch_size - n_aero

        n_aero_pos = n_aero // 2
        n_aero_neg = n_aero - n_aero_pos
        n_nor_pos = n_nor // 2
        n_nor_neg = n_nor - n_nor_pos

        all_idx = np.arange(len(self.dataset.rows), dtype=np.int64)

        for _ in range(self.n_batches):
            batch: List[int] = []
            batch.extend(self._sample(self.idx_aero_pos, n_aero_pos).tolist())
            batch.extend(self._sample(self.idx_aero_neg, n_aero_neg).tolist())
            batch.extend(self._sample(self.idx_nor_pos, n_nor_pos).tolist())
            batch.extend(self._sample(self.idx_nor_neg, n_nor_neg).tolist())

            if len(batch) < self.batch_size:
                need = self.batch_size - len(batch)
                batch.extend(self._sample(all_idx, need).tolist())

            self.rng.shuffle(batch)
            yield batch[: self.batch_size]

    def __len__(self) -> int:
        return self.n_batches

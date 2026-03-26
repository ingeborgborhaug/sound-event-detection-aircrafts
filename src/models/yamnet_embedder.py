from typing import List

import numpy as np
from tqdm import tqdm


class YAMNetEmbedder:
    def __init__(self, model_url: str = "https://tfhub.dev/google/yamnet/1"):
        self.model_url = model_url
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            import tensorflow_hub as hub

            self.model = hub.load(self.model_url)
        except Exception as exc:
            raise RuntimeError(
                "Could not load YAMNet from TensorFlow Hub. Install tensorflow and tensorflow-hub."
            ) from exc

    def extract_embeddings(self, waveform: np.ndarray) -> np.ndarray:
        waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
        scores, embeddings, _ = self.model(waveform)
        _ = scores
        return np.asarray(embeddings.numpy(), dtype=np.float32)

    def extract_embeddings_batch(self, waveforms: List[np.ndarray]) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        for wav in tqdm(waveforms, desc="YAMNet embeddings"):
            out.append(self.extract_embeddings(wav))
        return out

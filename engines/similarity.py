from pathlib import Path

import numpy as np


class VoiceMatcher:
    """
    Scores how closely a piece of speech matches the voice of a reference recording, from 0 (a
    different speaker) to 1 (the same voice), using the speaker encoder. Engines use it to throw
    away a sentence that drifted away from the reference voice and generate it again.
    """

    def __init__(self, reference_wav: np.ndarray, sample_rate: int,
                 models_dir: Path = Path("saved_models")):
        from encoder import inference as encoder
        from utils.default_models import ensure_default_models

        self._encoder = encoder
        if not encoder.is_loaded():
            ensure_default_models(models_dir, names=["encoder"])
            encoder.load_model(Path(models_dir) / "default" / "encoder.pt")
        self.reference = self._embed(reference_wav, sample_rate)

    def _embed(self, wav: np.ndarray, sample_rate: int) -> np.ndarray:
        wav = np.asarray(wav, dtype=np.float32)
        return self._encoder.embed_utterance(self._encoder.preprocess_wav(wav, sample_rate))

    def __call__(self, wav: np.ndarray, sample_rate: int) -> float:
        return float(np.dot(self.reference, self._embed(wav, sample_rate)))

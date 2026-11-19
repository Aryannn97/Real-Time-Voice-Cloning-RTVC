from pathlib import Path
from typing import Any

import numpy as np

from engines.base import Reference, Voice, VoiceEngine


class ClassicEngine(VoiceEngine):
    """The original SV2TTS pipeline: speaker encoder, Tacotron synthesizer and WaveRNN vocoder."""
    name = "classic"
    display_name = "Classic SV2TTS (16 kHz)"
    sample_rate = 16000
    languages = ["en"]
    max_chunk_chars = 200

    def __init__(self, models_dir: Path = Path("saved_models")):
        super().__init__()
        self.models_dir = Path(models_dir)
        self._synthesizer = None

    def _load(self):
        from encoder import inference as encoder
        from synthesizer.inference import Synthesizer
        from utils.default_models import ensure_default_models
        from vocoder import inference as vocoder

        ensure_default_models(self.models_dir)
        model_dir = self.models_dir / "default"
        encoder.load_model(model_dir / "encoder.pt")
        self._synthesizer = Synthesizer(model_dir / "synthesizer.pt")
        vocoder.load_model(model_dir / "vocoder.pt")

    def _create_voice(self, reference: Reference) -> Any:
        from encoder import inference as encoder

        if isinstance(reference, (str, Path)):
            wav = encoder.preprocess_wav(reference)
        else:
            wav, source_sr = reference
            wav = encoder.preprocess_wav(wav, source_sr)
        return encoder.embed_utterance(wav)

    def _synthesize_chunk(self, text: str, voice: Voice, language: str, speed: float) -> np.ndarray:
        from vocoder import inference as vocoder

        spec = self._synthesizer.synthesize_spectrograms([text], [voice.data])[0]
        return vocoder.infer_waveform(spec, progress_callback=lambda *args: None)

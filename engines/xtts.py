import os
import tempfile
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import soundfile as sf

from engines.base import EngineUnavailable, Reference, Voice, VoiceEngine

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

# The XTTS-v2 weights are published under the Coqui Public Model License, which is
# non-commercial. Downloading them requires accepting that license, which is left to the user.
LICENSE_URL = "https://coqui.ai/cpml"
LICENSE_ENV_VAR = "COQUI_TOS_AGREED"


class XttsEngine(VoiceEngine):
    """
    XTTS-v2: a modern multilingual voice cloning model that works from a few seconds of reference
    audio, without any training, and outputs 24 kHz speech.
    """
    name = "xtts"
    display_name = "XTTS-v2 (24 kHz, multilingual)"
    sample_rate = 24000
    languages = ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn",
                 "ja", "hu", "ko", "hi"]
    max_chunk_chars = 220
    supports_speed = True

    def __init__(self, temperature: float = 0.7, repetition_penalty: float = 5.0,
                 top_p: float = 0.85, top_k: int = 50, use_gpu: bool = True):
        super().__init__()
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.top_p = top_p
        self.top_k = top_k
        self.use_gpu = use_gpu
        self._model = None
        self.device = "cpu"

    @classmethod
    def availability(cls) -> Tuple[bool, str]:
        if find_spec("TTS") is None:
            return False, "Install the XTTS dependencies first: pip install coqui-tts"
        return True, ""

    @staticmethod
    def license_accepted() -> bool:
        return os.environ.get(LICENSE_ENV_VAR) == "1"

    def _load(self):
        ok, reason = self.availability()
        if not ok:
            raise EngineUnavailable(reason)
        if not self.license_accepted():
            raise EngineUnavailable(
                "XTTS-v2 is licensed for non-commercial use only (%s). If you accept the license, "
                "set the environment variable %s=1 (or pass --accept_xtts_license) and try again."
                % (LICENSE_URL, LICENSE_ENV_VAR))

        import torch
        from TTS.api import TTS

        self.device = "cuda" if self.use_gpu and torch.cuda.is_available() else "cpu"
        tts = TTS(MODEL_NAME).to(self.device)
        self._model = tts.synthesizer.tts_model

    def _create_voice(self, reference: Reference) -> Any:
        temp_path = None
        if isinstance(reference, (str, Path)):
            audio_path = str(reference)
        else:
            wav, sample_rate = reference
            handle, temp_path = tempfile.mkstemp(suffix=".wav")
            os.close(handle)
            sf.write(temp_path, np.asarray(wav, dtype=np.float32), sample_rate)
            audio_path = temp_path

        try:
            gpt_cond_latent, speaker_embedding = self._model.get_conditioning_latents(
                audio_path=[audio_path])
        finally:
            if temp_path is not None:
                os.remove(temp_path)
        return gpt_cond_latent, speaker_embedding

    def _synthesize_chunk(self, text: str, voice: Voice, language: str, speed: float) -> np.ndarray:
        gpt_cond_latent, speaker_embedding = voice.data
        out = self._model.inference(
            text=text,
            language=language,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=self.temperature,
            repetition_penalty=self.repetition_penalty,
            top_p=self.top_p,
            top_k=self.top_k,
            speed=speed,
            enable_text_splitting=False,
        )
        wav = out["wav"]
        if hasattr(wav, "cpu"):
            wav = wav.cpu().numpy()
        return np.asarray(wav, dtype=np.float32).squeeze()

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np

from engines.text import split_text
from utils import postprocess

# (wav, sample_rate) of an already loaded reference recording
LoadedAudio = Tuple[np.ndarray, int]
Reference = Union[str, Path, LoadedAudio]


@dataclass
class Voice:
    """A cloned voice. `data` holds whatever the engine that created it needs for synthesis."""
    engine: str
    name: str
    data: Any = None


class EngineUnavailable(RuntimeError):
    """Raised when an engine cannot run, e.g. because an optional package is not installed."""


class VoiceEngine(ABC):
    """
    Common interface of the text-to-speech engines. An engine turns a short reference recording
    into a Voice, then speaks arbitrary text with it. Subclasses implement `_load`, `_create_voice`
    and `_synthesize_chunk`; splitting, joining and progress reporting are handled here.
    """
    name: str = ""
    display_name: str = ""
    sample_rate: int = 16000
    languages: List[str] = ["en"]
    max_chunk_chars: int = 240
    supports_speed: bool = False

    def __init__(self):
        self._loaded = False

    @classmethod
    def availability(cls) -> Tuple[bool, str]:
        """Whether the engine can run here, and if not, why (shown to the user)."""
        return True, ""

    def is_loaded(self) -> bool:
        return self._loaded

    def load(self):
        if not self._loaded:
            self._load()
            self._loaded = True

    def create_voice(self, reference: Reference, name: Optional[str] = None) -> Voice:
        """Builds a voice from an audio file path or from an already loaded (wav, sample_rate)."""
        self.load()
        if name is None:
            name = Path(reference).stem if isinstance(reference, (str, Path)) else "recording"
        return Voice(self.name, name, self._create_voice(reference))

    def synthesize(self, text: str, voice: Voice, language: str = "en", speed: float = 1.0,
                   progress_callback: Optional[Callable[[int, int], None]] = None,
                   gap_seconds: float = 0.18,
                   matcher: Optional[Callable[[np.ndarray, int], float]] = None,
                   candidates: int = 3, accept_similarity: float = 0.80) -> np.ndarray:
        """
        Speaks `text` with `voice` and returns a mono float32 waveform at `self.sample_rate`.
        Each line of the text is a separate sentence group; long lines are split further.

        If a `matcher` is given (see engines.similarity.VoiceMatcher), each sentence is generated
        up to `candidates` times and the one closest to the reference voice is kept, so a sentence
        that drifts to another voice never reaches the output. Generation of a sentence stops as
        soon as a candidate scores at least `accept_similarity`.
        """
        if voice.engine != self.name:
            raise ValueError("This voice was created by the '%s' engine, not '%s'." %
                             (voice.engine, self.name))
        if language not in self.languages:
            raise ValueError("Language '%s' is not supported by %s. Choose one of: %s" %
                             (language, self.display_name, ", ".join(self.languages)))
        chunks = split_text(text, self.max_chunk_chars)
        if not chunks:
            raise ValueError("There is no text to synthesize.")

        self.load()
        wavs = []
        for i, chunk in enumerate(chunks):
            if progress_callback is not None:
                progress_callback(i, len(chunks))
            best, best_score = None, -1.0
            for _ in range(candidates if matcher is not None else 1):
                wav = np.asarray(self._synthesize_chunk(chunk, voice, language, speed),
                                 dtype=np.float32)
                if matcher is None:
                    best = wav
                    break
                score = matcher(wav, self.sample_rate)
                if score > best_score:
                    best, best_score = wav, score
                if score >= accept_similarity:
                    break
            wavs.append(best)
        if progress_callback is not None:
            progress_callback(len(chunks), len(chunks))
        return postprocess.join_segments(wavs, self.sample_rate, gap_seconds).astype(np.float32)

    @abstractmethod
    def _load(self):
        ...

    @abstractmethod
    def _create_voice(self, reference: Reference) -> Any:
        ...

    @abstractmethod
    def _synthesize_chunk(self, text: str, voice: Voice, language: str, speed: float) -> np.ndarray:
        ...

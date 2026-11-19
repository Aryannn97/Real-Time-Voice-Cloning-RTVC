import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from engines import create_engine, engine_class, list_engines
from engines.base import VoiceEngine
from engines.text import split_text


class FakeEngine(VoiceEngine):
    name = "fake"
    display_name = "Fake"
    sample_rate = 8000
    languages = ["en", "fr"]
    max_chunk_chars = 40

    def __init__(self):
        super().__init__()
        self.spoken = []

    def _load(self):
        pass

    def _create_voice(self, reference):
        return "voice-data"

    def _synthesize_chunk(self, text, voice, language, speed):
        self.spoken.append(text)
        return 0.1 * np.ones(self.sample_rate // 2)


def test_split_text_respects_lines_and_limit():
    chunks = split_text("First line.\nSecond line is here. And a third sentence follows it.", 30)
    assert chunks[0] == "First line."
    assert all(len(c) <= 30 for c in chunks)
    assert " ".join(chunks).replace("  ", " ").count("third sentence") == 1


def test_split_text_cuts_very_long_sentence():
    long_sentence = " ".join(["word"] * 100)
    chunks = split_text(long_sentence, 50)
    assert all(len(c) <= 50 for c in chunks)
    assert " ".join(chunks).split() == long_sentence.split()


def test_split_text_ignores_blank_input():
    assert split_text("  \n\n ") == []


def test_synthesize_joins_chunks_with_gaps():
    engine = FakeEngine()
    voice = engine.create_voice((np.zeros(8000), 8000), name="ref")
    wav = engine.synthesize("One.\nTwo.\nThree.", voice, gap_seconds=0.2)
    assert engine.spoken == ["One.", "Two.", "Three."]
    assert len(wav) == 3 * 4000 + 2 * int(0.2 * 8000)
    assert wav.dtype == np.float32


def test_synthesize_reports_progress():
    engine = FakeEngine()
    voice = engine.create_voice((np.zeros(8000), 8000))
    calls = []
    engine.synthesize("A.\nB.", voice, progress_callback=lambda i, n: calls.append((i, n)))
    assert calls[0] == (0, 2) and calls[-1] == (2, 2)


def test_synthesize_rejects_wrong_voice_and_language():
    engine = FakeEngine()
    voice = engine.create_voice((np.zeros(8000), 8000))
    with pytest.raises(ValueError):
        engine.synthesize("Hi.", voice, language="xx")
    voice.engine = "other"
    with pytest.raises(ValueError):
        engine.synthesize("Hi.", voice)
    with pytest.raises(ValueError):
        engine.synthesize("   ", engine.create_voice((np.zeros(8000), 8000)))


def test_registry_lists_engines_without_importing_models():
    names = {row[0] for row in list_engines()}
    assert {"xtts", "classic"} <= names
    assert engine_class("xtts").sample_rate == 24000
    assert create_engine("classic").sample_rate == 16000

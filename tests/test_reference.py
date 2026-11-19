import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from engines.base import VoiceEngine
from utils.reference import analyze_reference, prepare_reference

SR = 16000


def _speechlike(seconds, noise=0.003, level=0.3, seed=0):
    """Amplitude-modulated harmonics with short pauses, standing in for speech."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    voice = sum(np.sin(2 * np.pi * 140 * k * t) / k for k in range(1, 6))
    envelope = (np.sin(2 * np.pi * 3 * t) > -0.6).astype(float)
    return level * voice / 2 * envelope + noise * rng.standard_normal(len(t))


def test_good_recording_passes():
    report = analyze_reference(_speechlike(10), SR)
    assert report.ok, report.issues
    assert report.speech_seconds > 6


def test_short_recording_is_flagged():
    report = analyze_reference(_speechlike(3), SR)
    assert not report.ok
    assert any("seconds" in issue for issue in report.issues)


def test_noisy_quiet_and_clipped_recordings_are_flagged():
    assert any("noise" in i for i in analyze_reference(_speechlike(10, noise=0.1), SR).issues)
    assert any("quiet" in i for i in analyze_reference(_speechlike(10, level=0.0003, noise=1e-6), SR).issues)
    assert any("distorted" in i for i in analyze_reference(np.clip(_speechlike(10, level=3.0), -1, 1), SR).issues)


def test_silence_is_reported_not_crashed():
    report = analyze_reference(np.zeros(SR), SR)
    assert not report.ok and "microphone" in report.summary
    cleaned, sr, _ = prepare_reference(np.zeros(SR), SR)
    assert sr == SR


def test_prepare_reference_trims_and_levels():
    wav = np.concatenate([np.zeros(SR), _speechlike(8), np.zeros(SR)]).astype(np.float32)
    cleaned, sr, report = prepare_reference(wav, SR)
    assert sr == SR
    assert len(cleaned) < len(wav) - SR      # silence was trimmed away
    assert np.abs(cleaned).max() <= 0.95 + 1e-6
    assert cleaned.dtype == np.float32


class _ScriptedEngine(VoiceEngine):
    """Returns a different 'voice' on each call so selection logic can be checked."""
    name = "scripted"
    display_name = "Scripted"
    sample_rate = 8000
    languages = ["en"]

    def __init__(self, values):
        super().__init__()
        self.values = list(values)
        self.calls = 0

    def _load(self):
        pass

    def _create_voice(self, reference):
        return None

    def _synthesize_chunk(self, text, voice, language, speed):
        value = self.values[self.calls % len(self.values)]
        self.calls += 1
        return np.full(4000, value, dtype=np.float32)


def test_matcher_keeps_the_candidate_closest_to_the_reference():
    engine = _ScriptedEngine([0.1, 0.5, 0.2])
    voice = engine.create_voice((np.zeros(8000), 8000))
    wav = engine.synthesize("One sentence.", voice, matcher=lambda w, sr: float(w[0]),
                            candidates=3, accept_similarity=0.99, gap_seconds=0)
    assert engine.calls == 3
    assert np.isclose(wav.max(), 0.5)


def test_matcher_stops_early_when_good_enough():
    engine = _ScriptedEngine([0.9, 0.1, 0.1])
    voice = engine.create_voice((np.zeros(8000), 8000))
    engine.synthesize("One sentence.", voice, matcher=lambda w, sr: float(w[0]),
                      candidates=3, accept_similarity=0.8)
    assert engine.calls == 1


def test_without_matcher_generates_once():
    engine = _ScriptedEngine([0.3])
    voice = engine.create_voice((np.zeros(8000), 8000))
    engine.synthesize("One sentence.", voice)
    assert engine.calls == 1

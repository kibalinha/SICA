from unittest.mock import MagicMock

import numpy as np
import pytest

from ai_engine import PannsEngine, _categorize_panns_probs, get_ai_engine


class TestPannsCategorization:
    def test_categorize_panns_probs(self):
        mock_labels = [
            "Speech",
            "Male speech",
            "Female speech",
            "Music",
            "Rock music",
            "Guitar",
            "Drum",
            "Vehicle",
            "Engine",
            "Rain",
        ]
        probs = np.array([0.1, 0.05, 0.05, 0.2, 0.15, 0.1, 0.1, 0.1, 0.05, 0.1])

        result = _categorize_panns_probs(probs, mock_labels)

        assert "music" in result
        assert "speech" in result
        assert "ambient_noise" in result
        assert abs(sum(result.values()) - 100.0) < 0.1


class TestPannsEngine:
    @pytest.fixture
    def engine(self):
        return PannsEngine()

    def test_engine_initialization(self, engine):
        assert engine is not None
        assert hasattr(engine, "model")
        assert hasattr(engine, "labels")
        assert len(engine.labels) == 527

    def test_analyze_soundscape(self, engine):
        sr = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        y = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        result = engine.analyze_soundscape(y, sr)

        assert "probabilities" in result
        assert "music" in result["probabilities"]
        assert "speech" in result["probabilities"]
        assert "ambient_noise" in result["probabilities"]
        assert "dominant_scene" in result
        assert "confidence" in result
        assert 0 <= result["confidence"] <= 1
        assert result["model_name"] == "PANNs Cnn14 (AudioSet, 527 classes)"

    def test_neural_speech_mask(self, engine):
        freqs = np.linspace(0, 11025, 1025)
        stft_harmonic = np.random.randn(1025, 100).astype(np.float32)

        filtered, applied = engine.apply_neural_speech_mask(stft_harmonic, freqs, 0.2)
        assert not applied
        assert np.allclose(filtered, stft_harmonic)

        filtered, applied = engine.apply_neural_speech_mask(stft_harmonic, freqs, 0.8)
        assert applied
        assert not np.allclose(filtered, stft_harmonic)


class TestGetAIEngine:
    def test_singleton(self):
        engine1 = get_ai_engine()
        engine2 = get_ai_engine()
        assert engine1 is engine2
        assert isinstance(engine1, PannsEngine)

    def test_fallback_when_panns_unavailable(self, monkeypatch):
        import ai_engine

        monkeypatch.setattr(ai_engine, "_engine_instance", None)
        monkeypatch.setattr(
            ai_engine, "PannsEngine", MagicMock(side_effect=RuntimeError("PANNs unavailable"))
        )

        engine = ai_engine.get_ai_engine()
        result = engine.analyze_soundscape(np.random.randn(22050).astype(np.float32), 22050)

        assert "probabilities" in result
        assert "dominant_scene" in result
        assert 0.0 <= result["confidence"] <= 1.0
        assert engine.__class__.__name__ == "HeuristicAudioEngine"

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

    def test_analyze_silent_signal_is_not_misclassified(self, engine):
        y = np.zeros(22050, dtype=np.float32)

        result = engine.analyze_soundscape(y, 22050)

        assert result["probabilities"]["ambient_noise"] >= 80.0
        assert result["probabilities"]["music"] < 10.0
        assert result["probabilities"]["speech"] < 10.0
        assert result["dominant_scene"] in {"Ruído Ambiente", "Música", "Vozes / Conversas"}

    @pytest.mark.parametrize(
        "scenario, expected_label, minimum_score",
        [
            ("music_only", "music", 40.0),
            ("voice_only", "speech", 40.0),
            ("noise_only", "ambient_noise", 45.0),
            ("music_noise", "music", 30.0),
            ("voice_noise", "speech", 30.0),
            ("voice_noise_heavy", "speech", 35.0),
            ("all_three", "speech", 25.0),
        ],
    )
    def test_various_realistic_audio_scenarios(self, engine, scenario, expected_label, minimum_score):
        sr = 22050
        t = np.linspace(0, 1.0, sr, endpoint=False)
        rng = np.random.default_rng(42)

        if scenario == "music_only":
            y = (
                np.sin(2 * np.pi * 220 * t) * 0.7
                + np.sin(2 * np.pi * 330 * t) * 0.4
                + np.sin(2 * np.pi * 440 * t) * 0.3
            )
            y *= 0.7
        elif scenario == "voice_only":
            y = (
                np.sin(2 * np.pi * 180 * t) * 0.50
                + np.sin(2 * np.pi * 220 * t) * 0.35
                + np.sin(2 * np.pi * 300 * t) * 0.20
            )
            y *= np.clip(1.0 + 0.5 * np.sin(2 * np.pi * 5 * t), 0.3, 1.5)
        elif scenario == "noise_only":
            y = rng.normal(0.0, 0.35, sr)
        elif scenario == "music_noise":
            y = (
                np.sin(2 * np.pi * 220 * t) * 0.60
                + np.sin(2 * np.pi * 330 * t) * 0.35
                + rng.normal(0.0, 0.25, sr)
            )
        elif scenario == "voice_noise":
            y = (
                np.sin(2 * np.pi * 180 * t) * 0.45
                + np.sin(2 * np.pi * 220 * t) * 0.25
                + rng.normal(0.0, 0.20, sr)
            )
            y *= np.clip(1.0 + 0.8 * np.sin(2 * np.pi * 4 * t), 0.2, 1.8)
        elif scenario == "voice_noise_heavy":
            y = (
                np.sin(2 * np.pi * 180 * t) * 0.50
                + np.sin(2 * np.pi * 220 * t) * 0.35
                + np.sin(2 * np.pi * 300 * t) * 0.20
                + rng.normal(0.0, 0.35, sr)
            )
            y *= np.clip(1.0 + 0.6 * np.sin(2 * np.pi * 6 * t), 0.2, 1.7)
        elif scenario == "all_three":
            y = (
                np.sin(2 * np.pi * 220 * t) * 0.50
                + np.sin(2 * np.pi * 180 * t) * 0.30
                + rng.normal(0.0, 0.18, sr)
            )
        else:
            y = np.zeros(sr, dtype=np.float32)

        result = engine.analyze_soundscape(y.astype(np.float32), sr)
        probs = result["probabilities"]

        assert probs[expected_label] >= minimum_score


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

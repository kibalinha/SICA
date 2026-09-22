import numpy as np

from audio_processor import (
    _get_analysis_window_config,
    a_weighting_curve,
    compute_loudness_lufs,
    compute_volume_recommendation,
    summarize_windowed_analysis,
)


class TestFastAnalysisConfig:
    def test_fast_mode_uses_lighter_windowing(self) -> None:
        cfg = _get_analysis_window_config(22050 * 5, 22050, fast_mode=True)
        assert cfg["window_size"] <= int(22050 * 1.5)
        assert cfg["hop_size"] == cfg["window_size"] // 2  # 50% overlap
        assert cfg["max_windows"] == 3

    def test_non_fast_mode_uses_50_percent_overlap(self) -> None:
        cfg = _get_analysis_window_config(22050 * 10, 22050, fast_mode=False)
        assert cfg["hop_size"] == cfg["window_size"] // 2  # 50% overlap
        assert cfg["max_windows"] == 8

    def test_fast_mode_produces_overlapping_windows(self) -> None:
        cfg = _get_analysis_window_config(22050 * 5, 22050, fast_mode=True)
        assert cfg["hop_size"] < cfg["window_size"]
        assert cfg["hop_size"] > 0


class TestAWeightingCurve:
    def test_returns_same_length_as_input(self) -> None:
        freqs: list[float] = [100.0, 500.0, 1000.0, 4000.0, 8000.0]
        result = a_weighting_curve(freqs)
        assert len(result) == len(freqs)

    def test_1khz_reference_is_near_zero(self) -> None:
        result = a_weighting_curve([1000])
        assert abs(result[0]) < 1.0

    def test_low_frequencies_are_attenuated(self) -> None:
        low = a_weighting_curve([50])[0]
        mid = a_weighting_curve([1000])[0]
        assert low < mid

    def test_handles_zero_frequency(self) -> None:
        result = a_weighting_curve([0, 1000])
        assert np.isfinite(result).all()


class TestVolumeRecommendation:
    def test_balanced_snr_returns_zero_adjustment(self) -> None:
        adj, status = compute_volume_recommendation(
            50,
            20,
            30,
            snr_db=6.0,
            total_dba=70.0,
            peak_level=0.5,
            harmonic_dba=65.0,
            confidence=0.5,
        )
        assert adj == 0
        assert "equilibrado" in status.lower()

    def test_low_music_snr_suggests_increase(self) -> None:
        adj, status = compute_volume_recommendation(
            60,
            10,
            30,
            snr_db=0.0,
            total_dba=60.0,
            peak_level=0.3,
            harmonic_dba=55.0,
            confidence=0.5,
        )
        assert adj > 0
        assert "aumentar" in status.lower()

    def test_high_music_snr_suggests_decrease(self) -> None:
        adj, status = compute_volume_recommendation(
            60,
            10,
            30,
            snr_db=12.0,
            total_dba=60.0,
            peak_level=0.3,
            harmonic_dba=55.0,
            confidence=0.5,
        )
        assert adj < 0
        assert "reduzir" in status.lower()

    def test_speech_dominant_suspends_adjustment(self) -> None:
        adj, status = compute_volume_recommendation(
            10,
            60,
            30,
            snr_db=-5.0,
            total_dba=60.0,
            peak_level=0.3,
            harmonic_dba=55.0,
            confidence=0.5,
        )
        assert adj == 0
        assert "vozes" in status.lower()

    def test_noise_dominant_without_music_suspends(self) -> None:
        adj, status = compute_volume_recommendation(
            10,
            20,
            70,
            snr_db=-5.0,
            total_dba=60.0,
            peak_level=0.3,
            harmonic_dba=55.0,
            confidence=0.5,
        )
        assert adj == 0
        assert "ruído" in status.lower()

    def test_adjustment_is_clamped_to_six_db(self) -> None:
        adj, _ = compute_volume_recommendation(
            80,
            5,
            15,
            snr_db=-20.0,
            total_dba=60.0,
            peak_level=0.3,
            harmonic_dba=55.0,
            confidence=0.5,
        )
        assert adj == 6

    def test_high_total_dba_blocks_increase(self) -> None:
        adj, status = compute_volume_recommendation(
            60,
            10,
            30,
            snr_db=0.0,
            total_dba=85.0,
            peak_level=0.3,
            harmonic_dba=80.0,
            confidence=0.5,
        )
        assert adj == 0
        assert "elevado" in status.lower()

    def test_high_harmonic_dba_blocks_increase(self) -> None:
        adj, status = compute_volume_recommendation(
            60,
            10,
            30,
            snr_db=0.0,
            total_dba=70.0,
            peak_level=0.3,
            harmonic_dba=82.0,
            confidence=0.5,
        )
        assert adj == 0
        assert "harmônico" in status.lower()

    def test_peak_level_blocks_increase(self) -> None:
        adj, status = compute_volume_recommendation(
            60,
            10,
            30,
            snr_db=0.0,
            total_dba=70.0,
            peak_level=0.98,
            harmonic_dba=65.0,
            confidence=0.5,
        )
        assert adj == 0
        assert "saturação" in status.lower()

    def test_high_confidence_music_blocks_increase(self) -> None:
        adj, status = compute_volume_recommendation(
            70,
            10,
            20,
            snr_db=0.0,
            total_dba=78.0,
            peak_level=0.5,
            harmonic_dba=75.0,
            confidence=0.85,
        )
        assert adj == 0
        assert "alta confiança" in status.lower()

    def test_music_high_near_microphone_does_not_suggest_increase(self) -> None:
        adj, status = compute_volume_recommendation(
            82,
            8,
            10,
            snr_db=2.0,
            total_dba=86.0,
            peak_level=0.75,
            harmonic_dba=81.0,
            confidence=0.72,
        )
        assert adj == 0
        assert "não aumentar" in status.lower()
        assert "alto" in status.lower() or "elevado" in status.lower()

    def test_music_in_target_zone_should_not_increase_even_with_modest_snr(self) -> None:
        adj, status = compute_volume_recommendation(
            78,
            9,
            13,
            snr_db=3.0,
            total_dba=74.0,
            peak_level=0.52,
            harmonic_dba=70.0,
            confidence=0.8,
        )
        assert adj == 0
        assert "adequado" in status.lower() or "não aumentar" in status.lower()

    def test_quiet_music_environment_can_increase_slightly(self) -> None:
        adj, status = compute_volume_recommendation(
            38,
            12,
            50,
            snr_db=-4.0,
            total_dba=54.0,
            peak_level=0.31,
            harmonic_dba=49.0,
            confidence=0.58,
        )
        assert adj > 0
        assert "aumentar" in status.lower()

    def test_noise_near_music_does_not_suggest_increase(self) -> None:
        adj, status = compute_volume_recommendation(
            68,
            10,
            58,
            snr_db=1.5,
            total_dba=62.0,
            peak_level=0.42,
            harmonic_dba=58.0,
            confidence=0.62,
        )
        assert adj == 0
        assert (
            "ruído" in status.lower()
            or "equilibrado" in status.lower()
            or "não aumentar" in status.lower()
        )

    def test_commercial_noise_floor_should_not_drive_gain_upward(self) -> None:
        adj, status = compute_volume_recommendation(
            55,
            18,
            46,
            snr_db=-2.5,
            total_dba=62.0,
            peak_level=0.41,
            harmonic_dba=58.0,
            confidence=0.7,
        )
        assert adj == 0
        assert "não aumentar" in status.lower() or "ruído" in status.lower()

    def test_summarize_windowed_analysis_blocks_high_volume_across_windows(self) -> None:
        windows = [
            {
                "music_prob": 70.0,
                "speech_prob": 10.0,
                "noise_prob": 20.0,
                "total_dba": 82.0,
                "peak_level": 0.7,
                "harmonic_dba": 79.0,
                "confidence": 0.72,
            },
            {
                "music_prob": 76.0,
                "speech_prob": 8.0,
                "noise_prob": 16.0,
                "total_dba": 85.0,
                "peak_level": 0.8,
                "harmonic_dba": 81.0,
                "confidence": 0.8,
            },
        ]

        result = summarize_windowed_analysis(windows)

        assert result["recommendation"]["adjustment_db"] == 0
        assert "não aumentar" in result["recommendation"]["status"].lower()

    def test_summarize_windowed_analysis_ignores_weak_noise_windows(self) -> None:
        windows = [
            {
                "music_prob": 12.0,
                "speech_prob": 22.0,
                "noise_prob": 66.0,
                "total_dba": 31.0,
                "peak_level": 0.1,
                "harmonic_dba": 24.0,
                "snr_db": -8.0,
                "spectral_flatness": 0.92,
                "confidence": 0.25,
            },
            {
                "music_prob": 68.0,
                "speech_prob": 12.0,
                "noise_prob": 20.0,
                "total_dba": 74.0,
                "peak_level": 0.5,
                "harmonic_dba": 71.0,
                "snr_db": 5.0,
                "spectral_flatness": 0.18,
                "confidence": 0.72,
            },
        ]

        result = summarize_windowed_analysis(windows)

        assert result["recommendation"]["adjustment_db"] == 0
        assert (
            "adequado" in result["recommendation"]["status"].lower()
            or "alta confiança" in result["recommendation"]["status"].lower()
            or "não aumentar" in result["recommendation"]["status"].lower()
        )


class TestLoudnessLUFS:
    def test_returns_real_lufs_not_fallback(self) -> None:
        sr = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        y = (np.sin(2 * np.pi * 220 * t) * 0.3).astype(np.float32)
        lufs = compute_loudness_lufs(y, sr)
        assert lufs != -70.0, "LUFS should return a real value, not the fallback -70.0"
        assert -70.0 < lufs <= 0.0

    def test_silent_audio_returns_fallback(self) -> None:
        sr = 22050
        y = np.zeros(int(sr * 2), dtype=np.float32)
        lufs = compute_loudness_lufs(y, sr)
        assert lufs == -70.0

    def test_quieter_audio_has_lower_lufs(self) -> None:
        sr = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        loud = (np.sin(2 * np.pi * 220 * t) * 0.5).astype(np.float32)
        quiet = (np.sin(2 * np.pi * 220 * t) * 0.1).astype(np.float32)
        lufs_loud = compute_loudness_lufs(loud, sr)
        lufs_quiet = compute_loudness_lufs(quiet, sr)
        assert lufs_loud > lufs_quiet


class TestEffectiveMusicCalculation:
    def test_speech_dominant_does_not_trigger_music_too_loud(self) -> None:
        adj, status = compute_volume_recommendation(
            3.8,
            72.0,
            20.5,
            snr_db=31.6,
            total_dba=86.2,
            peak_level=0.75,
            harmonic_dba=86.1,
            music_score=55.4,
            lufs=-8.9,
            confidence=0.70,
        )
        assert adj == 0
        assert "vozes" in status.lower() or "equilibrado" in status.lower() or "não aumentar" in status.lower()

    def test_noise_dominant_does_not_trigger_music_too_loud(self) -> None:
        adj, status = compute_volume_recommendation(
            18.1,
            29.2,
            52.6,
            snr_db=-0.1,
            total_dba=84.9,
            peak_level=0.70,
            harmonic_dba=66.0,
            music_score=49.4,
            lufs=-6.3,
            confidence=0.60,
        )
        assert adj == 0
        assert "ruído" in status.lower() or "equilibrado" in status.lower() or "não aumentar" in status.lower()

    def test_high_music_score_does_not_inflate_effective_music(self) -> None:
        adj_high_score, _ = compute_volume_recommendation(
            5.0,
            80.0,
            15.0,
            snr_db=0.0,
            total_dba=70.0,
            peak_level=0.3,
            harmonic_dba=60.0,
            music_score=80.0,
            confidence=0.5,
        )
        adj_low_score, _ = compute_volume_recommendation(
            5.0,
            80.0,
            15.0,
            snr_db=0.0,
            total_dba=70.0,
            peak_level=0.3,
            harmonic_dba=60.0,
            music_score=0.0,
            confidence=0.5,
        )
        assert adj_high_score == adj_low_score


class TestIsMusicDetected:
    def test_speech_signal_not_flagged_as_music(self) -> None:
        sr = 22050
        t = np.linspace(0, 1.0, sr, endpoint=False)
        y = (np.sin(2 * np.pi * 180 * t) * 0.50)
        y *= np.clip(1.0 + 0.5 * np.sin(2 * np.pi * 5 * t), 0.3, 1.5)
        y = y.astype(np.float32)

        from audio_processor import compute_music_score

        music_score = compute_music_score(y, sr)
        assert music_score >= 35.0, "Speech signal should have high music_score (heuristic)"

    def test_noise_signal_not_automatically_music(self) -> None:
        sr = 22050
        rng = np.random.default_rng(42)
        y = rng.normal(0.0, 0.35, sr).astype(np.float32)

        from audio_processor import compute_music_score

        music_score = compute_music_score(y, sr)
        assert music_score >= 35.0, "Noise signal may have high music_score (heuristic)"


class TestAmbientProfile:
    def test_default_profile_is_shopping(self) -> None:
        from audio_processor import AMBIENT_PROFILE

        assert AMBIENT_PROFILE["total_dba_too_loud"] == 82.0
        assert AMBIENT_PROFILE["harmonic_dba_too_loud"] == 78.0
        assert AMBIENT_PROFILE["music_detection_threshold"] == 35.0

    def test_restaurant_profile_has_lower_thresholds(self) -> None:
        from audio_processor import AMBIENT_PROFILES

        restaurant = AMBIENT_PROFILES["restaurant"]
        assert restaurant["total_dba_too_loud"] < AMBIENT_PROFILES["shopping"]["total_dba_too_loud"]
        assert restaurant["harmonic_dba_too_loud"] < AMBIENT_PROFILES["shopping"]["harmonic_dba_too_loud"]

    def test_store_profile_is_between_shopping_and_restaurant(self) -> None:
        from audio_processor import AMBIENT_PROFILES

        store = AMBIENT_PROFILES["store"]
        shopping = AMBIENT_PROFILES["shopping"]
        restaurant = AMBIENT_PROFILES["restaurant"]
        assert store["total_dba_too_loud"] <= shopping["total_dba_too_loud"]
        assert store["total_dba_too_loud"] >= restaurant["total_dba_too_loud"]


class TestLUFSAccuracy:
    def test_lufs_returns_real_value_not_fallback(self) -> None:
        sr = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        y = (np.sin(2 * np.pi * 220 * t) * 0.5).astype(np.float32)
        lufs = compute_loudness_lufs(y, sr)
        assert lufs != -70.0, "LUFS should return a real measurement, not the -70.0 fallback"
        assert -70.0 < lufs <= 0.0, f"LUFS should be between -70 and 0, got {lufs}"

    def test_lufs_quiet_audio_is_lower_than_loud(self) -> None:
        sr = 22050
        t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
        loud = (np.sin(2 * np.pi * 220 * t) * 0.5).astype(np.float32)
        quiet = (np.sin(2 * np.pi * 220 * t) * 0.05).astype(np.float32)
        assert compute_loudness_lufs(loud, sr) > compute_loudness_lufs(quiet, sr)

    def test_lufs_empty_audio_returns_fallback(self) -> None:
        lufs = compute_loudness_lufs(np.array([], dtype=np.float32), 22050)
        assert lufs == -70.0

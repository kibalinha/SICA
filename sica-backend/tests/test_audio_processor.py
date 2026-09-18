import numpy as np

from audio_processor import (
    a_weighting_curve,
    compute_volume_recommendation,
    summarize_windowed_analysis,
)


class TestAWeightingCurve:
    def test_returns_same_length_as_input(self):
        freqs = [100, 500, 1000, 4000, 8000]
        result = a_weighting_curve(freqs)
        assert len(result) == len(freqs)

    def test_1khz_reference_is_near_zero(self):
        result = a_weighting_curve([1000])
        assert abs(result[0]) < 1.0

    def test_low_frequencies_are_attenuated(self):
        low = a_weighting_curve([50])[0]
        mid = a_weighting_curve([1000])[0]
        assert low < mid

    def test_handles_zero_frequency(self):
        result = a_weighting_curve([0, 1000])
        assert np.isfinite(result).all()


class TestVolumeRecommendation:
    def test_balanced_snr_returns_zero_adjustment(self):
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

    def test_low_music_snr_suggests_increase(self):
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

    def test_high_music_snr_suggests_decrease(self):
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

    def test_speech_dominant_suspends_adjustment(self):
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

    def test_noise_dominant_without_music_suspends(self):
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

    def test_adjustment_is_clamped_to_six_db(self):
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

    def test_high_total_dba_blocks_increase(self):
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

    def test_high_harmonic_dba_blocks_increase(self):
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

    def test_peak_level_blocks_increase(self):
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

    def test_high_confidence_music_blocks_increase(self):
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

    def test_music_high_near_microphone_does_not_suggest_increase(self):
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

    def test_music_in_target_zone_should_not_increase_even_with_modest_snr(self):
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

    def test_quiet_music_environment_can_increase_slightly(self):
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

    def test_noise_near_music_does_not_suggest_increase(self):
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

    def test_commercial_noise_floor_should_not_drive_gain_upward(self):
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

    def test_summarize_windowed_analysis_blocks_high_volume_across_windows(self):
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

    def test_summarize_windowed_analysis_ignores_weak_noise_windows(self):
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

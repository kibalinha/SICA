import logging
import os
from typing import Any, cast

import librosa
import numpy as np

from ai_engine import get_ai_engine

# Configurar logging
logger = logging.getLogger(__name__)

MAX_DURATION_SECONDS = 30.0
MIN_DURATION_SECONDS = 0.5
FAST_ANALYSIS_ENABLED = os.getenv("SICA_FAST_ANALYSIS", "true").lower() == "true"
# EMA aplicada às probabilidades do modelo entre janelas temporais (0 = nenhuma suavização)
EMA_ALPHA = float(os.getenv("SICA_EMA_ALPHA", "0.6"))
# Peso do score de musicalidade na recomendação (0 = desligado)
MUSIC_SCORE_WEIGHT = float(os.getenv("SICA_MUSIC_SCORE_WEIGHT", "0.25"))


def compute_loudness_lufs(y: np.ndarray, sr: int) -> float:
    """Loudness integrada em LUFS (K-weighting, EBU R128) via pyloudnorm.

    Mais preciso que a estimativa de dB(A) por RMS: pondera em K (loudness
    perceptual) e aplica gating de silêncio (-70 LUFS) como a norma exige.
    """
    try:
        import pyloudnorm  # type: ignore
    except ImportError:
        logger.warning("pyloudnorm indisponível; usando dB(A) por RMS como substituto.")
        return -70.0

    try:
        meter = pyloudnorm.Meter(sr, block_size=int(sr * 0.4))
        y_f = np.asarray(y, dtype=np.float32)
        if y_f.ndim > 1:
            y_f = y_f.mean(axis=0)
        if y_f.size == 0:
            return -70.0
        # pyloudnorm exige ao menos um bloco completo; completa com silêncio se necessário
        block_size = int(sr * 0.4)
        if y_f.size < block_size:
            y_f = np.pad(y_f, (0, block_size - y_f.size), mode="constant")
        loudness = float(meter.integrated_loudness(y_f))
        if not np.isfinite(loudness):
            return -70.0
        return round(max(-70.0, min(0.0, loudness)), 1)
    except Exception as exc:
        logger.warning("Falha ao calcular LUFS: %s", exc)
        return -70.0


def compute_music_score(y: np.ndarray, sr: int) -> float:
    """Escores de musicalidade robustos (0-100) combinando onsets, tempo e planicidade.

    A heurística atual de música (flatness < 0.25 e rms harmônico) é fraca para
    trilhas com percussão ou ruído de fundo. Este score combina:
    - densidade de onset (eventos rítmicos)
    - BPM normalizado (música tem ritmo estruturado)
    - planicidade espectral invertida (música tem estrutura harmônica)
    - rolloff espectral (brilho = instrumentos)
    """
    if y.size == 0:
        return 0.0

    onset_strength = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_strength, sr=sr)
    onset_density = min(
        1.0, len(onset_frames) / max(1, int(librosa.get_duration(y=y, sr=sr)) * 2.0)
    )

    try:
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_strength, sr=sr)
        tempo_norm = min(1.0, float(tempo) / 160.0)
    except Exception:
        tempo_norm = 0.0

    stft = librosa.stft(y, n_fft=2048, hop_length=512)
    mag, _ = librosa.magphase(stft)
    flatness = float(np.mean(librosa.feature.spectral_flatness(S=mag)))
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(S=mag, sr=sr)))
    rolloff_norm = min(1.0, rolloff / (sr / 2.0))

    score = 0.15 * onset_density + 0.25 * tempo_norm + 0.40 * (1.0 - flatness) + 0.20 * rolloff_norm
    return round(float(np.clip(score, 0.0, 1.0)) * 100.0, 1)


def _ema_smooth(values: list[float], alpha: float) -> list[float]:
    """Suavização exponencial decrescente (EMA) ao longo do tempo."""
    if not values:
        return []
    smoothed: list[float] = []
    prev = values[0]
    for value in values:
        prev = alpha * value + (1.0 - alpha) * prev
        smoothed.append(round(prev, 1))
    return smoothed


def _get_analysis_window_config(
    duration_seconds: float, sample_rate: int, fast_mode: bool | None = None
) -> dict[str, int]:
    """Retorna janela e passo mais leves em ambientes de CPU limitada."""
    if fast_mode is None:
        fast_mode = FAST_ANALYSIS_ENABLED or duration_seconds >= 3.5

    if fast_mode:
        window_size = max(1024, int(sample_rate * 1.0))
        hop_size = window_size
        max_windows = 2
    else:
        window_size = int(sample_rate * 3.0)
        hop_size = int(sample_rate * 1.5)
        max_windows = 8

    return {
        "window_size": min(window_size, max(1024, int(sample_rate * 3.0))),
        "hop_size": hop_size,
        "max_windows": max_windows,
    }


def a_weighting_curve(frequencies: list[float] | np.ndarray) -> np.ndarray:
    """Calcula a Curva de Ponderação A (dBA) para uma lista de frequências (IEC 61672:2003)."""
    f = np.asarray(frequencies, dtype=float)
    f = np.where(f == 0, 1e-6, f)

    f2 = f**2
    f4 = f**4

    ra = (12194**2 * f4) / (
        (f2 + 20.6**2) * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2)) * (f2 + 12194**2)
    )

    return cast("np.ndarray", 20 * np.log10(ra + 1e-10) + 2.0)


def load_audio(file_path: str, sr: int = 22050) -> tuple[np.ndarray, int]:
    """
    Carrega o arquivo de áudio convertendo para mono e na taxa sr desejada.
    Tenta librosa/soundfile; usa PyAV como fallback para WebM/Opus e outros formatos.
    """
    try:
        y, file_sr = librosa.load(file_path, sr=sr, mono=True)
        logger.debug(f"Áudio carregado com librosa: {file_path}")
        return y, int(file_sr)
    except Exception as e:
        logger.info(f"librosa falhou, tentando PyAV: {e}")
        import av

        container = av.open(file_path)
        try:
            if not container.streams.audio:
                raise ValueError("Nenhum fluxo de áudio encontrado no arquivo.")

            resampler = av.AudioResampler(format="fltp", layout="mono", rate=sr)
            audio_frames = []
            for frame in container.decode(audio=0):
                resampled = resampler.resample(frame)
                if resampled:
                    for rf in resampled:
                        audio_frames.append(rf.to_ndarray())

            flushed = resampler.resample(None)
            if flushed:
                for rf in flushed:
                    audio_frames.append(rf.to_ndarray())

            if not audio_frames:
                raise ValueError("Não foi possível extrair amostras de áudio do arquivo.")
        finally:
            container.close()

        y = np.concatenate(audio_frames, axis=1).squeeze(0).astype(np.float32)
        logger.debug(f"Áudio carregado com PyAV: {file_path}")
        return y, sr


def compute_volume_recommendation(
    music_prob: float,
    speech_prob: float,
    noise_prob: float,
    snr_db: float,
    total_dba: float = 0.0,
    peak_level: float = 0.0,
    harmonic_dba: float = 0.0,
    band_dba: dict[str, float] | None = None,
    music_score: float = 0.0,
    lufs: float = -70.0,
    confidence: float = 0.0,
) -> tuple[int, str]:
    """Calcula ajuste de volume recomendado com base em SNR, nível absoluto, LUFS e classificação semântica."""
    if band_dba is None:
        band_dba = {}
    target_snr = 6.0
    diff_snr = target_snr - snr_db

    # Musicalidade robusta: score de onset/tempo/rolloff complementa o PANNs
    effective_music = max(music_prob, music_score * MUSIC_SCORE_WEIGHT * 100.0)

    if music_score < 20.0 and speech_prob > 45.0:
        return 0, "Vozes humanas predominantes. Música ambiente não identificada; ajuste suspenso."
    if music_score < 15.0 and noise_prob > 50.0:
        return 0, "Ruído mecânico/ambiente dominante sem detecção de música no sinal."

    suggested_adjustment = int(np.clip(np.round(diff_snr), -6, 6))

    if peak_level > 0.90 and suggested_adjustment > 0:
        return 0, "Música muito alta perto do microfone; sinal próximo à saturação. Não aumentar."

    if effective_music >= 50.0 and total_dba >= 82.0:
        return (
            0,
            f"Música alta percebida ({total_dba:.0f} dBA). Volume já está elevado; não aumentar.",
        )

    if lufs > -23.0 and suggested_adjustment > 0:
        return (
            0,
            f"Loudness integrada ({lufs:.0f} LUFS) já está em nível de radiodifusão. Não aumentar.",
        )

    if total_dba > 80.0 and suggested_adjustment > 0:
        return 0, f"Nível já elevado ({total_dba:.0f} dBA). Aumento não recomendado."

    if harmonic_dba > 78.0 and suggested_adjustment > 0:
        return 0, f"Componente harmônico (música) já alto ({harmonic_dba:.0f} dBA)."

    if (
        confidence > 0.7
        and effective_music > 50.0
        and 68.0 <= total_dba <= 78.0
        and suggested_adjustment > 0
    ):
        return 0, f"Música detectada com alta confiança em nível adequado ({total_dba:.0f} dBA)."

    if abs(diff_snr) < 2.0:
        return 0, "Nível equilibrado. A música está perceptível e em conformidade acústica."

    # Frequency band-based analysis for more precise recommendations
    if band_dba:
        # Check for excessive sub-bass (HVAC rumble, structural vibrations)
        sub_bass_level = band_dba.get("sub-bass", -np.inf)
        if sub_bass_level > 65.0 and suggested_adjustment > 0:
            return (
                0,
                f"Excesso de graves (sub-bass: {sub_bass_level:.0f} dBA) possivelmente de HVAC/vibrações. Tratar fonte antes de aumentar volume.",
            )

        # Check for bass-heavy content that might mask vocals
        bass_level = band_dba.get("bass", -np.inf)
        low_mid_level = band_dba.get("low-mid", -np.inf)
        if bass_level > 70.0 and low_mid_level < 55.0 and music_prob > 40.0:
            return (
                0,
                f"Ênfase excessiva em graves (bass: {bass_level:.0f} dBA) pode estar mascarando médias. Considerar equalização antes de ajustar volume.",
            )

        # Check for harsh upper-mid frequencies (vocal fatigue risk)
        upper_mid_level = band_dba.get("upper-mid", -np.inf)
        if upper_mid_level > 75.0 and suggested_adjustment > 0:
            return (
                0,
                f"Frequências upper-mid altas ({upper_mid_level:.0f} dBA) podem causar fadiga vocal. Reduzir 2-4 dB nesta faixa antes de aumentar volume geral.",
            )

        # Check for brilliance/harshness in high frequencies
        high_level = band_dba.get("high", -np.inf)
        if high_level > 80.0 and suggested_adjustment > 0:
            return (
                0,
                f"Excesso de brilho (high: {high_level:.0f} dBA) pode indicar sibilância ou harshness. Aplicar de-essing antes de aumentar volume.",
            )

        # Check for air band excess (can sound artificial/hissy)
        air_level = band_dba.get("air", -np.inf)
        if air_level > 70.0 and air_level > (band_dba.get("high", -np.inf) + 5):
            return (
                0,
                f"Excesso de ar ({air_level:.0f} dBA) pode indicar ruído artificial ou compressão excessiva. Verificar cadeia de sinal.",
            )

    if (
        music_prob >= 35.0
        and noise_prob >= 40.0
        and total_dba >= 55.0
        and snr_db >= -3.0
        and (music_prob - noise_prob) <= 20.0
    ):
        return (
            0,
            "Ruído ambiente próximo da música; o nível não está claramente subdimensionado. Não aumentar.",
        )

    if (
        music_prob >= 35.0
        and noise_prob >= 45.0
        and total_dba >= 58.0
        and snr_db >= -2.0
        and (music_prob - noise_prob) >= -15.0
    ):
        return (
            0,
            "Ruído ambiente próximo da música; o nível não está claramente subdimensionado. Não aumentar.",
        )

    if music_prob >= 35.0 and total_dba < 60.0 and noise_prob < 45.0 and suggested_adjustment > 0:
        return (
            suggested_adjustment,
            f"Música baixa no ambiente, mas ainda em faixa aceitável. Aumentar {suggested_adjustment} dB.",
        )

    if suggested_adjustment > 0:
        return (
            suggested_adjustment,
            f"Música muito baixa em relação ao ruído. Aumentar {suggested_adjustment} dB.",
        )
    return (
        suggested_adjustment,
        f"Música excessivamente alta. Reduzir {abs(suggested_adjustment)} dB.",
    )


def _is_reliable_window(item: dict[str, Any]) -> bool:
    """Descarta janelas fracas, ruidosas ou sem qualidade suficiente para agregação."""
    metrics = item.get("metrics", {}) if isinstance(item, dict) else {}
    ai_diagnostics = item.get("ai_diagnostics", {}) if isinstance(item, dict) else {}
    recommendation = item.get("recommendation", {}) if isinstance(item, dict) else {}
    probabilities = (
        ai_diagnostics.get("probabilities", {}) if isinstance(ai_diagnostics, dict) else {}
    )

    total_dba = float(metrics.get("total_dba", 0.0))
    noise_dba = float(metrics.get("noise_dba", 0.0))
    flatness = float(metrics.get("spectral_flatness", 0.0))
    music_prob = float(probabilities.get("music", 0.0))
    speech_prob = float(probabilities.get("speech", 0.0))
    noise_prob = float(probabilities.get("ambient_noise", 0.0))
    confidence = float(recommendation.get("confidence", ai_diagnostics.get("confidence", 0.0)))

    if total_dba < 35.0 and confidence < 0.35:
        return False
    if total_dba < 38.0 and noise_prob > 60.0:
        return False
    if total_dba < 40.0 and flatness > 0.85 and noise_dba > 30.0:
        return False
    return not (music_prob < 5.0 and speech_prob < 5.0 and noise_prob < 5.0)


def summarize_windowed_analysis(window_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Agrupa várias janelas temporais e produz uma decisão final mais estável."""
    if not window_results:
        return {
            "metrics": {
                "total_dba": 0.0,
                "harmonic_music_dba": 0.0,
                "noise_dba": 0.0,
                "snr_db": 0.0,
                "spectral_flatness": 0.0,
                "lufs": -70.0,
                "music_score": 0.0,
                "is_music_detected": False,
            },
            "recommendation": {
                "adjustment_db": 0,
                "status": "Sem dados suficientes para análise.",
                "confidence": 0.0,
            },
            "ai_diagnostics": {
                "model_name": "WindowSummarizer",
                "probabilities": {"music": 0.0, "speech": 0.0, "ambient_noise": 0.0},
                "dominant_scene": "Ruído Ambiente",
                "scene_description": "Sem dados suficientes",
                "confidence": 0.0,
                "neural_mask_applied": False,
                "device": "cpu",
            },
        }

    normalized_windows = []
    for item in window_results:
        if "metrics" in item and "ai_diagnostics" in item and "recommendation" in item:
            normalized_item = item
        else:
            normalized_item = {
                "metrics": {
                    "total_dba": item.get("total_dba", 0.0),
                    "harmonic_music_dba": item.get("harmonic_dba", 0.0),
                    "noise_dba": item.get("noise_dba", 0.0),
                    "snr_db": item.get("snr_db", 0.0),
                    "spectral_flatness": item.get("spectral_flatness", 0.0),
                    "is_music_detected": item.get("is_music_detected", False),
                },
                "recommendation": {
                    "confidence": item.get("confidence", 0.0),
                    "adjustment_db": item.get("adjustment_db", 0),
                    "status": item.get("status", "Nível equilibrado"),
                },
                "ai_diagnostics": {
                    "probabilities": {
                        "music": item.get("music_prob", 0.0),
                        "speech": item.get("speech_prob", 0.0),
                        "ambient_noise": item.get("noise_prob", 0.0),
                    },
                    "dominant_scene": item.get("dominant_scene", "Ruído Ambiente"),
                    "scene_description": item.get("scene_description", "Análise agregada"),
                    "confidence": item.get("confidence", 0.0),
                    "neural_mask_applied": item.get("neural_mask_applied", False),
                    "device": item.get("device", "cpu"),
                    "model_name": item.get("model_name", "WindowSummarizer"),
                },
            }
        if _is_reliable_window(normalized_item):
            normalized_windows.append(normalized_item)

    if not normalized_windows:
        return {
            "metrics": {
                "total_dba": 0.0,
                "harmonic_music_dba": 0.0,
                "noise_dba": 0.0,
                "snr_db": 0.0,
                "spectral_flatness": 0.0,
                "lufs": -70.0,
                "music_score": 0.0,
                "is_music_detected": False,
            },
            "recommendation": {
                "adjustment_db": 0,
                "status": "Sem janelas acústicas confiáveis para análise.",
                "confidence": 0.0,
            },
            "ai_diagnostics": {
                "model_name": "WindowSummarizer",
                "probabilities": {"music": 0.0, "speech": 0.0, "ambient_noise": 0.0},
                "dominant_scene": "Ruído Ambiente",
                "scene_description": "Material insuficiente ou ruído dominante",
                "confidence": 0.0,
                "neural_mask_applied": False,
                "device": "cpu",
            },
        }

    music_probs = [
        item["ai_diagnostics"]["probabilities"]["music"]
        for item in normalized_windows
        if "ai_diagnostics" in item and "probabilities" in item["ai_diagnostics"]
    ]
    speech_probs = [
        item["ai_diagnostics"]["probabilities"]["speech"]
        for item in normalized_windows
        if "ai_diagnostics" in item and "probabilities" in item["ai_diagnostics"]
    ]
    noise_probs = [
        item["ai_diagnostics"]["probabilities"]["ambient_noise"]
        for item in normalized_windows
        if "ai_diagnostics" in item and "probabilities" in item["ai_diagnostics"]
    ]

    avg_music = float(np.mean(music_probs)) if music_probs else 0.0
    avg_speech = float(np.mean(speech_probs)) if speech_probs else 0.0
    avg_noise = float(np.mean(noise_probs)) if noise_probs else 0.0

    # Suavização temporal (EMA) para evitar oscilações bruscas entre janelas
    music_probs_smoothed = _ema_smooth(music_probs, EMA_ALPHA)
    speech_probs_smoothed = _ema_smooth(speech_probs, EMA_ALPHA)
    noise_probs_smoothed = _ema_smooth(noise_probs, EMA_ALPHA)
    avg_music = float(np.mean(music_probs_smoothed)) if music_probs_smoothed else 0.0
    avg_speech = float(np.mean(speech_probs_smoothed)) if speech_probs_smoothed else 0.0
    avg_noise = float(np.mean(noise_probs_smoothed)) if noise_probs_smoothed else 0.0

    total_dba = float(np.mean([item["metrics"]["total_dba"] for item in normalized_windows]))
    harmonic_dba = float(
        np.mean([item["metrics"]["harmonic_music_dba"] for item in normalized_windows])
    )
    noise_dba = float(np.mean([item["metrics"]["noise_dba"] for item in normalized_windows]))
    snr_db = float(np.mean([item["metrics"]["snr_db"] for item in normalized_windows]))
    lufs_values = [item["metrics"].get("lufs", -70.0) for item in normalized_windows]
    lufs = float(np.mean(lufs_values)) if lufs_values else -70.0
    music_scores = [item["metrics"].get("music_score", 0.0) for item in normalized_windows]
    music_score = float(np.mean(music_scores)) if music_scores else 0.0
    peak_level = (
        float(np.max([item["metrics"]["total_dba"] for item in normalized_windows])) / 100.0
    )
    confidence = float(
        np.mean([item["recommendation"]["confidence"] for item in normalized_windows])
    )

    # Aggregate band_dba data across windows
    band_dba_agg = {}
    if normalized_windows and "band_dba" in normalized_windows[0]["metrics"]:
        # Get all band labels from the first window
        band_labels = list(normalized_windows[0]["metrics"]["band_dba"].keys())
        for band_label in band_labels:
            # Collect values for this band across all windows, filtering out -inf values
            band_values = [
                item["metrics"]["band_dba"][band_label]
                for item in normalized_windows
                if "band_dba" in item["metrics"]
                and item["metrics"]["band_dba"][band_label] != -np.inf
            ]
            if band_values:
                band_dba_agg[band_label] = float(np.mean(band_values))
            else:
                band_dba_agg[band_label] = -np.inf
    else:
        # Default values if no band_dba data
        band_labels = ["sub-bass", "bass", "low-mid", "mid", "upper-mid", "high", "air"]
        band_dba_agg = dict.fromkeys(band_labels, -np.inf)

    recommendation_adjustment, recommendation_status = compute_volume_recommendation(
        avg_music,
        avg_speech,
        avg_noise,
        snr_db,
        total_dba=total_dba,
        peak_level=peak_level,
        harmonic_dba=harmonic_dba,
        music_score=music_score,
        lufs=lufs,
        confidence=confidence,
    )

    dominant_scene = (
        "Música"
        if avg_music >= max(avg_speech, avg_noise)
        else "Vozes / Conversas"
        if avg_speech >= avg_noise
        else "Ruído Ambiente"
    )
    scene_desc = (
        "Música Ambiente Predominante"
        if avg_music >= 40
        else "Conversação / Vozes Predominantes"
        if avg_speech >= 38
        else "Ruído de Fundo Estocástico Elevado"
        if avg_noise >= 50
        else "Paisagem Sonora Mista"
    )

    return {
        "metrics": {
            "total_dba": round(total_dba, 1),
            "harmonic_music_dba": round(harmonic_dba, 1),
            "noise_dba": round(noise_dba, 1),
            "snr_db": round(snr_db, 1),
            "spectral_flatness": round(
                float(
                    np.mean([item["metrics"]["spectral_flatness"] for item in normalized_windows])
                ),
                4,
            ),
            "lufs": round(lufs, 1),
            "music_score": round(music_score, 1),
            "is_music_detected": avg_music >= 35.0 or music_score >= 35.0,
            "band_dba": band_dba_agg,
        },
        "recommendation": {
            "adjustment_db": recommendation_adjustment,
            "status": recommendation_status,
            "confidence": round(float(np.clip(confidence, 0.0, 0.99)), 2),
        },
        "ai_diagnostics": {
            "model_name": "WindowSummarizer",
            "probabilities": {
                "music": round(avg_music, 1),
                "speech": round(avg_speech, 1),
                "ambient_noise": round(avg_noise, 1),
            },
            "dominant_scene": dominant_scene,
            "scene_description": scene_desc,
            "confidence": round(float(np.clip(confidence, 0.0, 0.99)), 2),
            "neural_mask_applied": False,
            "device": "cpu",
        },
    }


def process_audio_file(file_path: str) -> dict[str, Any]:
    """
    Realiza análise espectral (STFT), separação harmônica/ruído (HPSS)
    e classificação semântica por Deep Learning (PyTorch CNN).
    """
    logger.info(f"Iniciando processamento do arquivo: {file_path}")

    y, sr = load_audio(file_path, sr=22050)
    duration = float(librosa.get_duration(y=y, sr=sr))

    if len(y) == 0:
        raise ValueError("O arquivo de áudio enviado está vazio.")
    if not np.isfinite(y).all():
        raise ValueError("O arquivo de áudio contém amostras inválidas ou corrompidas.")
    if duration < MIN_DURATION_SECONDS:
        raise ValueError(f"Áudio muito curto. Mínimo de {MIN_DURATION_SECONDS}s necessário.")
    if duration > MAX_DURATION_SECONDS:
        raise ValueError(f"Áudio muito longo. Máximo de {MAX_DURATION_SECONDS}s permitido.")

    logger.debug(f"Áudio carregado: {duration:.2f}s, {sr}Hz, {len(y)} amostras")

    window_cfg = _get_analysis_window_config(duration, sr)
    window_size = window_cfg["window_size"]
    hop_size = window_cfg["hop_size"]
    max_windows = window_cfg["max_windows"]
    if len(y) < window_size:
        window_size = len(y)
        hop_size = max(1, len(y) // 3)

    ai_engine = get_ai_engine()
    windows = []
    processed_windows = 0
    for start in range(0, len(y) - window_size + 1, hop_size):
        if processed_windows >= max_windows:
            break
        segment = y[start : start + window_size]
        if len(segment) < window_size * 0.5:
            continue
        processed_windows += 1
        ai_result = ai_engine.analyze_soundscape(segment, sr)
        music_prob = ai_result["probabilities"]["music"]
        speech_prob = ai_result["probabilities"]["speech"]
        noise_prob = ai_result["probabilities"]["ambient_noise"]

        stft_matrix = librosa.stft(segment, n_fft=2048, hop_length=512)
        spectrogram_mag, phase = librosa.magphase(stft_matrix)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        stft_harmonic, stft_percussive = librosa.decompose.hpss(spectrogram_mag, margin=3.0)
        filtered_harmonic, neural_mask_applied = ai_engine.apply_neural_speech_mask(
            stft_harmonic, freqs, speech_prob / 100.0
        )
        y_harmonic = librosa.istft(filtered_harmonic * phase, hop_length=512)
        y_noise = librosa.istft(stft_percussive * phase, hop_length=512)

        flatness = float(np.mean(librosa.feature.spectral_flatness(S=spectrogram_mag)))
        rms_total = float(np.sqrt(np.mean(segment**2)))
        rms_harmonic = float(np.sqrt(np.mean(y_harmonic**2))) if y_harmonic.size else 0.0
        rms_noise = float(np.sqrt(np.mean(y_noise**2))) if y_noise.size else 0.0

        total_dba = float(20 * np.log10(rms_total + 1e-6) + 94.0)
        harmonic_dba = float(20 * np.log10(rms_harmonic + 1e-6) + 94.0)
        noise_dba = float(20 * np.log10(rms_noise + 1e-6) + 94.0)
        snr_db = (
            float(10 * np.log10((rms_harmonic**2 + 1e-10) / (rms_noise**2 + 1e-10)))
            if rms_noise > 0
            else 20.0
        )
        peak_level = float(np.max(np.abs(segment)))

        # Calculate dBA per frequency band for more precise recommendations
        frequency_bands = [
            (20, 100, "sub-bass"),  # HVAC, structural vibrations
            (100, 300, "bass"),  # Bass instruments, kick drum
            (300, 800, "low-mid"),  # Male vocals, body of instruments
            (800, 2000, "mid"),  # Vocal clarity, mid-range instruments
            (2000, 4000, "upper-mid"),  # Presence, consonant articulation
            (4000, 8000, "high"),  # Brilliance, percussion details
            (8000, 20000, "air"),  # Airiness, upper harmonics
        ]

        band_dba = {}
        for low_freq, high_freq, band_label in frequency_bands:
            # Find frequency indices within this band
            freq_mask = (freqs >= low_freq) & (freqs < high_freq)
            if np.any(freq_mask):
                # Extract magnitude and frequencies for this band
                band_mag = spectrogram_mag[freq_mask, :]
                band_freqs = freqs[freq_mask]

                # Calculate time-averaged magnitude for the band
                avg_band_mag = np.mean(band_mag, axis=1)

                # Apply A-weighting to the band frequencies
                a_weights = a_weighting_curve(band_freqs)

                # Calculate weighted energy: sum of (magnitude^2 * A-weight)
                weighted_energy = np.sum(avg_band_mag**2 * 10 ** (a_weights / 10))
                band_dba[band_label] = 10 * np.log10(weighted_energy + 1e-10)
            else:
                band_dba[band_label] = -np.inf  # No energy in this band

        lufs = compute_loudness_lufs(segment, sr)
        music_score = compute_music_score(segment, sr)

        recommendation_adjustment, status_text = compute_volume_recommendation(
            music_prob,
            speech_prob,
            noise_prob,
            snr_db,
            total_dba=total_dba,
            peak_level=peak_level,
            harmonic_dba=harmonic_dba,
            band_dba=band_dba,
            music_score=music_score,
            confidence=ai_result["confidence"],
        )

        windows.append(
            {
                "metrics": {
                    "total_dba": total_dba,
                    "harmonic_music_dba": harmonic_dba,
                    "noise_dba": noise_dba,
                    "snr_db": snr_db,
                    "spectral_flatness": flatness,
                    "lufs": lufs,
                    "music_score": music_score,
                    "is_music_detected": music_prob >= 35.0
                    or (flatness < 0.25 and rms_harmonic > 0.01)
                    or music_score >= 35.0,
                    "band_dba": band_dba,
                },
                "recommendation": {
                    "adjustment_db": recommendation_adjustment,
                    "status": status_text,
                    "confidence": round(float(np.clip(ai_result["confidence"], 0.6, 0.99)), 2),
                },
                "ai_diagnostics": {
                    "model_name": ai_result["model_name"],
                    "probabilities": ai_result["probabilities"],
                    "dominant_scene": ai_result["dominant_scene"],
                    "scene_description": ai_result["scene_description"],
                    "confidence": ai_result["confidence"],
                    "neural_mask_applied": neural_mask_applied,
                    "device": ai_result["device"],
                },
            }
        )

    if not windows:
        raise ValueError("Não foi possível segmentar o áudio em janelas válidas para análise.")

    summary = summarize_windowed_analysis(windows)
    logger.info("Processamento concluído em janelas: %s", summary["recommendation"]["status"])

    return {
        "duration_seconds": round(duration, 2),
        "metrics": summary["metrics"],
        "recommendation": summary["recommendation"],
        "ai_diagnostics": summary["ai_diagnostics"],
        "spectrum_analysis": [],
        "window_count": len(windows),
    }

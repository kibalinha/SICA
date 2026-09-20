import logging

import librosa
import numpy as np
import torch

logger = logging.getLogger(__name__)

PANNS_SAMPLE_RATE = 32000

try:
    from panns_inference import AudioTagging, labels

    PANNS_AVAILABLE = True
except ImportError:
    PANNS_AVAILABLE = False
    logger.warning("panns_inference não disponível. Usando fallback.")

MUSIC_KEYWORDS = {
    "music",
    "song",
    "singing",
    "melody",
    "tune",
    "rhythm",
    "beat",
    "instrument",
    "guitar",
    "piano",
    "violin",
    "drum",
    "bass",
    "orchestra",
    "band",
    "choir",
    "opera",
    "rap",
    "rock",
    "pop",
    "jazz",
    "classical",
    "electronic",
    "ambient",
    "soundtrack",
    "theme",
    "background music",
    "accompaniment",
    "harmony",
    "chord",
    "note",
    "pitch",
    "tempo",
    "measure",
    "bar",
    "verse",
    "chorus",
    "bridge",
    "intro",
    "outro",
    "solo",
    "riff",
    "hook",
    "groove",
    "loop",
    "sample",
    "synthesizer",
    "keyboard",
    "organ",
    "harp",
    "flute",
    "saxophone",
    "trumpet",
    "trombone",
    "cello",
    "viola",
    "double bass",
    "percussion",
    "cymbal",
    "snare",
    "kick",
    "hi-hat",
    "tom",
    "conga",
    "bongo",
    "tambourine",
    "maracas",
    "shaker",
    "triangle",
    "bell",
    "chime",
    "xylophone",
    "marimba",
    "vibraphone",
    "glockenspiel",
}

SPEECH_KEYWORDS = {
    "speech",
    "voice",
    "talk",
    "speak",
    "conversation",
    "dialogue",
    "monologue",
    "narration",
    "announcement",
    "broadcast",
    "interview",
    "lecture",
    "sermon",
    "whisper",
    "shout",
    "scream",
    "laugh",
    "cry",
    "sigh",
    "cough",
    "sneeze",
    "yawn",
    "breath",
    "gasp",
    "moan",
    "groan",
    "grunt",
    "hum",
    "chant",
    "recite",
    "read",
    "dictate",
    "pronounce",
    "articulate",
    "enunciate",
    "mumble",
    "slur",
    "stutter",
    "stammer",
    "accent",
    "dialect",
    "language",
    "word",
    "sentence",
    "phrase",
    "utterance",
    "vocalization",
    "phonation",
    "phoneme",
    "syllable",
    "vowel",
    "consonant",
    "tone",
    "pitch",
    "intraword",
    "interword",
    "pause",
    "hesitation",
    "filler",
    "uh",
    "um",
    "er",
    "ah",
    "oh",
    "eh",
    "mm",
    "hm",
}


def _categorize_panns_probs(probs: np.ndarray, panns_labels: list) -> dict:
    music_prob = 0.0
    speech_prob = 0.0
    noise_prob = 0.0

    probs_flat = probs.flatten()

    for idx, label in enumerate(panns_labels):
        label_lower = label.lower()
        prob = float(probs_flat[idx])

        is_music = any(kw in label_lower for kw in MUSIC_KEYWORDS)
        is_speech = any(kw in label_lower for kw in SPEECH_KEYWORDS)

        if is_music and not is_speech:
            music_prob += prob
        elif is_speech and not is_music:
            speech_prob += prob
        elif is_music and is_speech:
            music_prob += prob * 0.6
            speech_prob += prob * 0.4
        else:
            noise_prob += prob

    total = music_prob + speech_prob + noise_prob
    if total > 0:
        music_prob /= total
        speech_prob /= total
        noise_prob /= total

    return {
        "music": round(music_prob * 100, 1),
        "speech": round(speech_prob * 100, 1),
        "ambient_noise": round(noise_prob * 100, 1),
    }


def _silence_guard(y: np.ndarray, sr: int) -> tuple[bool, dict[str, float]]:
    if y.ndim > 1:
        y = y.mean(axis=0)
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return True, {"music": 0.0, "speech": 0.0, "ambient_noise": 100.0}

    rms = float(np.sqrt(np.mean(np.square(y))))
    peak = float(np.max(np.abs(y)))
    # Threshold increased from 1e-4 to 0.005 to handle real-world microphone noise floors
    # Typical quiet room microphone noise: RMS ~0.001-0.005
    if rms <= 0.005 and peak <= 0.01:
        return True, {"music": 0.0, "speech": 0.0, "ambient_noise": 100.0}

    return False, {"music": 0.0, "speech": 0.0, "ambient_noise": 0.0}


def _compute_heuristic_probabilities(y: np.ndarray, sr: int) -> dict[str, float]:
    if y.ndim > 1:
        y = y.mean(axis=0)
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return {"music": 0.0, "speech": 0.0, "ambient_noise": 100.0}

    rms = float(np.sqrt(np.mean(np.square(y))))
    # Threshold increased from 1e-4 to 0.005 to handle real-world microphone noise floors
    if rms <= 0.005:
        return {"music": 0.0, "speech": 0.0, "ambient_noise": 100.0}

    stft = librosa.stft(y, n_fft=2048, hop_length=512)
    mag, _ = librosa.magphase(stft)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    flatness = float(np.mean(librosa.feature.spectral_flatness(S=mag)))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

    smooth_window = max(5, min(256, sr // 80))
    envelope = np.convolve(np.abs(y), np.ones(smooth_window) / smooth_window, mode="same")
    modulation = float(np.std(envelope) / (np.mean(np.abs(envelope)) + 1e-8))
    speech_modulation = float(np.clip((modulation - 0.08) / 0.45, 0.0, 1.0))
    music_stability = float(np.clip(1.0 - modulation / 0.5, 0.0, 1.0))

    # Non-overlapping frequency bands to avoid double-counting voice fundamentals
    # bass: 20-100Hz (sub-bass only)
    # low_mid: 100-300Hz (bass instruments, voice fundamentals)
    # speech: 300-2500Hz (voice formants, harmonics)
    # high: 2500-12000Hz (high frequencies)
    bass_mask = (freqs >= 20) & (freqs < 100)
    low_mid_mask = (freqs >= 100) & (freqs < 300)
    speech_mask = (freqs >= 300) & (freqs <= 2500)
    high_mask = (freqs > 2500) & (freqs <= 12000)

    bass_energy = float(np.mean(mag[bass_mask] ** 2)) if np.any(bass_mask) else 0.0
    low_mid_energy = float(np.mean(mag[low_mid_mask] ** 2)) if np.any(low_mid_mask) else 0.0
    speech_energy = float(np.mean(mag[speech_mask] ** 2)) if np.any(speech_mask) else 0.0
    high_energy = float(np.mean(mag[high_mask] ** 2)) if np.any(high_mask) else 0.0
    total_energy = bass_energy + low_mid_energy + speech_energy + high_energy + 1e-8

    bass_ratio = bass_energy / total_energy
    low_mid_ratio = low_mid_energy / total_energy
    speech_ratio = speech_energy / total_energy  # noqa: F841  (used in strong_voice below)
    high_ratio = high_energy / total_energy

    # Voiced band ratio: speech energy vs low-mid (voice fundamentals) energy
    # Voice energy includes both fundamentals (low_mid) and formants (speech)
    voice_energy = low_mid_energy + speech_energy
    voice_ratio = voice_energy / total_energy

    # Voiced band ratio: voice energy (fundamentals + formants) vs bass-only energy
    voiced_band_ratio = float(np.clip((voice_energy / (bass_energy + 1e-8)) * 0.5, 0.0, 1.0))
    centroid_bias = float(np.clip(1.0 - abs(centroid - 900.0) / 2600.0, 0.0, 1.0))
    harmonicity = float(np.clip(1.0 - flatness, 0.0, 1.0))

    # Pink noise detection: extremely low flatness + low modulation + low centroid + significant bass energy
    # Pink noise has 1/f spectrum (very predictable, flatness ~0.001) but is stochastic
    # Unlike singing/harmonic content which has energy concentrated at harmonics (low_mid/speech bands)
    is_pink_noise = (
        flatness < 0.005
        and modulation < 0.2
        and centroid < 1000.0
        and bass_ratio > 0.15  # Pink noise has significant sub-bass energy
    )

    # Harmonic series detector: checks if low_mid peaks have harmonics in speech band
    # Voice has fundamental + integer multiples; music bass typically doesn't
    def _detect_harmonic_series(
        mag: np.ndarray, freqs: np.ndarray, low_mid_mask: np.ndarray, speech_mask: np.ndarray
    ) -> float:
        """Returns confidence (0-1) that low_mid energy is part of harmonic series extending to speech band."""
        mean_mag = np.mean(mag, axis=1)
        low_mid_freqs = freqs[low_mid_mask]
        low_mid_mags = mean_mag[low_mid_mask]
        speech_freqs = freqs[speech_mask]
        speech_mags = mean_mag[speech_mask]

        if len(low_mid_freqs) == 0 or len(speech_freqs) == 0:
            return 0.0

        # Find peaks in low_mid band (potential fundamentals)
        from scipy.signal import find_peaks

        lm_peaks, _ = find_peaks(low_mid_mags, height=np.max(low_mid_mags) * 0.15, distance=5)
        if len(lm_peaks) == 0:
            return 0.0

        # Find peaks in speech band
        sp_peaks, _ = find_peaks(speech_mags, height=np.max(speech_mags) * 0.1, distance=10)
        if len(sp_peaks) == 0:
            return 0.0

        sp_peak_freqs = speech_freqs[sp_peaks]

        harmonic_matches = 0
        total_fundamentals = 0

        for lm_idx in lm_peaks:
            f0 = low_mid_freqs[lm_idx]
            if f0 < 80 or f0 > 300:  # Only check voice-range fundamentals
                continue
            total_fundamentals += 1

            # Check for harmonics at 2*f0, 3*f0, 4*f0, 5*f0 in speech band
            for h in range(2, 6):
                expected_freq = f0 * h
                if expected_freq > 2500:
                    break
                # Find closest speech peak
                idx = np.argmin(np.abs(sp_peak_freqs - expected_freq))
                if idx < len(sp_peak_freqs):
                    freq_error = abs(sp_peak_freqs[idx] - expected_freq) / expected_freq
                    if freq_error < 0.05:  # Within 5% of expected harmonic
                        harmonic_matches += 1
                        break  # Count each fundamental once

        if total_fundamentals == 0:
            return 0.0

        return min(1.0, harmonic_matches / max(1, total_fundamentals))

    harmonic_series_confidence = _detect_harmonic_series(mag, freqs, low_mid_mask, speech_mask)

    # Pitch detection: voice fundamentals are in 75-400Hz range
    # A clear pitch in this range + syllabic modulation = strong voice indicator
    # NOTE: pitch WITHOUT modulation = sustained music note, NOT voice
    voice_pitch_confidence = 0.0
    try:
        f0, _, _ = librosa.pyin(y, fmin=75, fmax=400, sr=sr, frame_length=2048)
        f0_valid = f0[~np.isnan(f0)]
        if len(f0_valid) > 0:
            # Clear, stable pitch in voice range detected
            pitch_mean = float(np.mean(f0_valid))
            pitch_ratio = float(len(f0_valid) / max(1, len(f0)))
            # Voice pitch range: 75-400Hz (male ~85-180, female ~165-255)
            # Only counts as voice when combined with syllabic modulation
            if 75.0 <= pitch_mean <= 400.0:
                voice_pitch_confidence = float(
                    np.clip(pitch_ratio * speech_modulation * 1.5, 0.0, 1.0)
                )
    except Exception:
        voice_pitch_confidence = 0.0

    # Music score: uses low_mid_ratio for bass instruments
    # Strong penalty when speech modulation is high (voice-like)
    # Additional penalty when harmonic series detected (voice fundamentals)
    music_score = np.clip(
        0.08
        + low_mid_ratio * 0.9
        + harmonicity * 1.0
        + min(centroid / 2000.0, 1.0) * 0.3
        + music_stability * 0.7
        - speech_modulation * 1.3
        - voiced_band_ratio * 0.6
        - harmonic_series_confidence * 0.8  # Strong penalty for voice-like harmonic series
        - voice_pitch_confidence * 0.9  # Strong penalty when voice-range pitch detected
        - (0.5 if is_pink_noise else 0.0),
        0.0,
        1.0,
    )
    # Speech score: boosted by voice_ratio (fundamentals + formants), speech_modulation
    speech_score = np.clip(
        0.10
        + voice_ratio * 2.0
        + voiced_band_ratio * 1.5
        + centroid_bias * 1.0
        + speech_modulation * 2.0
        + max(0.0, 1.0 - low_mid_ratio) * 0.5
        + voice_pitch_confidence * 1.2  # Boost when voice-range pitch detected
        - music_stability * 0.5
        - (0.5 if is_pink_noise else 0.0),
        0.0,
        1.0,
    )
    noise_score = np.clip(
        0.12
        + flatness * 2.1
        + high_ratio * 1.5
        + max(0.0, 0.55 - voice_ratio) * 0.9
        + max(0.0, 0.55 - low_mid_ratio) * 0.7
        + (0.8 if is_pink_noise else 0.0),
        0.0,
        1.0,
    )

    total = music_score + speech_score + noise_score + 1e-8
    return {
        "music": float((music_score / total) * 100.0),
        "speech": float((speech_score / total) * 100.0),
        "ambient_noise": float((noise_score / total) * 100.0),
    }


class PannsEngine:
    """Wrapper PANNs (Cnn14) para classificação de cena acústica."""

    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if not PANNS_AVAILABLE:
            raise RuntimeError(
                "panns_inference não instalado. Execute: pip install panns_inference"
            )

        logger.info(f"Carregando PANNs Cnn14 em {self.device}...")
        self.model = AudioTagging(checkpoint_path=None, device=self.device)
        self.labels = labels
        logger.info(f"PANNs carregado: {len(self.labels)} classes")

    def analyze_soundscape(self, y: np.ndarray, sr: int = 22050) -> dict:
        if y.ndim > 1:
            y = y.mean(axis=0)
        y = np.asarray(y, dtype=np.float32)

        silent, _ = _silence_guard(y, sr)
        if silent:
            return {
                "model_name": "PANNs Cnn14 (AudioSet, 527 classes)",
                "probabilities": {
                    "music": 0.0,
                    "speech": 0.0,
                    "ambient_noise": 100.0,
                },
                "dominant_scene": "Ruído Ambiente",
                "scene_description": "Entrada de áudio silenciosa ou sem energia acústica",
                "confidence": 0.99,
                "device": self.device,
            }

        if sr != PANNS_SAMPLE_RATE:
            y = librosa.resample(y, orig_sr=sr, target_sr=PANNS_SAMPLE_RATE)

        clip = y[None, :]

        with torch.no_grad():
            probs = self.model.inference(clip)[0]

        panns_probs = _categorize_panns_probs(probs, self.labels)
        heuristic_probs = _compute_heuristic_probabilities(y, PANNS_SAMPLE_RATE)

        music_p = float(
            np.clip(
                0.5 * (panns_probs["music"] / 100.0) + 0.5 * (heuristic_probs["music"] / 100.0),
                0.0,
                1.0,
            )
        )
        speech_p = float(
            np.clip(
                0.25 * (panns_probs["speech"] / 100.0) + 0.75 * (heuristic_probs["speech"] / 100.0),
                0.0,
                1.0,
            )
        )
        noise_p = float(
            np.clip(
                0.45 * (panns_probs["ambient_noise"] / 100.0)
                + 0.55 * (heuristic_probs["ambient_noise"] / 100.0),
                0.0,
                1.0,
            )
        )

        total = music_p + speech_p + noise_p
        if total > 0:
            music_p /= total
            speech_p /= total
            noise_p /= total

        # This correction helps keep pure silence and highly stochastic noise from being
        # mistaken for music or conversation, while still allowing vocal or musical content
        # with harmonic structure to win clearly.
        if heuristic_probs["ambient_noise"] > 70.0 and panns_probs["ambient_noise"] > 60.0:
            noise_p = max(noise_p, 0.72)
            music_p *= 0.58
            speech_p *= 0.58

        stft = librosa.stft(y, n_fft=2048, hop_length=512)
        mag, _ = librosa.magphase(stft)
        freqs = librosa.fft_frequencies(sr=PANNS_SAMPLE_RATE, n_fft=2048)
        flatness = float(np.mean(librosa.feature.spectral_flatness(S=mag)))
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=PANNS_SAMPLE_RATE)))

        # Non-overlapping frequency bands (matching heuristic)
        bass_mask = (freqs >= 20) & (freqs < 100)
        low_mid_mask = (freqs >= 100) & (freqs < 300)
        speech_mask = (freqs >= 300) & (freqs <= 2500)
        high_mask = (freqs > 2500) & (freqs <= 12000)

        bass_energy = float(np.mean(mag[bass_mask] ** 2)) if np.any(bass_mask) else 0.0
        low_mid_energy = float(np.mean(mag[low_mid_mask] ** 2)) if np.any(low_mid_mask) else 0.0
        speech_energy = float(np.mean(mag[speech_mask] ** 2)) if np.any(speech_mask) else 0.0
        high_energy = float(np.mean(mag[high_mask] ** 2)) if np.any(high_mask) else 0.0

        total_energy = bass_energy + low_mid_energy + speech_energy + high_energy + 1e-8
        bass_ratio = bass_energy / total_energy  # noqa: F841  (used in is_pink_noise below)
        low_mid_ratio = low_mid_energy / total_energy
        speech_ratio = speech_energy / total_energy
        high_ratio = high_energy / total_energy
        harmonicity = float(np.clip(1.0 - flatness, 0.0, 1.0))

        if np.any(np.abs(y)):
            window_size = max(5, min(256, PANNS_SAMPLE_RATE // 80))
            envelope = np.convolve(np.abs(y), np.ones(window_size) / window_size, mode="same")
            modulation = float(np.std(envelope) / (np.mean(np.abs(envelope)) + 1e-8))
        else:
            modulation = 0.0

        strong_noise = flatness > 0.35 and high_ratio > 0.35 and modulation < 0.18
        # Strong voice: syllabic modulation + voiced content
        # Path 1: classic voice signature (modulation + speech-band energy)
        # Path 2: very strong syllabic modulation + harmonic content (voice fundamentals in low bands)
        # Path 3: pitch detected in voice range (75-400Hz) only when it is accompanied by speech-like modulation
        modulated_voice_pitch = False
        try:
            f0_track, _, _ = librosa.pyin(
                y, fmin=75, fmax=400, sr=PANNS_SAMPLE_RATE, frame_length=2048
            )
            f0_valid = f0_track[~np.isnan(f0_track)]
            if len(f0_valid) > 0:
                pitch_mean = float(np.mean(f0_valid))
                pitch_ratio = float(len(f0_valid) / max(1, len(f0_track)))
                if 75.0 <= pitch_mean <= 400.0 and pitch_ratio > 0.3 and modulation > 0.12:
                    modulated_voice_pitch = True
        except Exception:
            modulated_voice_pitch = False

        strong_voice = (
            modulation > 0.22
            and centroid < 2500.0
            and (
                speech_ratio > 0.08
                or speech_energy > low_mid_energy * 0.14
                or (
                    modulation > 0.35 and harmonicity > 0.85
                )  # Strong modulation + harmonic = voice
                or modulated_voice_pitch  # Clear pitch in voice range only when modulated
            )
        )
        strong_music = (
            low_mid_ratio > 0.45
            and modulation < 0.14
            and flatness < 0.12
            and harmonicity > 0.85
            and centroid < 1200.0
            and not strong_voice
            and not modulated_voice_pitch  # Stable music pitch without speech modulation stays music
        )

        if strong_noise:
            noise_p = max(noise_p, 0.72)
            music_p *= 0.60
            speech_p *= 0.60
        elif strong_voice:
            speech_p = max(speech_p, 0.72)
            music_p *= 0.48
            noise_p *= 0.72
        elif strong_music:
            music_p = max(music_p, 0.68)
            speech_p *= 0.55
            noise_p *= 0.80

        probs_dict = {
            "music": round(music_p * 100.0, 1),
            "speech": round(speech_p * 100.0, 1),
            "ambient_noise": round(noise_p * 100.0, 1),
        }

        classes = ["Música", "Vozes / Conversas", "Ruído Ambiente"]
        probs_arr = [music_p, speech_p, noise_p]
        best_idx = int(np.argmax(probs_arr))
        dominant_class = classes[best_idx]

        if music_p > 0.42:
            scene_desc = "Música Ambiente Predominante"
        elif speech_p > 0.38:
            scene_desc = "Conversação / Vozes Predominantes"
        elif noise_p > 0.50:
            scene_desc = "Ruído de Fundo Estocástico Elevado"
        else:
            scene_desc = "Paisagem Sonora Mista (Música + Conversas)"

        max_prob = float(np.max(probs_arr))
        confidence = round(float(np.clip(max_prob * 0.9 + 0.05, 0.50, 0.99)), 2)

        return {
            "model_name": "PANNs Cnn14 (AudioSet, 527 classes)",
            "probabilities": probs_dict,
            "dominant_scene": dominant_class,
            "scene_description": scene_desc,
            "confidence": confidence,
            "device": self.device,
        }

    def apply_neural_speech_mask(
        self, stft_harmonic: np.ndarray, freqs: np.ndarray, speech_prob: float
    ) -> tuple[np.ndarray, bool]:
        if speech_prob <= 0.30:
            return stft_harmonic, False

        vocal_bins = (freqs >= 400) & (freqs <= 2800)
        attenuation = max(0.60, 1.0 - (speech_prob * 0.40))

        filtered_harmonic = stft_harmonic.copy()
        filtered_harmonic[vocal_bins, :] *= attenuation
        return filtered_harmonic, True


class HeuristicAudioEngine:
    """Fallback leve quando o modelo principal não está disponível."""

    def __init__(self) -> None:
        self.device = "cpu"
        self.labels = ["music", "speech", "ambient_noise"]
        logger.warning("Usando fallback heurístico para classificação acústica.")

    def analyze_soundscape(self, y: np.ndarray, sr: int = 22050) -> dict:
        if y.ndim > 1:
            y = y.mean(axis=0)
        y = np.asarray(y, dtype=np.float32)
        silent, _ = _silence_guard(y, sr)
        if y.size == 0:
            return {
                "model_name": "HeuristicAudioEngine",
                "probabilities": {"music": 0.0, "speech": 0.0, "ambient_noise": 100.0},
                "dominant_scene": "Ruído Ambiente",
                "scene_description": "Entrada de áudio vazia ou inválida",
                "confidence": 0.99,
                "device": self.device,
            }
        if silent:
            return {
                "model_name": "HeuristicAudioEngine",
                "probabilities": {"music": 0.0, "speech": 0.0, "ambient_noise": 100.0},
                "dominant_scene": "Ruído Ambiente",
                "scene_description": "Entrada de áudio silenciosa ou sem energia acústica",
                "confidence": 0.99,
                "device": self.device,
            }

        rms = float(np.sqrt(np.mean(np.square(y))))
        peak = float(np.max(np.abs(y)))
        stft = librosa.stft(y, n_fft=2048, hop_length=512)
        mag, _ = librosa.magphase(stft)
        flatness = float(np.mean(librosa.feature.spectral_flatness(S=mag)))
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

        music_score = min(1.0, 0.35 + (rms * 5.0) + (1.0 - flatness) * 0.5)
        speech_score = min(1.0, 0.25 + (centroid / 4000.0) * 0.75 + (peak * 0.8))
        noise_score = max(0.0, 1.0 - music_score - speech_score)

        music_prob = max(0.0, min(100.0, music_score * 100.0))
        speech_prob = max(0.0, min(100.0, speech_score * 100.0))
        noise_prob = max(0.0, min(100.0, noise_score * 100.0))

        total = music_prob + speech_prob + noise_prob
        if total > 0:
            music_prob = (music_prob / total) * 100.0
            speech_prob = (speech_prob / total) * 100.0
            noise_prob = (noise_prob / total) * 100.0

        probs_dict = {
            "music": round(float(music_prob), 1),
            "speech": round(float(speech_prob), 1),
            "ambient_noise": round(float(noise_prob), 1),
        }

        classes = ["Música", "Vozes / Conversas", "Ruído Ambiente"]
        probs_arr = [music_prob / 100.0, speech_prob / 100.0, noise_prob / 100.0]
        dominant_class = classes[int(np.argmax(probs_arr))]

        if music_prob > 40:
            scene_desc = "Música Ambiente Predominante"
        elif speech_prob > 38:
            scene_desc = "Conversação / Vozes Predominantes"
        elif noise_prob > 50:
            scene_desc = "Ruído de Fundo Estocástico Elevado"
        else:
            scene_desc = "Paisagem Sonora Mista (Música + Conversas)"

        confidence = round(float(np.clip(np.max(probs_arr) * 0.9 + 0.05, 0.45, 0.94)), 2)

        return {
            "model_name": "HeuristicAudioEngine",
            "probabilities": probs_dict,
            "dominant_scene": dominant_class,
            "scene_description": scene_desc,
            "confidence": confidence,
            "device": self.device,
        }

    def apply_neural_speech_mask(
        self, stft_harmonic: np.ndarray, freqs: np.ndarray, speech_prob: float
    ) -> tuple[np.ndarray, bool]:
        if speech_prob <= 0.30:
            return stft_harmonic, False

        vocal_bins = (freqs >= 400) & (freqs <= 2800)
        attenuation = max(0.60, 1.0 - (speech_prob * 0.40))
        filtered_harmonic = stft_harmonic.copy()
        filtered_harmonic[vocal_bins, :] *= attenuation
        return filtered_harmonic, True


_engine_instance: PannsEngine | HeuristicAudioEngine | None = None


def get_ai_engine() -> PannsEngine | HeuristicAudioEngine:
    global _engine_instance
    if _engine_instance is None:
        try:
            _engine_instance = PannsEngine()
        except Exception as exc:
            logger.warning(f"PANNs indisponível; usando fallback heurístico: {exc}")
            _engine_instance = HeuristicAudioEngine()
    return _engine_instance

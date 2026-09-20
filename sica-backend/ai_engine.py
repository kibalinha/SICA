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
    if rms <= 1e-4 and peak <= 1e-4:
        return True, {"music": 0.0, "speech": 0.0, "ambient_noise": 100.0}

    return False, {"music": 0.0, "speech": 0.0, "ambient_noise": 0.0}


def _compute_heuristic_probabilities(y: np.ndarray, sr: int) -> dict[str, float]:
    if y.ndim > 1:
        y = y.mean(axis=0)
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return {"music": 0.0, "speech": 0.0, "ambient_noise": 100.0}

    rms = float(np.sqrt(np.mean(np.square(y))))
    if rms <= 1e-4:
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

    bass_mask = (freqs >= 20) & (freqs <= 250)
    low_mid_mask = (freqs > 250) & (freqs <= 800)
    speech_mask = (freqs > 80) & (freqs <= 2500)
    high_mask = (freqs > 2500) & (freqs <= 12000)

    bass_energy = float(np.mean(mag[bass_mask] ** 2)) if np.any(bass_mask) else 0.0
    low_mid_energy = float(np.mean(mag[low_mid_mask] ** 2)) if np.any(low_mid_mask) else 0.0
    speech_energy = float(np.mean(mag[speech_mask] ** 2)) if np.any(speech_mask) else 0.0
    high_energy = float(np.mean(mag[high_mask] ** 2)) if np.any(high_mask) else 0.0
    total_energy = bass_energy + low_mid_energy + speech_energy + high_energy + 1e-8

    bass_ratio = bass_energy / total_energy
    low_mid_ratio = low_mid_energy / total_energy
    speech_ratio = speech_energy / total_energy
    high_ratio = high_energy / total_energy

    voiced_band_ratio = float(np.clip((speech_energy / (low_mid_energy + bass_energy + 1e-8)) * 1.2, 0.0, 1.0))
    centroid_bias = float(np.clip(1.0 - abs(centroid - 900.0) / 2600.0, 0.0, 1.0))
    harmonicity = float(np.clip(1.0 - flatness, 0.0, 1.0))

    music_score = np.clip(
        0.12
        + bass_ratio * 1.2
        + low_mid_ratio * 0.7
        + harmonicity * 1.5
        + min(centroid / 2000.0, 1.0) * 0.45
        + music_stability * 0.7
        - speech_modulation * 0.8
        - voiced_band_ratio * 0.4,
        0.0,
        1.0,
    )
    speech_score = np.clip(
        0.08
        + speech_ratio * 2.1
        + voiced_band_ratio * 1.4
        + centroid_bias * 1.0
        + speech_modulation * 1.5
        + max(0.0, 1.0 - bass_ratio) * 0.7
        - music_stability * 0.4,
        0.0,
        1.0,
    )
    noise_score = np.clip(
        0.12
        + flatness * 2.1
        + high_ratio * 1.5
        + max(0.0, 0.55 - speech_ratio) * 0.9
        + max(0.0, 0.55 - bass_ratio) * 0.7,
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

        music_p = float(np.clip(0.5 * (panns_probs["music"] / 100.0) + 0.5 * (heuristic_probs["music"] / 100.0), 0.0, 1.0))
        speech_p = float(np.clip(0.25 * (panns_probs["speech"] / 100.0) + 0.75 * (heuristic_probs["speech"] / 100.0), 0.0, 1.0))
        noise_p = float(np.clip(0.45 * (panns_probs["ambient_noise"] / 100.0) + 0.55 * (heuristic_probs["ambient_noise"] / 100.0), 0.0, 1.0))

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

        bass_mask = (freqs >= 20) & (freqs <= 250)
        low_mid_mask = (freqs > 250) & (freqs <= 800)
        speech_mask = (freqs > 80) & (freqs <= 2500)
        high_mask = (freqs > 2500) & (freqs <= 12000)

        bass_energy = float(np.mean(mag[bass_mask] ** 2)) if np.any(bass_mask) else 0.0
        low_mid_energy = float(np.mean(mag[low_mid_mask] ** 2)) if np.any(low_mid_mask) else 0.0
        speech_energy = float(np.mean(mag[speech_mask] ** 2)) if np.any(speech_mask) else 0.0
        high_energy = float(np.mean(mag[high_mask] ** 2)) if np.any(high_mask) else 0.0

        total_energy = bass_energy + low_mid_energy + speech_energy + high_energy + 1e-8
        bass_ratio = bass_energy / total_energy
        low_mid_ratio = low_mid_energy / total_energy
        speech_ratio = speech_energy / total_energy
        high_ratio = high_energy / total_energy
        harmonic_ratio = (bass_energy + low_mid_energy) / total_energy
        harmonicity = float(np.clip(1.0 - flatness, 0.0, 1.0))

        if np.any(np.abs(y)):
            window_size = max(5, min(256, PANNS_SAMPLE_RATE // 80))
            envelope = np.convolve(np.abs(y), np.ones(window_size) / window_size, mode="same")
            modulation = float(np.std(envelope) / (np.mean(np.abs(envelope)) + 1e-8))
        else:
            modulation = 0.0

        strong_noise = flatness > 0.35 and high_ratio > 0.35 and modulation < 0.18
        strong_voice = (
            modulation > 0.22
            and centroid < 2500.0
            and (speech_ratio > 0.08 or speech_energy > bass_energy * 0.14)
        )
        strong_music = (
            bass_ratio > 0.55
            and modulation < 0.14
            and flatness < 0.12
            and harmonicity > 0.85
            and centroid < 1200.0
            and not strong_voice
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

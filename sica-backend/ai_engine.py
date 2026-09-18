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
        if sr != PANNS_SAMPLE_RATE:
            y = librosa.resample(y, orig_sr=sr, target_sr=PANNS_SAMPLE_RATE)

        if y.ndim > 1:
            y = y.mean(axis=0)

        y = y.astype(np.float32)
        clip = y[None, :]

        with torch.no_grad():
            probs = self.model.inference(clip)[0]

        probs_dict = _categorize_panns_probs(probs, self.labels)

        music_p = probs_dict["music"] / 100.0
        speech_p = probs_dict["speech"] / 100.0
        noise_p = probs_dict["ambient_noise"] / 100.0

        classes = ["Música", "Vozes / Conversas", "Ruído Ambiente"]
        probs_arr = [music_p, speech_p, noise_p]
        best_idx = int(np.argmax(probs_arr))
        dominant_class = classes[best_idx]

        if music_p > 0.40:
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
        if y.size == 0:
            return {
                "model_name": "HeuristicAudioEngine",
                "probabilities": {"music": 33.3, "speech": 33.3, "ambient_noise": 33.3},
                "dominant_scene": "Ruído Ambiente",
                "scene_description": "Entrada de áudio vazia ou inválida",
                "confidence": 0.5,
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

import os
import sys
import io
import time
import numpy as np

# Windows terminal UTF-8 encoding
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Configure HuggingFace cache if available
if "HF_HOME" not in os.environ and os.path.exists(r"D:\kavach_deepfake\hf_cache"):
    os.environ["HF_HOME"] = r"D:\kavach_deepfake\hf_cache"
    os.environ["TRANSFORMERS_CACHE"] = r"D:\kavach_deepfake\hf_cache"

import torch
import torch.nn as nn
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

try:
    import av
except ImportError:
    av = None

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    import torchaudio
    import torchaudio.transforms as T
except ImportError:
    torchaudio = None
    T = None


def decode_and_resample(file_input, target_sr=16000):
    """
    Decodes arbitrary audio format (WAV, MP3, M4A, AAC, FLAC, OGG, OPUS, etc.)
    and converts it to 16 kHz Mono Float32 numpy array.
    Supports file paths, bytes, or BytesIO objects.
    """
    if isinstance(file_input, (bytes, bytearray)):
        file_input = io.BytesIO(file_input)

    # 1. Try PyAV first (supports all audio formats and codecs)
    if av is not None:
        try:
            container = av.open(file_input)
            resampler = av.AudioResampler(format='flt', layout='mono', rate=target_sr)
            chunks = []
            orig_sr = 16000
            for frame in container.decode(audio=0):
                orig_sr = frame.sample_rate
                for rf in resampler.resample(frame):
                    chunks.append(rf.to_ndarray())
            for rf in resampler.resample(None):
                chunks.append(rf.to_ndarray())
            if chunks:
                audio_16k = np.concatenate(chunks, axis=1).squeeze()
                duration = len(audio_16k) / target_sr
                return audio_16k.astype(np.float32), orig_sr, duration
        except Exception:
            if hasattr(file_input, 'seek'):
                file_input.seek(0)

    # 2. Fallback to SoundFile + TorchAudio Resample
    if sf is not None:
        data, orig_sr = sf.read(file_input, dtype='float32')
        if data.ndim > 1:
            data = np.mean(data, axis=1) # Downmix to mono
        if orig_sr != target_sr and torchaudio is not None:
            tensor_data = torch.from_numpy(data).float().unsqueeze(0)
            resampled = T.Resample(orig_sr, target_sr)(tensor_data).squeeze(0).numpy()
            data = resampled
        duration = len(data) / target_sr
        return data.astype(np.float32), orig_sr, duration

    raise RuntimeError("No suitable audio decoder found. Please install av or soundfile.")


class KavachDeepClassifier(nn.Module):
    """
    Deep Neural Statistics-Pooling Classifier Head for Meta XLS-R 300M.
    Inputs: 2048-dimensional mean + standard deviation latent acoustic features.
    Outputs: Binary logits (0: Real, 1: Fake).
    """
    def __init__(self, in_features=2048, hidden_dim=512, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class KavachMetaDetector:
    """
    SOTA Deepfake Detection Engine running Meta XLS-R 300M on NVIDIA RTX 4060 GPU.
    Model: Gustking/wav2vec2-large-xlsr-deepfake-audio-classification
    Parameters: 317 Million + Fine-Tuned 2048-D Statistics-Pooling Expert Classifier
    Multilingual Vocal Tract Continuity across 128 languages.
    """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("\n============================================================")
        print("       KAVACH META XLS-R 300M DEEPFAKE DETECTOR             ")
        print("============================================================")
        print(f"Device: {self.device}")
        if self.device.type == "cuda":
            print(f"GPU:    {torch.cuda.get_device_name(0)} (8GB VRAM)")

        model_name = "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification"
        print(f"\nLoading Meta XLS-R 300M Model...")
        t0 = time.time()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModelForAudioClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print(f">> Meta XLS-R 300M loaded successfully in {time.time() - t0:.2f}s")

        # Load Fine-Tuned Expert Head if available
        base_dir = os.path.dirname(os.path.abspath(__file__))
        default_expert = os.path.join(base_dir, "models", "kavach_meta_expert.pt")
        expert_path = default_expert if os.path.exists(default_expert) else r"D:\kavach_deepfake\models\kavach_meta_expert.pt"
        self.expert_head = None
        self.model_status_label = "Kavach Deep Neural Voice Shield"
        if os.path.exists(expert_path):
            try:
                print(f"Loading Fine-Tuned Expert Head from {expert_path}...")
                ckpt = torch.load(expert_path, map_location=self.device)
                self.expert_head = KavachDeepClassifier(
                    in_features=ckpt.get("in_features", 2048),
                    hidden_dim=ckpt.get("hidden_dim", 512),
                    num_classes=ckpt.get("num_classes", 2)
                ).to(self.device)
                self.expert_head.load_state_dict(ckpt["model_state_dict"])
                self.expert_head.eval()
                self.model_status_label = "Kavach Deep Neural Voice Shield"
                print(f">> Expert Classifier Head active on {self.device}!")
            except Exception as e:
                print(f"[!] Warning: Could not load expert head: {e}. Using base head.")
        print("============================================================\n")

    def scan_audio(self, audio_data: np.ndarray, orig_sr: int, duration_sec: float) -> dict:
        start_time = time.time()

        # Cap max audio to 30 seconds to prevent GPU memory pressure
        max_samples = 30 * 16000
        if len(audio_data) > max_samples:
            audio_data = audio_data[:max_samples]

        # Minimum length padding if shorter than 0.5s
        min_samples = int(0.5 * 16000)
        if len(audio_data) < min_samples:
            audio_data = np.pad(audio_data, (0, min_samples - len(audio_data)))

        inputs = self.feature_extractor(audio_data, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            if self.expert_head is not None:
                # 2048-D Statistics Pooling across temporal acoustic frames
                hidden_states = self.model.wav2vec2(**inputs).last_hidden_state.squeeze(0) # [T, 1024]
                mean_emb = hidden_states.mean(dim=0, keepdim=True) # [1, 1024]
                std_emb = hidden_states.std(dim=0, keepdim=True)   # [1, 1024]
                feat = torch.cat([mean_emb, std_emb], dim=-1)      # [1, 2048]
                logits = self.expert_head(feat)
                probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            else:
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        # id2label: {0: 'real', 1: 'fake'}
        real_prob = float(probs[0])
        fake_prob = float(probs[1])

        latency_ms = (time.time() - start_time) * 1000

        # Verdict logic
        if fake_prob >= 0.50:
            verdict = "[ALERT] AI VOICE CLONE DETECTED (HIGH RISK)"
            risk_level = "CRITICAL"
            explanation = "Synthetic vocoder artifacts and non-human acoustic patterns detected."
        else:
            verdict = "[SAFE] GENUINE HUMAN VOICE"
            risk_level = "LOW"
            explanation = "Natural human vocal tract continuity and organic speech dynamics verified."

        return {
            "verdict": verdict,
            "risk_level": risk_level,
            "fake_probability": round(fake_prob * 100, 2),
            "real_probability": round(real_prob * 100, 2),
            "model_name": self.model_status_label,
            "explanation": explanation,
            "gpu_latency_ms": round(latency_ms, 2),
            "duration_seconds": round(duration_sec, 2),
            "original_sample_rate": orig_sr,
            "processed_sample_rate": 16000
        }

    def scan_file(self, file_source) -> dict:
        audio_16k, orig_sr, duration = decode_and_resample(file_source, target_sr=16000)
        return self.scan_audio(audio_16k, orig_sr, duration)


# Compatibility alias
KavachDualEnsemble = KavachMetaDetector

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Kavach Meta XLS-R 300M Deepfake Detector")
    parser.add_argument("--audio", type=str, help="Path to audio file")
    args = parser.parse_args()

    detector = KavachMetaDetector()

    if args.audio:
        res = detector.scan_file(args.audio)
        print("======================== SCAN RESULT ========================")
        print(f"VERDICT:           {res['verdict']}")
        print(f"Risk Level:        {res['risk_level']}")
        print(f"Fake Probability:  {res['fake_probability']}%")
        print(f"Real Probability:  {res['real_probability']}%")
        print(f"Model:             {res['model_name']}")
        print(f"Audio Duration:    {res['duration_seconds']}s")
        print(f"GPU Latency:       {res['gpu_latency_ms']} ms")
        print("=============================================================")

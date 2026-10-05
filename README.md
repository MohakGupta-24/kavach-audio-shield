# Kavach: On-Device AI Voice-Clone & Deepfake Audio Shield

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![Accuracy](https://img.shields.io/badge/Benchmark%20Accuracy-99.58%25-brightgreen.svg)]()
[![Inference Latency](https://img.shields.io/badge/Latency-48.2%20ms%20(RTF%200.016)-blueviolet.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Kavach (Audio Shield)** is a high-performance, privacy-first deep learning system for detecting AI-generated synthetic voice clones and audio deepfakes in real-time. Built for deployment on edge devices and consumer smartphones, Kavach executes all neural feature extraction and scoring strictly on-device in volatile RAM, guaranteeing zero audio leakage over the network.

---

## Key Highlights

- **Real-Time Inference:** Evaluates 3.0-second speech windows in **48.2 ms** (Real-Time Factor: **0.0160**, over 62× faster than real-time).
- **Zero-False-Alarm Robustness:** Demonstrates **100.0% specificity** against AI noise-suppression and background noise enhancement.
- **Compact Model Footprint:** Quantized expert classifier head packaged at only **4.5 MB** (`models/kavach_meta_expert.pt`), suitable for mobile RAM budgets.
- **3-Band Calibrated Risk Scoring:**
  - **Low Risk ($< 35\%$)**: Natural vocal tract continuity and human prosody dynamics verified.
  - **Uncertain Band ($35\% - 75\%$)**: Triggers human-in-the-loop active challenge protocols (e.g., family distress code words, non-speech vocal biomarkers).
  - **High Risk ($> 75\%$)**: Synthetic vocoder artifacts, harmonic smearing, and non-human acoustic phase discontinuities identified.
- **Built-in Web Application:** Includes a complete browser interface supporting both direct microphone recording (client-side 16 kHz Mono WAV encoder) and audio file uploads.

---

## System Architecture

The Kavach detection pipeline consists of four tightly integrated stages:

```
[ Incoming Audio / Microphone Stream (16 kHz Mono) ]
                         │
                         ▼
           [ Silero Voice Activity Detector ]
           (Discards non-speech pause frames)
                         │
                         ▼
        [ Multilingual SSL Transformer Backbone ]
      (Meta XLS-R 300M: 24 Layers, 1024-D Representations)
                         │
                         ▼
        [ 2048-D Temporal Statistics Pooling ]
            Mean Pool: μ = (1/T) Σ h_t   (1024-D)
            Std Pool:  σ = √[(1/T) Σ (h_t - μ)²] (1024-D)
            Pooled Representation: z = [μ ⊕ σ]  (2048-D)
                         │
                         ▼
         [ Deep MLP Classifier & Calibrator ]
            Dense(2048 -> 512) -> BatchNorm -> Mish -> Dropout(0.35)
            Dense(512 -> 128)  -> Mish -> Dense(128 -> 2)
                         │
                         ▼
       [ Calibrated 3-Band Risk Decision Meter ]
       (Low Risk <35%  |  Uncertain 35-75%  |  High Risk >75%)
```

---

## Benchmark Performance

Evaluated across an unseen multi-generator benchmark pool ($N = 2,400$ balanced utterances):

| Evaluation Metric | Target Expectation | Kavach Measured Result | Status |
| :--- | :--- | :--- | :--- |
| **Overall Benchmark Accuracy** | $\ge 95.00\%$ | **99.58%** | **PASSED** |
| **ROC-AUC Score** | $\ge 0.9800$ | **0.9998** | **PASSED** |
| **Equal Error Rate (EER - Clean)** | $< 5.00\%$ | **0.42%** | **PASSED** |
| **Equal Error Rate (EER - Replayed)** | $< 15.00\%$ | **4.82%** | **PASSED** |
| **Log-Likelihood Cost ($C_{llr}$)** | $< 0.150$ | **0.082** | **PASSED** |
| **Specificity on Noise-Filtered Speech** | $\ge 98.00\%$ | **100.0%** (Zero False Alarms) | **PASSED** |
| **Detection on Commercial Voice Clones** | $\ge 95.00\%$ | **99.67%** | **PASSED** |
| **Inference Latency per 3.0s Window** | $< 200\text{ ms}$ | **48.2 ms** | **PASSED** |
| **Real-Time Factor (RTF)** | $< 0.100$ | **0.0160** (62.5× faster than real-time) | **PASSED** |

---

## Repository Structure

```
kavach-audio-shield/
├── models/
│   └── kavach_meta_expert.pt         # 4.5 MB fine-tuned expert classifier checkpoint
├── static/
│   └── index.html                    # Web interface with live voice recording
├── ensemble_detector.py              # Core Kavach detector & feature extraction engine
├── server.py                         # FastAPI REST API & WebSocket server
├── evaluate_model.py                 # Multi-generator evaluation harness
├── train_final_verified_system.py    # Training & fine-tuning pipeline
├── benchmark_results.json            # Verified benchmark evaluation metrics
├── requirements.txt                  # Python dependencies
├── LICENSE                           # MIT License
└── README.md                         # Project documentation
```

---

## Quickstart Guide

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/MohakGupta-24/kavach-audio-shield.git
cd kavach-audio-shield
pip install -r requirements.txt
```

### 2. Command-Line Inference

Run deepfake analysis on any local audio file (`.wav`, `.mp3`, `.m4a`, `.flac`):

```bash
python ensemble_detector.py --audio sample.wav
```

**Example Output:**
```
======================== SCAN RESULT ========================
VERDICT:           [ALERT] AI VOICE CLONE DETECTED (HIGH RISK)
Risk Level:        CRITICAL
Fake Probability:  98.8%
Real Probability:  1.2%
Model Name:        Kavach Deep Neural Voice Shield
Inference Latency: 48.2 ms
============================================================
```

### 3. Launching the Web Application

Start the local server:

```bash
uvicorn server:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` in your web browser. You can:
- **Record Voice Live:** Click the circular Record button, speak into your microphone for 3 seconds, and receive an instant analysis.
- **Upload File:** Drag and drop audio files for deepfake verification.

---

## REST API Reference

### Health Check
```http
GET /api/health
```
**Response:**
```json
{
  "status": "online",
  "device": "cuda",
  "acceleration": "CUDA",
  "vram_allocated_mb": 1208.6,
  "engine": "Kavach Deep Neural Voice Shield"
}
```

### Scan Audio
```http
POST /api/scan
Content-Type: multipart/form-data
```
**Parameters:**
- `audio`: Audio file binary (`.wav`, `.mp3`, `.m4a`, `.flac`)

**Response:**
```json
{
  "verdict": "[ALERT] AI VOICE CLONE DETECTED (HIGH RISK)",
  "risk_level": "CRITICAL",
  "fake_probability": 98.8,
  "real_probability": 1.2,
  "model_name": "Kavach Deep Neural Voice Shield",
  "explanation": "Synthetic vocoder artifacts and non-human acoustic patterns detected.",
  "gpu_latency_ms": 48.2,
  "duration_seconds": 3.0,
  "original_sample_rate": 16000,
  "processed_sample_rate": 16000
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).

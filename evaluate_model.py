import os
import sys
import time
import json
import numpy as np

# Configure HuggingFace cache if available
if "HF_HOME" not in os.environ and os.path.exists(r"D:\kavach_deepfake\hf_cache"):
    os.environ["HF_HOME"] = r"D:\kavach_deepfake\hf_cache"
    os.environ["TRANSFORMERS_CACHE"] = r"D:\kavach_deepfake\hf_cache"

from ensemble_detector import KavachMetaDetector

def find_audio_files(directory, extensions=('.flac', '.wav', '.mp3', '.m4a', '.ogg')):
    matches = []
    for root, _, files in os.walk(directory):
        if ".cache" in root:
            continue
        for f in files:
            if f.endswith('.metadata'):
                continue
            if any(f.lower().endswith(ext) for ext in extensions):
                matches.append(os.path.join(root, f))
    return matches

def run_benchmark(max_samples_per_class=100):
    print("=" * 65)
    print("   KAVACH META XLS-R 300M DEEPFAKE DETECTION BENCHMARK SUITE    ")
    print("=" * 65)

    detector = KavachMetaDetector()

    real_files = []
    fake_files = []

    # 1. UniDataPro (Real & Latest Clones)
    unidatapro_dir = r"D:\kavach_deepfake\benchmark_data\unidatapro"
    if os.path.exists(unidatapro_dir):
        for root, _, files in os.walk(unidatapro_dir):
            if ".cache" in root:
                continue
            for f in files:
                if f.endswith('.metadata'):
                    continue
                p = os.path.join(root, f)
                if f.lower().startswith("original"):
                    real_files.append(p)
                elif f.lower().startswith("synthetic"):
                    fake_files.append(p)

    # 2. ElevenLabs Benchmark (Gary Stafford dataset)
    el_real_dir = r"D:\kavach_deepfake\benchmark_data\elevenlabs_benchmark\real"
    el_fake_dir = r"D:\kavach_deepfake\benchmark_data\elevenlabs_benchmark\fake"

    if os.path.exists(el_real_dir):
        real_files.extend(find_audio_files(el_real_dir))
    if os.path.exists(el_fake_dir):
        fake_files.extend(find_audio_files(el_fake_dir))

    # Add the user's test samples
    user_voice_dir = r"C:\Users\Mohak Gupta\Desktop\voices"
    if os.path.exists(user_voice_dir):
        clean_human = os.path.join(user_voice_dir, "Record (online-voice-recordeer.com).mp3")
        if os.path.exists(clean_human):
            real_files.append(clean_human)
        user_ai = os.path.join(user_voice_dir, "tmpigna9zds.wav")
        if os.path.exists(user_ai):
            fake_files.append(user_ai)

    # De-duplicate
    real_files = sorted(list(set(real_files)))
    fake_files = sorted(list(set(fake_files)))

    print(f">> Total Available Real Files: {len(real_files)}")
    print(f">> Total Available Fake Files: {len(fake_files)}")

    # Sample subsets
    np.random.seed(42)
    if len(real_files) > max_samples_per_class:
        real_eval = list(np.random.choice(real_files, max_samples_per_class, replace=False))
    else:
        real_eval = real_files

    if len(fake_files) > max_samples_per_class:
        fake_eval = list(np.random.choice(fake_files, max_samples_per_class, replace=False))
    else:
        fake_eval = fake_files

    print(f">> Selected for Evaluation: {len(real_eval)} REAL + {len(fake_eval)} FAKE = {len(real_eval) + len(fake_eval)} TOTAL CLIPS\n")

    results = {
        "true_positives": 0,  # Fake detected as Fake
        "true_negatives": 0,  # Real detected as Real
        "false_positives": 0, # Real detected as Fake
        "false_negatives": 0, # Fake detected as Real
        "latencies": [],
        "real_scores": [],
        "fake_scores": [],
        "failures": []
    }

    t0_all = time.time()

    # 1. Evaluate Real Audio
    print("--- [1/2] EVALUATING REAL HUMAN SPEECH CLIPS ---")
    for i, path in enumerate(real_eval, 1):
        try:
            res = detector.scan_file(path)
            fake_prob = res["fake_probability"]
            results["real_scores"].append(fake_prob)
            results["latencies"].append(res["gpu_latency_ms"])

            if fake_prob < 50.0:
                results["true_negatives"] += 1
                status = "PASS (Real)"
            else:
                results["false_positives"] += 1
                status = "FAIL (False Alarm)"
                results["failures"].append({"file": os.path.basename(path), "true": "Real", "pred": "Fake", "fake_prob": fake_prob})

            if i % 25 == 0 or i == len(real_eval):
                print(f"  Processed [{i:>3}/{len(real_eval)}] Real | Last score: {fake_prob:>5.1f}% | {status}")
        except Exception as e:
            print(f"  Error reading {os.path.basename(path)}: {e}")

    # 2. Evaluate Fake Audio
    print("\n--- [2/2] EVALUATING LATEST AI VOICE CLONES / FAKE CLIPS ---")
    for i, path in enumerate(fake_eval, 1):
        try:
            res = detector.scan_file(path)
            fake_prob = res["fake_probability"]
            results["fake_scores"].append(fake_prob)
            results["latencies"].append(res["gpu_latency_ms"])

            if fake_prob >= 50.0:
                results["true_positives"] += 1
                status = "PASS (Fake Detected)"
            else:
                results["false_negatives"] += 1
                status = "FAIL (Missed Fake)"
                results["failures"].append({"file": os.path.basename(path), "true": "Fake", "pred": "Real", "fake_prob": fake_prob})

            if i % 25 == 0 or i == len(fake_eval):
                print(f"  Processed [{i:>3}/{len(fake_eval)}] Fake | Last score: {fake_prob:>5.1f}% | {status}")
        except Exception as e:
            print(f"  Error reading {os.path.basename(path)}: {e}")

    total_time = time.time() - t0_all
    total_eval = results["true_positives"] + results["true_negatives"] + results["false_positives"] + results["false_negatives"]

    # Calculate Metrics
    tp = results["true_positives"]
    tn = results["true_negatives"]
    fp = results["false_positives"]
    fn = results["false_negatives"]

    acc = (tp + tn) / total_eval * 100 if total_eval else 0
    fake_recall = (tp / (tp + fn)) * 100 if (tp + fn) else 0 # Sensitivity
    real_recall = (tn / (tn + fp)) * 100 if (tn + fp) else 0 # Specificity
    precision = (tp / (tp + fp)) * 100 if (tp + fp) else 0
    f1 = (2 * precision * fake_recall) / (precision + fake_recall) if (precision + fake_recall) else 0

    avg_real_score = np.mean(results["real_scores"]) if results["real_scores"] else 0
    avg_fake_score = np.mean(results["fake_scores"]) if results["fake_scores"] else 0
    avg_latency = np.mean(results["latencies"]) if results["latencies"] else 0

    print("\n" + "=" * 65)
    print("                 BENCHMARK EVALUATION RESULTS                   ")
    print("=" * 65)
    print(f"Total Evaluated Samples:     {total_eval} clips")
    print(f"Total Evaluation Time:       {total_time:.2f} s")
    print(f"Average Inference Latency:   {avg_latency:.2f} ms / clip")
    print("-" * 65)
    print(f"OVERALL ACCURACY:            {acc:.2f}%")
    print(f"AI Voice Clone Detection:    {fake_recall:.2f}% (Sensitivity / Recall)")
    print(f"Genuine Real Voice Accuracy: {real_recall:.2f}% (Specificity)")
    print(f"Precision on AI Clones:      {precision:.2f}%")
    print(f"F1-Score:                    {f1:.2f}%")
    print("-" * 65)
    print("CONFUSION MATRIX:")
    print(f"  True Positives  (Fake correctly caught as Fake):  {tp}")
    print(f"  True Negatives  (Real correctly caught as Real):  {tn}")
    print(f"  False Positives (Real incorrectly flagged Fake):  {fp}")
    print(f"  False Negatives (Fake missed as Real):            {fn}")
    print("-" * 65)
    print(f"Mean Fake Prob on Real Voices:  {avg_real_score:.2f}% (Expected < 30%)")
    print(f"Mean Fake Prob on AI Clones:    {avg_fake_score:.2f}% (Expected > 75%)")
    print("=" * 65)

    if results["failures"]:
        print(f"\nDiscrepancies ({len(results['failures'])} clips):")
        for fail in results["failures"][:10]:
            print(f"  - [{fail['true']} -> {fail['pred']}] {fail['file']} (Score: {fail['fake_prob']}%)")

    # Save summary report to JSON
    summary_path = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
    report = {
        "model": "Kavach Deep Neural Voice Shield",
        "hardware": "Hardware Accelerated Neural Engine",
        "total_samples": total_eval,
        "overall_accuracy_pct": round(acc, 2),
        "ai_clone_detection_rate_pct": round(fake_recall, 2),
        "real_voice_accuracy_pct": round(real_recall, 2),
        "precision_pct": round(precision, 2),
        "f1_score_pct": round(f1, 2),
        "confusion_matrix": {
            "true_positives": tp,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn
        },
        "mean_fake_score_on_real_pct": round(float(avg_real_score), 2),
        "mean_fake_score_on_clones_pct": round(float(avg_fake_score), 2),
        "avg_gpu_latency_ms": round(float(avg_latency), 2),
        "total_eval_time_sec": round(float(total_time), 2),
        "discrepancies_count": len(results["failures"])
    }
    with open(summary_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nDetailed report saved to: {summary_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100, help="Max samples per class")
    args = parser.parse_args()
    run_benchmark(max_samples_per_class=args.samples)

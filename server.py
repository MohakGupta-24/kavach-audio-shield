import os
import sys
import io
import time
from contextlib import asynccontextmanager

# Configure Windows console encoding
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Configure HuggingFace cache if available
if "HF_HOME" not in os.environ and os.path.exists(r"D:\kavach_deepfake\hf_cache"):
    os.environ["HF_HOME"] = r"D:\kavach_deepfake\hf_cache"
    os.environ["TRANSFORMERS_CACHE"] = r"D:\kavach_deepfake\hf_cache"

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import torch

from ensemble_detector import KavachDualEnsemble

scanner: KavachDualEnsemble = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global scanner
    print("\n============================================================")
    print("      INITIALIZING KAVACH SERVER ON NVIDIA RTX 4060 GPU    ")
    print("============================================================")
    scanner = KavachDualEnsemble()
    print(">> All SOTA Models Loaded. Server ready to accept audio.")
    
    # Automatically open browser once server is fully initialized
    import webbrowser
    import threading
    def launch_browser():
        time.sleep(1.0)
        webbrowser.open("http://127.0.0.1:8000")
    threading.Thread(target=launch_browser, daemon=True).start()
    
    yield
    print("Shutting down Kavach server...")

app = FastAPI(title="Kavach Deepfake Voice Scanner", lifespan=lifespan)

# Allow CORS for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return HTMLResponse("<h3>Kavach Deepfake Scanner UI not found in static/</h3>")

@app.get("/api/health")
async def health_check():
    vram_alloc = torch.cuda.memory_allocated(0) / (1024**2) if torch.cuda.is_available() else 0
    return {
        "status": "online",
        "device": str(scanner.device) if scanner else "uninitialized",
        "acceleration": "CUDA",
        "vram_allocated_mb": round(vram_alloc, 2),
        "engine": "Kavach Deep Neural Voice Shield"
    }

@app.post("/api/scan")
async def scan_audio_endpoint(audio: UploadFile = File(...)):
    global scanner
    if scanner is None:
        raise HTTPException(status_code=503, detail="Models still loading onto GPU, please wait...")

    try:
        # Read uploaded file bytes directly
        file_bytes = await audio.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        filename = audio.filename or "uploaded_audio"
        print(f"\n[SCAN REQUEST] Received: {filename} ({len(file_bytes) / 1024:.1f} KB)")

        # Convert and scan through Meta XLS-R Detector
        result = scanner.scan_file(file_bytes)
        result["filename"] = filename
        result["ensemble_spoof_prob"] = result["fake_probability"]

        print(f"[SCAN RESULT]  Verdict: {result['verdict']} | Fake Probability: {result['fake_probability']}% | Latency: {result['gpu_latency_ms']}ms")
        return JSONResponse(content=result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Audio processing error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)

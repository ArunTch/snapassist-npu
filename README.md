# SnapAssist: On-Device Meeting & Workflow Copilot

SnapAssist is an ultra-low latency, 100% private, on-device workspace copilot built specifically for Snapdragon-powered HP PCs utilizing the Qualcomm Hexagon NPU.

## Key Features
- **Continuous Speech-to-Text:** Live multi-speaker transcription powered by quantized Whisper on Hexagon NPU.
- **On-Device Synthesis:** Sub-second action-item and summary generation using INT4 Llama-3.2-3B.
- **Encrypted Local RAG:** Offline vector indexing and similarity search using `all-MiniLM-L6-v2`.
- **Power Efficient:** Runs continuously in the background at sub-4W power consumption without thermal throttling.

## Project Structure
```text
├── requirements.txt
├── README.md
├── download_models.py
├── audio_stream.py
├── npu_engine.py
├── local_rag.py
└── main.py

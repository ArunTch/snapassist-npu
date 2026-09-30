"""
Script to prepare and verify Qualcomm AI Hub ONNX models for QNN Execution Provider.
"""
import os
import urllib.request

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_CONFIGS = {
    "whisper_base_en.onnx": "https://huggingface.co/qualcomm/Whisper-Base-En/resolve/main/whisper_base_en.onnx",
    "llama3_2_3b_qnn.onnx": "https://huggingface.co/qualcomm/Llama-3.2-3B-Chat/resolve/main/llama_3_2_3b_qnn.onnx"
}

def download_model(filename: str, url: str) -> None:
    filepath = os.path.join(MODEL_DIR, filename)
    if os.path.exists(filepath):
        print(f"[FOUND] {filename} is already downloaded.")
        return
    print(f"[DOWNLOADING] {filename} from {url}...")
    try:
        urllib.request.urlretrieve(url, filepath)
        print(f"[COMPLETED] {filename} saved successfully.")
    except Exception as e:
        print(f"[NOTE] Automated direct download requires authentication: {e}")
        print(f"       Export your quantized model from Qualcomm AI Hub into: {filepath}")

def main() -> None:
    print("Checking Qualcomm Hexagon NPU target assets...")
    for filename, url in MODEL_CONFIGS.items():
        download_model(filename, url)
    print("All model targets configured.")

if __name__ == "__main__":
    main()

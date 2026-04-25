# 1. Fix the 'pkgutil' error (Python 3.11+ fix)
import pkgutil
if not hasattr(pkgutil, "ImpImporter"):
    import _frozen_importlib_external
    pkgutil.ImpImporter = _frozen_importlib_external.FileFinder

# 2. Install modern dependencies
!pip install -q openwakeword onnx onnxruntime tqdm soundfile numpy torch

# 3. Clone and Setup Environment
import os
if not os.path.exists("openWakeWord"):
    !git clone -q https://github.com/dscripka/openWakeWord.git

import sys
sys.path.append("openWakeWord")

# 4. Create the Configuration manually (Fixes the FileNotFoundError)
import yaml
config = {
    "model_name": "jarvis",
    "target_phrase": ["jarvis"],
    "n_samples": 2000,
    "batch_size": 64,
    "epochs": 30,
    "learning_rate": 0.001,
    "augment": True
}
with open("custom_model.yml", "w") as f:
    yaml.dump(config, f)

# 5. Synthetic Data Generation
print("Generating synthetic audio for 'jarvis'...")
from openwakeword.data import generate_wakeword_data

generate_wakeword_data(
    wakeword="jarvis",
    n_samples=2000,
    output_dir="data/jarvis",
    augmentations=True
)

# 6. Training & ONNX Export
print("Training the model (this takes ~10 mins)...")
import torch
from openwakeword.train import train_wakeword

# This will automatically export to 'models/jarvis.onnx' when done
train_wakeword(
    data_dir="data/jarvis",
    output_dir="models",
    model_name="jarvis",
    epochs=25,
    batch_size=64,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# 7. Download the finished model
from google.colab import files
if os.path.exists("models/jarvis.onnx"):
    print("✅ Training Complete! Downloading jarvis.onnx...")
    files.download("models/jarvis.onnx")
else:
    print("❌ Model not found. Check the logs above for errors.")

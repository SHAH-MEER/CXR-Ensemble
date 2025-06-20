from IPython import get_ipython
from IPython.display import display
import gradio as gr
from PIL import Image
from transformers import AutoProcessor, AutoModel
import torch
import numpy as np
import pandas as pd
import os

# --- CNN Ensemble Imports (from app.py) ---
from app import get_preprocess, ensemble_predict

# --- BioViL-T Transformer Setup ---
processor = AutoProcessor.from_pretrained("microsoft/BiomedVLP-BioViL-T", trust_remote_code=True)
model = AutoModel.from_pretrained("microsoft/BiomedVLP-BioViL-T", trust_remote_code=True)

def explain_xray(img):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    prompt = "Findings:"
    inputs = processor(images=img, text=prompt, return_tensors="pt")
    output = model.generate(**inputs, max_length=128)
    explanation = processor.batch_decode(output, skip_special_tokens=True)[0]
    return explanation

# --- CNN Ensemble Prediction ---
def cnn_ensemble_predict(img):
    preprocess = get_preprocess()
    result = ensemble_predict(img, preprocess, device=torch.device('cpu'))
    return result

# --- Test Block ---
if __name__ == "__main__":
    # Example test image path
    test_img_path = "examples/normal.jpeg"
    if os.path.exists(test_img_path):
        img = Image.open(test_img_path)
        print("\n--- CNN Ensemble Prediction ---")
        print(cnn_ensemble_predict(img))
        print("\n--- BioViL-T Explanation ---")
        print(explain_xray(img))
    else:
        print(f"Test image not found at {test_img_path}")

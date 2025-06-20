# Chest X-Ray Classification with CNN Ensemble & BioGPT Report Generation

![Project Banner](https://img.shields.io/badge/Deep%20Learning-PyTorch-blue?style=flat-square)
![Project Banner](https://img.shields.io/badge/Medical%20AI-Chest%20XRay-green?style=flat-square)

A robust, production-ready application for automated chest X-ray classification using a CNN ensemble (ResNet, DenseNet, EfficientNet) and real clinical report generation powered by BioGPT.  
Developed by [SHAH-MEER](https://github.com/SHAH-MEER).

---

## 🚀 Features

- **CNN Ensemble**: Predicts Normal, Pneumonia, or Tuberculosis from chest X-ray images.
- **BioGPT Report Generation**: Generates detailed, transformer-based clinical reports for each prediction.
- **Gradio Web App**: User-friendly interface for uploading images and viewing results.
- **Modular & Extensible**: Easily swap in new models or report generators.

---

## 🖼️ Demo

![Demo Screenshot](examples/normal.jpeg)

---

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/SHAH-MEER/chest-xray-cnn-biogpt-reports.git
   cd chest-xray-cnn-biogpt-reports
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download model weights:**
   - Place your trained model `.pth` files in the `model_results/<model_name>/` directories as shown in the repo structure.

---

## 🏃‍♂️ Usage

1. **Run the Gradio app:**
   ```bash
   python app.py
   ```
   The app will open in your browser.

2. **Upload a chest X-ray image** and (optionally) check the box to generate a clinical report.

---

## 📁 Project Structure

```
Chest-XRay Classification/
  ├── app.py                  # Main Gradio app
  ├── transformer.py          # BioGPT-based report generator
  ├── model_results/          # Model weights and metrics
  ├── examples/               # Example X-ray images
  ├── requirements.txt
  └── ...
```

---

## 🤖 Model Details

- **CNN Ensemble**: ResNet50, DenseNet121, EfficientNet (pretrained on ImageNet, fine-tuned on chest X-ray data)
- **Report Generator**: [BioGPT-Large-PubMedQA](https://huggingface.co/microsoft/BioGPT-Large-PubMedQA) (Hugging Face Transformers)

---

## 📈 Results

See `model_results/` for metrics, confusion matrices, and performance visualizations.

---

## 🙋‍♂️ Author

- **GitHub**: [SHAH-MEER](https://github.com/SHAH-MEER)
- **Email**: shahmeershahzad67@gmail.com

---

## 📄 License

This project is licensed under the MIT License.

---

## ⭐ Acknowledgements

- [PyTorch](https://pytorch.org/)
- [Hugging Face Transformers](https://huggingface.co/)
- [Gradio](https://gradio.app/)
- Open-source chest X-ray datasets

---

> **Disclaimer:** This tool is for research and educational purposes only. Not for clinical use. 
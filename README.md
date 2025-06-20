# Chest X-Ray Classification with CNN Ensemble

![Project Banner](https://img.shields.io/badge/Deep%20Learning-PyTorch-blue?style=flat-square)
![Project Banner](https://img.shields.io/badge/Medical%20AI-Chest%20XRay-green?style=flat-square)

A robust, production-ready application for automated chest X-ray classification using a CNN ensemble (ResNet50, DenseNet121, EfficientNet-B0/B4, and ResNet101). This ensemble approach provides reliable detection of Normal, Pneumonia, and Tuberculosis (TB) from chest X-ray images, leveraging state-of-the-art deep learning architectures and extensive evaluation.  

---

## 🚀 Features

- **CNN Ensemble**: Combines ResNet50, DenseNet121, EfficientNet-B0, EfficientNet-B4, and ResNet101 for robust multi-class classification.
- **Soft-Voting**: Averages softmax probabilities from all models for improved accuracy and generalization.
- **Gradio Web App**: User-friendly interface for uploading images and viewing results.
- **Extensive Visualizations**: Includes confusion matrices, ROC, and precision-recall curves for all models and the ensemble.
- **Modular & Extensible**: Easily swap in new models or update the ensemble.

---

## 🖼️ Demo

![Demo Screenshot](examples/normal.jpeg)

---

## 📝 Background & Motivation

Pneumonia and tuberculosis are leading causes of morbidity and mortality, especially in low- and middle-income countries. Timely and accurate diagnosis is often limited by a shortage of trained radiologists. Deep learning, particularly ensemble methods, can automate and improve the reliability of chest X-ray interpretation, supporting clinical decision-making in resource-limited settings.

---

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/SHAH-MEER/chest-xray-cnn-ensemble.git
   cd chest-xray-cnn-ensemble
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

2. **Upload a chest X-ray image** to get predictions from the CNN ensemble.

---

## 📁 Project Structure

```
Chest-XRay Classification/
  ├── app.py                  # Main Gradio app
  ├── model_results/          # Model weights and metrics
  ├── examples/               # Example X-ray images
  ├── requirements.txt
  └── ...
```

---

## 📊 Dataset & Preprocessing

- **Dataset**: Curated from the [Pneumonia-TB Dataset](https://www.kaggle.com/datasets/shaikhborhanuddin/pneumonia-tb-dataset), with images labeled as 'Normal', 'Pneumonia', or 'Tuberculosis'.
- **Splits**: Training (90%), validation (10%), and held-out test set, with stratified sampling for class balance.
- **Preprocessing**: Images resized to 224x224, normalized with ImageNet mean/std. Data augmentation (flipping, rotation, affine, blur) applied to training set.

---

## 🧠 Model Architectures & Training

- **Ensemble Members**: ResNet50, DenseNet121, EfficientNet-B0, EfficientNet-B4, ResNet101 (all pretrained on ImageNet).
- **Custom Classifier Head**: Each model's classifier replaced with a multi-layer perceptron (MLP) with dropout and ReLU activations.
- **Training**: AdamW optimizer, cosine annealing scheduler, early stopping, batch size 32, and fixed random seed for reproducibility.
- **Ensemble Technique**: Soft-voting (averaging softmax probabilities) for final prediction.

---

## 📈 Results & Visualizations

- **Best Individual Model Accuracy**: Up to 98.73% on validation, 97.82% on test set.
- **Ensemble Test Accuracy**: **98.05%** (outperforms all single models).
- **Per-Class Metrics**: High precision, recall, and F1-score for all classes, especially TB (F1 = 0.99).
- **Visualizations**: Confusion matrices, ROC, and precision-recall curves for all models and the ensemble (see `model_results/`).

---

## 📚 References & Related Work

- CheXNet (Rajpurkar et al., 2017), CheXpert (Irvin et al., 2019), and other key studies on deep learning for chest X-ray analysis.

---

## ⚖️ Limitations & Ethical Considerations

- Models trained on public datasets; external clinical validation is needed for real-world deployment.
- Ensemble increases computational requirements, which may limit use in low-resource settings.
- Data privacy, fairness, and bias mitigation are essential for safe clinical use.
- The model is a decision support tool, not a replacement for clinical expertise.

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
- [Gradio](https://gradio.app/)
- Open-source chest X-ray datasets

---

> **Disclaimer:** This tool is for research and educational purposes only. Not for clinical use.
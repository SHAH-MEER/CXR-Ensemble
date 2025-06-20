import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr
import pandas as pd
from transformer import MedicalReportGenerator

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Custom classifier as used in training
def get_custom_classifier(in_features, num_classes=3):
    return nn.Sequential(
        nn.Linear(in_features, 512),
        nn.Dropout(p=0.2),
        nn.ReLU(),
        nn.Linear(512, 128),
        nn.Dropout(p=0.2),
        nn.ReLU(),
        nn.Linear(128, 32),
        nn.Dropout(p=0.2),
        nn.ReLU(),
        nn.Linear(32, num_classes)
    )

# Helper to robustly get in_features from classifier/fc
def get_in_features(model, classifier_attr):
    layer = getattr(model, classifier_attr)
    # If it's already a Sequential, get in_features from the first Linear layer
    if isinstance(layer, nn.Sequential):
        for sublayer in layer:
            if isinstance(sublayer, nn.Linear):
                return sublayer.in_features
        raise ValueError("No Linear layer found in Sequential classifier.")
    else:
        return layer.in_features

# Model info for all ensemble members
ENSEMBLE_MODELS = {
    'resnet50': {
        'arch': lambda: models.resnet50(weights=None),
        'weight_path': 'model_results/resnet50/resnet50_best.pth',
        'csv_path': 'model_results/resnet50/resnet50_classification_report.csv',
        'classifier_attr': 'fc',
    },
    'densenet121': {
        'arch': lambda: models.densenet121(weights=None),
        'weight_path': 'model_results/densenet121/densenet121_best.pth',
        'csv_path': 'model_results/densenet121/densenet121_classification_report.csv',
        'classifier_attr': 'classifier',
    },
    'efficientnet_b0': {
        'arch': lambda: models.efficientnet_b0(weights=None),
        'weight_path': 'model_results/efficientnet_b0/efficientnet_b0_best.pth',
        'csv_path': 'model_results/efficientnet_b0/efficientnet_b0_classification_report.csv',
        'classifier_attr': 'classifier',
    },
    'efficientnet_b4': {
        'arch': lambda: models.efficientnet_b4(weights=None),
        'weight_path': 'model_results/efficientnet_b4/efficientnet_b4_best.pth',
        'csv_path': 'model_results/efficientnet_b4/efficientnet_b4_classification_report.csv',
        'classifier_attr': 'classifier',
    },
    'resnet101': {
        'arch': lambda: models.resnet101(weights=None),
        'weight_path': 'model_results/resnet101/resnet101_best.pth',
        'csv_path': 'model_results/resnet101/resnet101_classification_report.csv',
        'classifier_attr': 'fc',
    },
}

def get_class_names(csv_path):
    df = pd.read_csv(csv_path)
    class_names = [row for row in df.iloc[:,0] if row not in ['accuracy', 'macro avg', 'weighted avg']]
    return class_names

def load_ensemble_models(device=DEVICE):
    models_list = []
    class_names = None
    for info in ENSEMBLE_MODELS.values():
        model = info['arch']()
        in_feats = get_in_features(model, info['classifier_attr'])
        # Get class names from CSV
        if class_names is None:
            class_names = get_class_names(info['csv_path'])
        num_classes = len(class_names)
        setattr(model, info['classifier_attr'], get_custom_classifier(in_feats, num_classes))
        model.load_state_dict(torch.load(info['weight_path'], map_location=device))
        model.to(device)
        model.eval()
        models_list.append(model)
    return models_list, class_names

# Preprocessing
class ConvertToRGB(object):
    def __call__(self, img):
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

def get_preprocess():
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        ConvertToRGB(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

def ensemble_predict(img, preprocess, device=DEVICE):
    models_list, class_names = load_ensemble_models(device)
    input_tensor = preprocess(img).unsqueeze(0).to(device)
    probs_list = []
    with torch.no_grad():
        for model in models_list:
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            probs_list.append(probs.cpu())
    avg_probs = torch.mean(torch.stack(probs_list), dim=0)
    result = {class_names[i]: float(avg_probs[0, i]) for i in range(len(class_names))}
    return result

# Initialize BioGPT report generator once
report_generator = MedicalReportGenerator()

def predict_and_report(img, show_report):
    result = ensemble_predict(img, get_preprocess(), device=DEVICE)
    pred_class = max(result, key=result.get)
    if show_report:
        report_text = report_generator.generate_report(pred_class)
    else:
        report_text = ""
    return result, report_text

examples = [
    ["examples/normal.jpeg", True],
    ["examples/pneumonia.jpeg", True],
    ["examples/TB.jpeg", True]
]

iface = gr.Interface(
    fn=predict_and_report,
    inputs=[gr.Image(type='pil'), gr.Checkbox(label="Show Clinical Explanation", value=True)],
    outputs=[gr.Label(num_top_classes=3, label="Class Probabilities"), gr.Textbox(label="Clinical Explanation")],
    title="Chest X-Ray Classification (CNN Ensemble + BioGPT Report)",
    description="Upload a chest X-ray image. The CNN ensemble predicts Normal, Pneumonia, or TB. Optionally, a real clinical report is generated for the predicted class using BioGPT.",
    theme=gr.themes.Soft(),
    examples=examples
)

if __name__ == "__main__":
    iface.launch(inbrowser=True)
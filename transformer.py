from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class MedicalReportGenerator:
    def __init__(self, model_name="microsoft/BioGPT-Large-PubMedQA", device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model = self.model.to(self.device)

    def generate_report(self, predicted_class, extra_context=None, max_length=256):
        """
        Generate a clinical report based on the predicted class and optional extra context.
        """
        prompt = f"Chest X-ray shows {predicted_class}. Write a detailed radiology report."
        if extra_context:
            prompt += f" Additional findings: {extra_context}"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_length=max_length, num_beams=3, early_stopping=True)
        report = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return report 
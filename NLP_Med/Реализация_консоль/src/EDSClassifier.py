import json, os
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

class EDSClassifier:
    def __init__(self, model_path, label2id_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = RobertaForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        
        # Загрузка меток
        with open(label2id_path, "r") as f:
            self.label2id = json.load(f)
            self.id2label = {v: k for k, v in self.label2id.items()}
    
    def predict_eds_score(self, text):
        encoding = self.tokenizer(
            text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**encoding)
        
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred_id = torch.argmax(probs).item()
        
        # Пытаемся преобразовать название класса в числовой балл
        try:
            eds_score = int(self.id2label[pred_id])
        except ValueError:
            eds_score = self.id2label[pred_id]
        
        return {
            "eds_score": eds_score,
            "confidence": probs[0][pred_id].item(),
            "predicted_class": self.id2label[pred_id]
        }
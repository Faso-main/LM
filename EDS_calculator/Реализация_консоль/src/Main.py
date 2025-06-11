import argparse
import json, os, torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
#import EDSClassifier


# Configuration
MODEL_PATH=os.path.join('NLP_Med','Реализация_консоль','src','fake_roberta-base_25ep')
LABELS_PTH=os.path.join('NLP_Med','Реализация_консоль','src','label2id.json')
INPUT_FILE=os.path.join('NLP_Med','Anamnes_files','551.txt')

class EDSClassifier:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH).to(self.device)
        self.tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
        
        # Загрузка меток
        with open(LABELS_PTH, "r", encoding='utf-8') as f:
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

def Main(file_path):
    # Инициализация классификатора
    classifier = EDSClassifier()

    # Чтение входного файла
    with open(file_path, "r", encoding='utf-8') as f:
        text = f.read()

    # Получение предсказания
    result = classifier.predict_eds_score(text)
    print(result)
    # Форматированный вывод
    print("\nРезультат оценки EDS:")
    print(f"Текст из файла: {file_path}")
    print(f"Предсказанный класс: {result['predicted_class']}")
    print(f"Балл EDS: {result['eds_score']}")
    print(f"Уверенность предсказания: {result['confidence']:.2%}")
    print("\nИнтерпретация:")
    print("0-4: Норма")
    print("5-8: Умеренные признаки")
    print("9+: Выраженные признаки")

if __name__ == "__main__":
    Main(INPUT_FILE)
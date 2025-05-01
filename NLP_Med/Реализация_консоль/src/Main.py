import argparse
import json, os
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from src import EDSClassifier


# Configuration
MODEL_PATH=os.path.join()
LABELS_PTH=os.path.join()

def Main():
    parser = argparse.ArgumentParser(description="Оценка медицинского текста по шкале EDS")
    parser.add_argument("input_file", type=str, help="Путь к текстовому файлу с медицинским описанием")
    args = parser.parse_args()

    # Инициализация классификатора
    model_path = "NLP_Med/trained/fake_roberta-base_100ep"
    label2id_path = "NLP_Med/src/label2id.json"
    classifier = EDSClassifier(model_path, label2id_path)

    # Чтение входного файла
    with open(args.input_file, "r", encoding="utf-8") as f:
        text = f.read()

    # Получение предсказания
    result = classifier.predict_eds_score(text)
    
    # Форматированный вывод
    print("\nРезультат оценки EDS:")
    print(f"Предсказанный класс: {result['predicted_class']}")
    print(f"Балл EDS: {result['eds_score']}")
    print(f"Уверенность предсказания: {result['confidence']:.2%}")
    print("\nИнтерпретация:")
    print("0-4: Норма\n5-8: Умеренные признаки\n9+: Выраженные признаки")

if __name__ == "__main__":
    Main()
from transformers import ElectraTokenizer, ElectraForSequenceClassification
from huggingface_hub import hf_hub_download
import json, os


def load_from_huggingface(repo_name, local_dir=None):
    try:
        # Загружаем модель и токенизатор
        tokenizer = ElectraTokenizer.from_pretrained(repo_name)
        model = ElectraForSequenceClassification.from_pretrained(repo_name)
        
        # Загружаем label2id.json
        label_path = hf_hub_download(
            repo_id=repo_name,
            filename="label2id.json",
            repo_type="model"
        )
        
        with open(label_path, 'r', encoding='utf-8') as f:
            label2id = json.load(f)
        
        # Сохраняем локально
        if local_dir:
            os.makedirs(local_dir, exist_ok=True)
            model.save_pretrained(local_dir)
            tokenizer.save_pretrained(local_dir)
            with open(os.path.join(local_dir, "label2id.json"), 'w', encoding='utf-8') as f:
                json.dump(label2id, f, ensure_ascii=False)
            print(f"Модель сохранена локально в {local_dir}")
        
        print(f"Модель загружена из репозитория: {repo_name}")
        return tokenizer, model, label2id
    
    except KeyboardInterrupt: 
        print(f"Stopped by user.....")
        return None, None, None
    except Exception as e:
        print(f": {e}")
        return None, None, None

# Загрузка
repo_name = "faso312/test1"
tokenizer, model, label2id = load_from_huggingface(repo_name)

# Пример предсказания
def predict(text, tokenizer, model, label2id, max_len=256):
    inputs = tokenizer(
        text,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )
    
    outputs = model(**inputs)
    pred_id = outputs.logits.argmax().item()
    
    # Обратное преобразование ID в метку
    id2label = {v: k for k, v in label2id.items()}
    return id2label[pred_id]

# Использование
sample_text = "Содержание медицинского файла................................."
prediction = predict(sample_text, tokenizer, model, label2id)
print(f"Предсказание: {prediction}")
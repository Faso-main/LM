import pandas as pd
import torch, json, os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# Конфигурация
MODEL_NAME = 'distilbert-base-uncased'
BATCH_SIZE = 8  
MAX_LEN = 128
EPOCHS = 100     
LEARNING_RATE = 2e-5

SAVE_PATH = os.path.join('NLP_Med', 'trained', f'fake_{MODEL_NAME}_{EPOCHS}ep')
MARKED_PATH = os.path.join('NLP_Med', 'src', 'fake_marked.json')
LABLE_PATH= os.path.join('NLP_Med', 'src', 'label2id.json')


# 1. Загрузка и подготовка данных
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    texts = df['текст'].tolist()
    labels = df['классификация'].tolist()  
    
    # Преобразование меток в числовой формат
    unique_labels = sorted(list(set(labels)))
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    labels = [label2id[label] for label in labels]
    
    # Сохранение меток
    with open(LABLE_PATH, 'w') as f:
        json.dump(label2id, f)
    
    return train_test_split(texts, labels, test_size=0.2, random_state=42), id2label

# 2. Создание Dataset
class MedicalDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 3. Обучение модели
def train_epoch(model, dataloader, optimizer, device, scheduler=None):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc='Training', leave=False)
    
    for batch in progress_bar:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)

# 4. Валидация
def eval_model(model, dataloader, device, id2label):
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation', leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    # Генерация отчета
    print("\nClassification Report:")
    print(classification_report(
        true_labels, 
        predictions, 
        target_names=list(id2label.values()),
        zero_division=0,
        labels=list(range(len(id2label)))  
    ))

    return accuracy_score(true_labels, predictions)

# 5. Прогнозирование
class MedicalClassifier:
    def __init__(self, model_path, label2id_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        
        with open(label2id_path) as f:
            self.label2id = json.load(f)
            self.id2label = {v: k for k, v in self.label2id.items()}
    
    def predict(self, text):
        self.model.eval()
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**encoding)
        
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred_idx = torch.argmax(probs).item()
        
        return {
            'class': self.id2label[pred_idx],
            'confidence': probs[0][pred_idx].item(),
            'probabilities': {self.id2label[i]: p.item() for i, p in enumerate(probs[0])}
        }

# Основной пайплайн
def main():
    # Загрузка данных
    (train_texts, val_texts, train_labels, val_labels), id2label = load_data(MARKED_PATH)
    
    # Инициализация модели
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(id2label),
        id2label=id2label,
        label2id={v: k for k, v in id2label.items()}
    )
    
    # Создание DataLoader
    train_dataset = MedicalDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    val_dataset = MedicalDataset(val_texts, val_labels, tokenizer, MAX_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=2)
    
    # Настройка устройства и оптимизатора
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, correct_bias=False)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=EPOCHS)
    
    # Цикл обучения
    best_acc = 0
    for epoch in range(EPOCHS):
        print(f'\nEpoch {epoch + 1}/{EPOCHS}')
        train_loss = train_epoch(model, train_loader, optimizer, device, scheduler)
        val_acc = eval_model(model, val_loader, device, id2label)
        
        # Сохранение лучшей модели
        if val_acc > best_acc:
            best_acc = val_acc
            model.save_pretrained(SAVE_PATH)
            tokenizer.save_pretrained(SAVE_PATH)
            print(f"New best model saved with accuracy {val_acc:.4f}")
        
        print(f'Train Loss: {train_loss:.4f} | Val Accuracy: {val_acc:.4f}')
    
    print("\nTraining completed!")

if __name__ == '__main__':
    main()

# Пример использования после обучения:
# classifier = MedicalClassifier(SAVE_PATH, 'label2id.json')
# result = classifier.predict('Пациент жалуется на нарушение зрения')
# print(result)

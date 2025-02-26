import pandas as pd
import torch, json, os
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Конфигурация
MODEL_NAME = 'bert-base-uncased'
BATCH_SIZE = 16
MAX_LEN = 128
EPOCHS = 5
LEARNING_RATE = 2e-5

# 1. Загрузка и подготовка данных
def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    texts = df['текст'].tolist()
    labels = df['класс'].tolist()
    
    # Преобразование меток в числовой формат
    unique_labels = list(set(labels))
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    labels = [label2id[label] for label in labels]
    
    return train_test_split(texts, labels, test_size=0.2, random_state=42), label2id

# 2. Создание Dataset
class TextDataset(Dataset):
    def init(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def len(self):
        return len(self.texts)
    
    def getitem(self, idx):
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
def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
    
    return total_loss / len(dataloader)

# 4. Валидация
def evaluate(model, dataloader, device):
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    return accuracy_score(true_labels, predictions)

# Основной пайплайн
def main():
    # Загрузка данных
    (train_texts, val_texts, train_labels, val_labels), label2id = load_data('marked.json')
    
    # Инициализация токенизатора и модели
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id))
    
    # Создание DataLoader
    train_dataset = TextDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer, MAX_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Настройка устройства и оптимизатора


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Цикл обучения
    for epoch in range(EPOCHS):
        train_loss = train(model, train_loader, optimizer, device)
        val_acc = evaluate(model, val_loader, device)
        
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print(f'Train loss: {train_loss:.4f}')
        print(f'Validation accuracy: {val_acc:.4f}\n')
    
    # Сохранение модели
    model.save_pretrained('bert-text-classifier')
    tokenizer.save_pretrained('bert-text-classifier')

if 'name' == 'main':
    main()
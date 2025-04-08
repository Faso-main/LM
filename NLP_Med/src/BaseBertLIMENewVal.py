import pandas as pd
import torch, json, os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import psutil
from lime.lime_text import LimeTextExplainer
from sklearn.utils.class_weight import compute_class_weight

# Конфигурация
MODEL_NAME = 'bert-base-uncased'
BATCH_SIZE = 16
MAX_LEN = 256
EPOCHS = 20
LEARNING_RATE = 3e-5
WARMUP_STEPS = 100

# Пути
SAVE_PATH = os.path.join('NLP_Med', 'trained', f'fake_{MODEL_NAME}_{EPOCHS}ep')
MARKED_PATH = os.path.join('NLP_Med', 'src', 'fake_marked.json')
LABLE_PATH = os.path.join('NLP_Med', 'src', 'label2id.json')
RESULTS_PATH = os.path.join('NLP_Med', 'src', 'results', f'fake_{MODEL_NAME}_{EPOCHS}ep.csv')
RESULTS_DIR_PATH = os.path.join('NLP_Med', 'src', 'results')
IMG_PATH = os.path.join('NLP_Med', 'src', 'results', f'fake_plot_{MODEL_NAME}_{EPOCHS}ep.png')

# 1. Улучшенная загрузка данных с обработкой редких классов
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    texts = df['текст'].tolist()
    labels = df['классификация'].tolist()
    
    # Анализ распределения классов
    label_counts = pd.Series(labels).value_counts()
    print("Распределение классов до обработки:")
    print(label_counts)
    
    # Удаляем классы с менее чем 2 примерами
    valid_labels = label_counts[label_counts >= 2].index
    filtered_data = [(text, label) for text, label in zip(texts, labels) if label in valid_labels]
    
    if len(filtered_data) < len(texts):
        removed = len(texts) - len(filtered_data)
        print(f"\nУдалено {removed} примеров из классов с недостаточным количеством образцов")
    
    texts = [item[0] for item in filtered_data]
    labels = [item[1] for item in filtered_data]
    
    # Проверяем после фильтрации
    label_counts = pd.Series(labels).value_counts()
    print("\nРаспределение классов после обработки:")
    print(label_counts)
    
    # Преобразование меток
    unique_labels = sorted(list(set(labels)))
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    labels = [label2id[label] for label in labels]
    
    # Сохранение меток
    with open(LABLE_PATH, 'w', encoding='utf-8') as f:
        json.dump(label2id, f, ensure_ascii=False)
    
    # Проверяем, что все классы имеют достаточное количество примеров
    min_samples = min(pd.Series(labels).value_counts())
    if min_samples < 2:
        raise ValueError("После фильтрации остались классы с менее чем 2 примерами. Необходимо больше данных.")
    
    # Стратифицированное разделение
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels)
    
    return (train_texts, val_texts, train_labels, val_labels), id2label, label2id

# 2. Датасет
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
            return_tensors='pt',
            truncation_strategy='longest_first'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 3. Обучение с учетом весов классов
def train_epoch(model, dataloader, optimizer, device, scheduler, class_weights=None):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc='Training', leave=False)
    
    if class_weights is not None:
        class_weights = class_weights.to(device)
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
    else:
        loss_fct = torch.nn.CrossEntropyLoss()
    
    for batch in progress_bar:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = loss_fct(outputs.logits.view(-1, model.config.num_labels), labels.view(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
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
    
    # Подробный отчет
    print("\nDetailed Classification Report:")
    report = classification_report(
        true_labels, 
        predictions, 
        target_names=list(id2label.values()),
        zero_division=0,
        output_dict=True
    )
    print(classification_report(
        true_labels, 
        predictions, 
        target_names=list(id2label.values()),
        zero_division=0
    ))
    
    # Метрики
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')
    recall = recall_score(true_labels, predictions, average='weighted')
    memory_usage = psutil.Process().memory_info().rss / 1024 ** 2
    
    return accuracy, f1, recall, memory_usage

# Функция для предсказаний (для LIME)
def predict_proba(texts, model, tokenizer, device, id2label):
    inputs = tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits.detach().cpu().numpy()
    # Преобразование logits в вероятности с помощью softmax
    probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    return probabilities

# Функция для LIME объяснений
def explain_with_lime(model, tokenizer, device, id2label, text, num_features=10, num_samples=5000):
    explainer = LimeTextExplainer(class_names=list(id2label.values()))
    
    # Функция-обертка для предсказаний
    def classifier_fn(texts):
        return predict_proba(texts, model, tokenizer, device, id2label)
    
    # Генерация объяснения
    explanation = explainer.explain_instance(
        text, 
        classifier_fn, 
        num_features=num_features, 
        num_samples=num_samples
    )
    
    return explanation

# Основной пайплайн
def main():
    # Загрузка данных
    try:
        (train_texts, val_texts, train_labels, val_labels), id2label, label2id = load_data(MARKED_PATH)
    except ValueError as e:
        print(f"Ошибка загрузки данных: {e}")
        return
    
    # Вычисление весов классов
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    
    # Инициализация модели
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id
    )
    
    # Даталоадеры
    train_dataset = MedicalDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    val_dataset = MedicalDataset(val_texts, val_labels, tokenizer, MAX_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0)
    
    # Устройство и оптимизатор
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.gradient_checkpointing_enable()
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, correct_bias=False)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps
    )
    
    # Обучение
    best_f1 = 0
    results = []
    
    for epoch in range(EPOCHS):
        print(f'\nEpoch {epoch + 1}/{EPOCHS}')
        start_time = time.time()
        
        train_loss = train_epoch(
            model, train_loader, optimizer, device, 
            scheduler, class_weights
        )
        
        val_acc, f1, recall, memory_usage = eval_model(
            model, val_loader, device, id2label
        )
        
        epoch_time = time.time() - start_time
        
        results.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_accuracy': val_acc,
            'f1_score': f1,
            'recall': recall,
            'epoch_time': epoch_time,
            'memory_usage': memory_usage
        })
        
        # Сохранение лучшей модели по F1-score
        if f1 > best_f1:
            best_f1 = f1
            model.save_pretrained(SAVE_PATH)
            tokenizer.save_pretrained(SAVE_PATH)
            print(f"New best model saved with F1-score {f1:.4f}")
        
        print(f'Epoch {epoch + 1} results:')
        print(f'Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | F1: {f1:.4f} | Recall: {recall:.4f}')
        print(f'Time: {epoch_time:.2f}s | Memory: {memory_usage:.2f}MB')
    
    # Сохранение результатов
    results_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"\nTraining completed! Results saved to {RESULTS_PATH}")
    
    # Визуализация
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(results_df['epoch'], results_df['train_loss'], label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(results_df['epoch'], results_df['val_accuracy'], label='Val Accuracy')
    plt.plot(results_df['epoch'], results_df['f1_score'], label='F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    
    plt.tight_layout()
    os.makedirs(RESULTS_DIR_PATH, exist_ok=True)
    plt.savefig(IMG_PATH)
    plt.show()
    
    # LIME объяснения только если модель достаточно хороша
    if best_f1 > 0.5:  # Порог качества
        try:
            print("\nГенерация объяснений с помощью LIME...")
            
            # Выбираем несколько случайных примеров из валидационного набора
            sample_indices = np.random.choice(len(val_texts), size=min(3, len(val_texts)), replace=False)
            
            for i, idx in enumerate(sample_indices):
                text = val_texts[idx]
                true_label = id2label[val_labels[idx]]
                
                # Получаем предсказание модели
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                predicted_label = id2label[torch.argmax(outputs.logits).item()]
                
                print(f"\nПример {i+1}:")
                print(f"Текст: {text[:200]}...")
                print(f"Истинный класс: {true_label}")
                print(f"Предсказанный класс: {predicted_label}")
                
                # Генерация объяснения
                explanation = explain_with_lime(
                    model, tokenizer, device, id2label, text,
                    num_features=10, num_samples=1000
                )
                
                # Визуализация объяснения
                """
                Выдаёт ошибку Ipyhon. Не смог исправить
                print("\nОбъяснение (важные слова для предсказания):")
                explanation.show_in_notebook(text=True)
                """

                
                # Сохранение объяснения в HTML                                                          
                html_path = os.path.join('NLP_Med', 'src', 'results', 'graph_visualization', f'lime_explanation_{i+1}.html')
                explanation.save_to_file(html_path)
                print(f"Объяснение сохранено в {html_path}")
                
        except Exception as e:
            print(f"\nОшибка при генерации LIME объяснений: {str(e)}")
    else:
        print("\nСлишком низкое качество модели (F1-score <= 0.5)")

if __name__ == '__main__':
    main()
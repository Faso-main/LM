import spacy
import re
import os
import torch
from transformers import BertTokenizer, BertModel
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)

class EDSSCalculator:
    """
    Класс для расчета шкалы EDSS на основе извлеченных данных.
    """
    def __init__(self):
        """
        Инициализация словаря для хранения баллов по различным категориям.
        """
        self.scores = {key: 0 for key in [
            'mobility', 'sensory', 'visual', 'bladder_bowel', 'cognitive', 
            'motor', 'cerebellar', 'speech', 'mental_state', 'fatigue'
        ]}

    def assess(self, category, level):
        """
        Оценка симптомов по категориям и уровням тяжести.
        """
        scoring = {
            'sensory': {'normal': 1, 'mild': 2, 'moderate': 3, 'severe': 4},
            'visual': self.visual_assessment(level),
            'bladder_bowel': {'normal': 1, 'mild_incontinence': 2, 'severe_incontinence': 3},
            'cognitive': {'normal': 1, 'mild': 2, 'moderate': 3, 'severe': 4},
            'motor': {'normal': 1, 'mild': 2, 'moderate': 3, 'severe': 4},
            'cerebellar': {'normal': 1, 'mild': 2, 'severe': 3},
            'speech': {'normal': 1, 'mild': 2, 'moderate': 3, 'severe': 4},
            'mental_state': {'normal': 1, 'mild': 2, 'moderate': 3, 'severe': 4},
            'fatigue': {'none': 1, 'mild': 2, 'moderate': 3, 'severe': 4},
        }
        
        if category in self.scores:
            if category == 'visual' and isinstance(level, tuple):
                self.scores[category] = scoring[category]
            else:
                self.scores[category] = scoring[category].get(level, 4)

    def visual_assessment(self, acuity):
        """
        Оценка остроты зрения.
        """
        if acuity is None:
            return 4  # Высокий балл, если данные отсутствуют
        if isinstance(acuity, tuple) and len(acuity) == 2:
            if acuity[0] >= 1.0 and acuity[1] >= 1.0:
                return 1
            elif acuity[0] >= 0.7 or acuity[1] >= 0.7:
                return 2
            else:
                return 3
        return 4  # Если данные некорректны

    def calculate_edss(self):
        """
        Расчет шкалы EDSS на основе накопленных баллов.
        """
        # Пример более сложной логики
        if self.scores['motor'] >= 3 and self.scores['sensory'] >= 3:
            return 6.0  # Тяжелая форма
        elif self.scores['visual'] >= 2:
            return 4.0  # Умеренная форма
        else:
            return sum(self.scores.values()) / len(self.scores)  # Усредненный балл

# Загрузка моделей и инициализация токенизатора
nlp = spacy.load('ru_core_news_md')
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

class TextExtractor:
    """
    Класс для извлечения информации из медицинских текстов.
    """
    def __init__(self):
        self.extractors = [
            self.extract_visual_acuity,
            self.extract_motor_strength,
            self.extract_sensory_feedback,
            self.extract_bladder_bowel_function,
            self.extract_cognitive_feedback,
            self.extract_fatigue,
            self.extract_speech_condition,
            self.extract_mental_state,
            self.extract_symptoms_onset,
            self.extract_cerebellar_symptoms,  # Новый метод
        ]

    def extract_information(self, text):
        """
        Извлечение информации из текста с использованием всех методов.
        """
        results = {}
        doc = nlp(text)
        for extractor in self.extractors:
            results.update(extractor(doc))
        return results

    def extract_visual_acuity(self, doc):
        """
        Извлечение данных о остроте зрения.
        """
        for sent in doc.sents:
            if "острота зрения" in sent.text:
                match = re.search(r'OD=(\S+);\s*OS=(\S+)', sent.text)
                if match:
                    try:
                        acuity_left = float(match.group(1).replace(',', '.'))
                        acuity_right = float(match.group(2).replace(',', '.'))
                        return {'visual_acuity': (acuity_left, acuity_right)}
                    except ValueError:
                        logging.warning("Некорректные данные остроты зрения")
        return {'visual_acuity': (0.0, 0.0)}  # Значение по умолчанию

    def extract_motor_strength(self, doc):
        """
        Извлечение данных о двигательной силе.
        """
        for sent in doc.sents:
            if "слабость" in sent.text or "неустойчивость" in sent.text:
                return {'motor_strength': 'severe'}
        return {'motor_strength': None}

    def extract_sensory_feedback(self, doc):
        """
        Извлечение данных о сенсорных нарушениях.
        """
        for sent in doc.sents:
            if "онемение" in sent.text or "покалывание" in sent.text:
                return {'sensory_feedback': 'mild'}
        return {'sensory_feedback': None}

    def extract_bladder_bowel_function(self, doc):
        """
        Извлечение данных о функции мочевого пузыря и кишечника.
        """
        for sent in doc.sents:
            if "недержание" in sent.text or "частые позывы" in sent.text:
                return {'bladder_bowel_function': 'mild_incontinence'}
        return {'bladder_bowel_function': None}

    def extract_cognitive_feedback(self, doc):
        """
        Извлечение данных о когнитивных нарушениях.
        """
        for sent in doc.sents:
            if "головокружение" in sent.text or "снижение памяти" in sent.text:
                return {'cognitive_feedback': 'mild'}
        return {'cognitive_feedback': None}

    def extract_fatigue(self, doc):
        """
        Извлечение данных об утомляемости.
        """
        for sent in doc.sents:
            if "утомляемость" in sent.text:
                return {'fatigue': 'moderate'}
        return {'fatigue': None}

    def extract_speech_condition(self, doc):
        """
        Извлечение данных о нарушениях речи.
        """
        for sent in doc.sents:
            if "дизартрия" in sent.text or "затруднение речи" in sent.text:
                return {'speech_condition': 'moderate'}
        return {'speech_condition': None}

    def extract_mental_state(self, doc):
        """
        Извлечение данных о психическом состоянии.
        """
        for sent in doc.sents:
            if "депрессия" in sent.text or "тревожность" in sent.text:
                return {'mental_state': 'moderate'}
        return {'mental_state': None}

    def extract_symptoms_onset(self, doc):
        """
        Извлечение данных о начале симптомов.
        """
        for sent in doc.sents:
            if "заболела" in sent.text or "появилось" in sent.text:
                return {'symptoms_onset': True}
        return {'symptoms_onset': False}

    def extract_cerebellar_symptoms(self, doc):
        """
        Извлечение данных о мозжечковых симптомах.
        """
        for sent in doc.sents:
            if "нистагм" in sent.text or "атаксия" in sent.text:
                return {'cerebellar_symptoms': 'severe'}
        return {'cerebellar_symptoms': None}

def read_text_file(file_path):
    """
    Чтение текстового файла.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        logging.error(f"Ошибка при чтении файла: {e}")
        return ""

def evaluate_model(text):
    """
    Оценка текста с использованием модели BERT.
    """
    model = BertModel.from_pretrained('bert-base-multilingual-cased')
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Используем усредненный эмбеддинг
    return outputs.last_hidden_state.mean(dim=1)

def main(user_id: str) -> None:
    """
    Основная функция для обработки данных и расчета EDSS.
    """
    file_path = os.path.join('text', f"{user_id}.txt")
    clinical_text = read_text_file(file_path)

    if not clinical_text:
        return

    # Извлечение данных из текста
    results = TextExtractor().extract_information(clinical_text)
    logging.info("Извлеченные данные: %s", results)

    # Расчет EDSS
    edss_calculator = EDSSCalculator()
    
    if results['visual_acuity'] is not None:
        edss_calculator.assess('visual', results['visual_acuity'])
    
    if results['sensory_feedback'] is not None:
        edss_calculator.assess('sensory', results['sensory_feedback'])

    if results['bladder_bowel_function'] is not None:
        edss_calculator.assess('bladder_bowel', results['bladder_bowel_function'])

    if results['cognitive_feedback'] is not None:
        edss_calculator.assess('cognitive', results['cognitive_feedback'])

    if results['motor_strength'] is not None:
        edss_calculator.assess('motor', results['motor_strength'])

    if results['speech_condition'] is not None:
        edss_calculator.assess('speech', results['speech_condition'])

    if results['mental_state'] is not None:
        edss_calculator.assess('mental_state', results['mental_state'])

    if results['fatigue'] is not None:
        edss_calculator.assess('fatigue', results['fatigue'])

    if results['cerebellar_symptoms'] is not None:
        edss_calculator.assess('cerebellar', results['cerebellar_symptoms'])

    # Расчет и вывод EDSS
    edss_score = edss_calculator.calculate_edss()
    logging.info(f'Рассчитанный уровень EDSS: {edss_score:.1f}')

    # Оценка текста с использованием BERT
    bert_output = evaluate_model(clinical_text)
    logging.info(f"BERT эмбеддинг: {bert_output}")

if __name__ == "__main__":
    main("551")
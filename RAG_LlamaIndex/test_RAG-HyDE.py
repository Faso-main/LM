import os
import torch
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from transformers import BitsAndBytesConfig, AutoTokenizer
from llama_index.readers.file import UnstructuredReader
from llama_index.core.indices.query.query_transform.base import (
    HyDEQueryTransform,
)
from llama_index.core.query_engine import TransformQueryEngine
from llama_index.core.query_engine import BaseQueryEngine 

# 1. Квантизация модели. Можно убрать, если библиотека не скачивается. Тогда нужно ставить load_in_4bit в model_kwargs
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Используется устройство: {DEVICE}")

model_kwargs = {}
if DEVICE == "cuda":
    compute_dtype = torch.bfloat16 
    try:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4", 
            bnb_4bit_use_double_quant=True # Дополнительная квантизация для повышения точности
        )
        model_kwargs["quantization_config"] = quantization_config
        print("Настройки 4-битной квантизации применены.")
    except ImportError:
        print("Ошибка: Библиотека 'bitsandbytes' не найдена. 4-битная квантизация отключена.")
        model_kwargs["torch_dtype"] = compute_dtype
    # Дополнительные аргументы для ускорения или оптимизации памяти на GPU
    model_kwargs["torch_dtype"] = compute_dtype
    model_kwargs["low_cpu_mem_usage"] = True # Уменьшение потребления CPU при загрузке
else: # DEVICE == "cpu"
    print("CUDA не доступна. Модель LLM будет загружена на CPU без 4-битной квантизации (если поддерживается моделью).")
    model_kwargs["torch_dtype"] = torch.float32 # Или оставьте по умолчанию
    model_kwargs["low_cpu_mem_usage"] = True # Может помочь с большими моделями на CPU

# Убедитесь, что токенизатор LLM имеет паддинг токен, если модель используется в батчах (например, для HyDE)
llm_tokenizer_kwargs = {}
try:
    llm_tokenizer = AutoTokenizer.from_pretrained("NousResearch/DeepHermes-3-Llama-3-3B-Preview")
    if llm_tokenizer.pad_token is None:
        # Llama-3 токенизаторы часто используют eos_token (<|end_of_text|>) как pad_token по умолчанию
        if llm_tokenizer.eos_token is not None:
             llm_tokenizer_kwargs['pad_token'] = llm_tokenizer.eos_token
             print(f"LLM токенизатор не имеет pad_token, присвоен eos_token: {llm_tokenizer.eos_token}")
        else:
             print("LLM токенизатор не имеет pad_token и eos_token. Могут возникнуть проблемы с батчевой обработкой.")

    else:
        print(f"LLM токенизатор имеет pad_token: {llm_tokenizer.pad_token}")
except Exception as e_tok_llm:
    print(f"Не удалось загрузить LLM токенизатор для проверки pad_token: {e_tok_llm}. Продолжаем без явной настройки pad_token.")
    llm_tokenizer_kwargs = {} 


# 2. Инициализация модели LLM с исправленными параметрами
try:
    Settings.llm = HuggingFaceLLM(
        model_name="NousResearch/DeepHermes-3-Llama-3-3B-Preview",
        tokenizer=llm_tokenizer, # Передаем загруженный токенизатор или None
        context_window=8500, # Контекстное окно модели 
        max_new_tokens=512, # Максимальное количество генерируемых токенов
        model_kwargs=model_kwargs, # Применяем настройки квантизации и dtype
        generate_kwargs={"temperature": 0.6,
                         "do_sample": True, # Включить сэмплирование при temperature > 0
                         "top_p": 0.99, # Отсечение токенов по вероятности
                         "top_k": 20}, # Отсечение токенов по K наиболее вероятным
        device_map="auto" if DEVICE == "cuda" else None, # Автоматическое распределение на GPU, если доступно
        system_prompt=( #Первая строка заставляет включать думающий режим в ИИ
        """
        Вы - эксперт по документам ГОСТ. Отвечайте на вопросы, используя только информацию 
        из предоставленных документов. Игнорируйте любые заголовки глав, разделов, подразделов, 
        номера страниц, колонтитулы, оглавление, ссылки или списки литературы, если они встречаются в тексте документов. 
        Сосредоточьтесь только на основном тексте параграфов. Отвечайте строго на русском языке. 
        Представьте ответ структурированно, в виде списка или последовательности шагов, без повторений. 
        Не пишите никаких вступлений, заключений или пояснений в начале или конце ответа.
        Описывай всё коротко. 
        "Answer:" или "Ответ:" писать нельзя. 
        Пиши сразу краткий ответ на вопрос
        """
        ),
    )
    print("LLM успешно загружена.")
except Exception as e:
    print(f"Ошибка при загрузке LLM: {e}")
    print("Проверьте установку transformers, accelerate, bitsandbytes. Убедитесь, что модель доступна.")

# 3. Загрузка эмбеддингов с обработкой ошибок и настройкой токенизатора
try:
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        model_kwargs=model_kwargs,
        device=DEVICE,
    )
    print("Модель эмбеддингов успешно загружена.")
except Exception as e:
    print(f"Ошибка при загрузке модели эмбеддингов: {e}") 
    print("Проверьте установку sentence-transformers или других зависимостей.")

# 4. Настройка парсера нод (чанков)
node_parser = SentenceSplitter(
    chunk_size=1024, # Размер 1 чанка. Чем больше размер, тем меньше чанков, но будет требоваться больше контекстного окна
    chunk_overlap=128, # На сколько 1 чанк заходит на другой
    include_metadata=True, # Включение метаданных в чанк
    paragraph_separator="\n\n"
)

Settings.transformations = [node_parser] 

# 5. Загрузка документов
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "data_GOST")
print(f"Поиск документов в папке: {DATA_PATH}")
if not os.path.exists(DATA_PATH):
    print(f"Ошибка: Папка с документами не найдена: {DATA_PATH}")
    print("Пожалуйста, создайте папку 'data_GOST' в том же каталоге, что и скрипт, и поместите туда PDF-файлы.")
    exit()

documents = [] 
try:
    # Используем SimpleDirectoryReader для сканирования папки
    loader = SimpleDirectoryReader(
        input_dir=DATA_PATH,
        file_extractor={".pdf": UnstructuredReader()},
        recursive=True, 
    )
    print("Загрузка документов...")
    documents = loader.load_data()
    print(f"Загружено {len(documents)} документов.")
    for doc in documents:
         if 'filename' in doc.metadata:
              doc.metadata['source'] = os.path.basename(doc.metadata['filename'])
              doc.metadata['doc_type'] = "ГОСТ"
         else:
              pass 

except Exception as e:
     print(f"Ошибка при загрузке документов из папки '{DATA_PATH}': {e}")
     print("Проверьте установку unstructured, pdfminer.six и других зависимостей для PDF.")
     documents = [] 

if not documents:
    print(f"Предупреждение: В папке '{DATA_PATH}' не найдено документов или не удалось их обработать. Индекс будет пустым.")


# 6. Создание или загрузка индекса 
index = None 
if documents:
    print(f"Создание нового индекса в памяти из загруженных документов...")
    try:
        index = VectorStoreIndex.from_documents(
            documents,
            show_progress=True # Показываем прогресс индексации
        )
        print("Индекс успешно создан в памяти.")
    except Exception as e:
        print(f"Ошибка при создании индекса: {e}")
        print("Индекс не будет создан.")
        index = None
else:
    print("Документы для создания индекса отсутствуют.")

# 7. Настройка движка запросов 
query_engine = None
if index: 
    print("Настройка Query Engine с использованием HyDE transform...")
    try:
        # Создаем HyDE transform
        hyde_transform = HyDEQueryTransform(include_original=False)
        # Получаем базовый query engine из индекса
        base_query_engine = index.as_query_engine(
            similarity_top_k=7 # Количество документов, которое базовый ретривер будет возвращать после каждого поиска (по гипотетическому документу и по оригинальному запросу)
        )
        # Оборачиваем базовый движок в TransformQueryEngine с HyDE
        query_engine = TransformQueryEngine(base_query_engine, query_transform=hyde_transform)

        print("Query Engine с HyDE успешно настроен.")
    except Exception as e:
        print(f"Ошибка при создании query engine с HyDE: {e}")
        print("Query engine не будет создан.")
        query_engine = None
else:
    print("Query engine не может быть создан, так как индекс отсутствует.")


# 8. Оптимизированный запрос
query = (
    """
    Что означает термин 'угроза безопасности информации'
    """
)

# 9. Выполнение запроса с обработкой ошибок и выводом уверенности
if query_engine:
    try:
        print(f"\nВыполнение запроса (через Query Engine с HyDE transform):\n'{query}'")
        # HyDE transform происходит внутри .query(), он сам вызовет LLM для генерации гипотетического документа,
        # затем вызовет базовый ретривер для поиска по этому гипотетическому документу (и оригиналу, если include_original=True),
        # и наконец, передаст найденные ноды в базовый синтезатор ответа (использующий LLM) для генерации финального ответа.
        response = query_engine.query(query)
        # Извлекаем чистый текст ответа от LLM
        # LlamaIndex может добавлять метаинформацию или источники после маркера
        response_text_full = str(response)
        source_delimiter = "---------------------\n" # Это стандартный разделитель в LlamaIndex 
        if source_delimiter in response_text_full:
            response_text = response_text_full.split(source_delimiter, 1)[0].strip()
        else:
            response_text = response_text_full.strip()
        print("\nОтвет:")
        print(response_text)
        # Вывод источников с оценкой сходства (уверенностью)
        # TransformQueryEngine передает source_nodes от базового query engine
        if hasattr(response, 'source_nodes') and response.source_nodes:
            print("\nИсточники (с оценкой сходства):")
            # Сортируем ноды по оценке сходства в убывающем порядке
            sorted_source_nodes = sorted(response.source_nodes, key=lambda node: node.score if node.score is not None else -1, reverse=True)
            # Выводим информацию для каждого использованного чанка
            for i, node in enumerate(sorted_source_nodes):
                source_name = node.node.metadata.get('source', 'Неизвестный источник')
                similarity_score = node.score
                # Убедимся, что content существует перед взятием превью
                node_content = node.get_content()
                # Обрезаем текст и заменяем переносы строк для компактного вывода
                text_preview = (node_content[:200].replace('\n', ' ') + '...') if node_content else "Пустой чанк"
                score_info = f" (Оценка: {similarity_score:.4f})" if similarity_score is not None else ""
                print(f"  {i+1}. Источник: {source_name}{score_info}\n     Чанк: \"{text_preview}\"")
        else:
            print("\nИсточники не найдены для этого ответа.")
    except Exception as e:
        print(f"\nПроизошла ошибка во время выполнения запроса: {str(e)}")
        # Попытка очистить кэш CUDA в случае ошибки
        if torch.cuda.is_available():
            print("Попытка очистить кэш CUDA...")
            torch.cuda.empty_cache()
            print("Кэш CUDA очищен.")

else:
    print("\nQuery engine не был создан из-за предыдущих ошибок (отсутствие документов, ошибка загрузки моделей или создания индекса).")
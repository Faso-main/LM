import os
import torch
from llama_index.core import SimpleDirectoryReader, Settings, Document
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from transformers import BitsAndBytesConfig, AutoTokenizer
from llama_index.readers.file import UnstructuredReader

# Импорты для RAPTOR
# Мы будем использовать RaptorPack
from llama_index.packs.raptor import RaptorPack
from llama_index.core.query_engine import RetrieverQueryEngine # Используется с кастомными ретриверами

# Импорты для кластеризации/векторного хранения, которые RaptorPack использует внутри
# Убедитесь, что установлены: pip install scikit-learn faiss-cpu llama-index-packs-raptor llama-index-vector-stores-chroma llama-index-embeddings-huggingface llama-index-llms-huggingface
# (Или faiss-gpu если есть CUDA)
# llama-index-vector-stores-chroma может понадобиться, т.к. RaptorPack по умолчанию использует ChromaDB in-memory

# Убедимся, что нужный тип query engine импортирован
from llama_index.core.query_engine import BaseQueryEngine # Для проверки типа

# 1. Квантизация модели и настройка устройств
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Используется устройство: {DEVICE}")

quantization_config = None
model_kwargs = {}
if DEVICE == "cuda":
    compute_dtype = torch.float16 # Используем float16 на GPU для эффективности
    try:
        # Попытка импортировать BitsAndBytesConfig
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        model_kwargs["quantization_config"] = quantization_config
        print("Настройки 4-битной квантизации применены.")
    except ImportError:
        print("Ошибка: Библиотека 'bitsandbytes' не найдена. 4-битная квантизация отключена.")
        quantization_config = None
        model_kwargs["torch_dtype"] = compute_dtype # На GPU все равно лучше использовать float16

    model_kwargs["torch_dtype"] = compute_dtype
    model_kwargs["low_cpu_mem_usage"] = True
else: # DEVICE == "cpu"
    print("CUDA не доступна. Модель LLM будет загружена на CPU без 4-битной квантизации.")
    model_kwargs["torch_dtype"] = torch.float32
    model_kwargs["low_cpu_mem_usage"] = True

# Настройка токенизатора LLM
llm_tokenizer_kwargs = {}
llm_tokenizer = None
llm_model_name = "NousResearch/DeepHermes-3-Llama-3-3B-Preview" # Вынесено для удобства
try:
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    if llm_tokenizer.pad_token is None:
        if llm_tokenizer.eos_token is not None:
             # Llama-3 токенизаторы часто используют eos_token (<|end_of_text|>) как pad_token по умолчанию
             llm_tokenizer.pad_token = llm_tokenizer.eos_token # Устанавливаем в объект токенизатора
             # print(f"LLM токенизатор не имеет pad_token, присвоен eos_token: {llm_tokenizer.eos_token}")
        else:
             print("LLM токенизатор не имеет pad_token и eos_token. Могут возникнуть проблемы.")
    # else:
        # print(f"LLM токенизатор имеет pad_token: {llm_tokenizer.pad_token}")
    # Не передаем tokenizer_kwargs явно в HuggingFaceLLM, т.к. он ожидает объект tokenizer
except Exception as e_tok_llm:
    print(f"Не удалось загрузить LLM токенизатор для проверки/настройки pad_token: {e_tok_llm}. Продолжаем без явной настройки pad_token.")
    llm_tokenizer = None # Сбрасываем на всякий случай


# 2. Инициализация модели LLM
Settings.llm = None
try:
    if quantization_config is None and "quantization_config" in model_kwargs:
        del model_kwargs["quantization_config"]
        print("quantization_config удален из model_kwargs.")

    Settings.llm = HuggingFaceLLM(
        model_name=llm_model_name,
        tokenizer=llm_tokenizer, # Передаем загруженный токенизатор или None
        context_window=8192,
        max_new_tokens=2048,
        model_kwargs=model_kwargs,
        generate_kwargs={"temperature": 0.6, "do_sample": True, "top_p": 0.95, "top_k": 20},
        device_map="auto" if DEVICE == "cuda" else None,
        system_prompt=(
        """
        Вы - эксперт по документам ГОСТ. Отвечайте на вопросы, используя только информацию
        из предоставленных документов. Игнорируйте любые заголовки глав, разделов, подразделов,
        номера страниц, колонтитулы, оглавление или списки литературы, если они встречаются в тексте документов.
        Сосредоточьтесь только на основном тексте параграфов. Отвечайте строго на русском языке.
        Представьте ответ структурированно, в виде списка или последовательности шагов, без повторений.
        Не пишите никаких вступлений, заключений или пояснений в начале или конце ответа.
        Описывай всё коротко.
        """
        ),
    )
    print("LLM успешно загружена.")
except Exception as e:
    print(f"Ошибка при загрузке LLM: {e}")
    print("Проверьте установку transformers, accelerate, bitsandbytes. Убедитесь, что модель доступна.")
    Settings.llm = None

# 3. Загрузка эмбеддингов
Settings.embed_model = None
try:
    embed_model_name = "Qwen/Qwen3-Embedding-0.6B"
    embedding_tokenizer_kwargs = {}
    embed_tokenizer = None
    try:
        embed_tokenizer = AutoTokenizer.from_pretrained(embed_model_name)
        if embed_tokenizer.pad_token is None and embed_tokenizer.eos_token is not None:
             embed_tokenizer.pad_token = embed_tokenizer.eos_token # Устанавливаем в объект токенизатора
             # print(f"Токенизатор эмбеддинга не имеет pad_token, присвоен eos_token: {embed_tokenizer.eos_token}")
        elif embed_tokenizer.pad_token is not None:
             pass # print(f"Токенизатор эмбеддинга имеет pad_token: {embed_tokenizer.pad_token}")
        else:
             print("Токенизатор эмбеддинга не имеет pad_token и eos_token (unexpected). Могут быть проблемы.")
    except Exception as e_tok_emb:
        print(f"Не удалось загрузить токенизатор эмбеддинга: {e_tok_emb}. Продолжаем без явной настройки pad_token.")
        embed_tokenizer = None

    Settings.embed_model = HuggingFaceEmbedding(
        model_name=embed_model_name,
        tokenizer=embed_tokenizer, # Передаем загруженный токенизатор или None
        device=DEVICE,
    )
    print("Модель эмбеддингов успешно загружена.")
except Exception as e:
    print(f"Ошибка при загрузке модели эмбеддингов: {e}")
    print("Проверьте установку sentence-transformers или других зависимостей.")
    Settings.embed_model = None

# Проверка наличия необходимых компонентов перед продолжением
if Settings.llm is None or Settings.embed_model is None:
    print("\n!! Произошли ошибки при загрузке LLM или модели эмбеддингов. Скрипт не может продолжить работу без них. !!\n")
    exit()

# 4. Настройка парсера нод (чанков) - Теперь он используется внутри RaptorPack
# Хотя RaptorPack может иметь свой парсер, лучше настроить его в Settings
# или передать явно, если pack это поддерживает. Для простоты, пусть
# Settings управляет парсером, и Pack, возможно, его подхватит.
# Если Pack не подхватывает Settings.transformations, то нужно будет передавать nodes напрямую
# как в предыдущем варианте. Проверим документацию RaptorPack.
# RaptorPack принимает documents и создает nodes сам, используя парсер из Settings или по умолчанию.
node_parser = SentenceSplitter(
    chunk_size=2048,
    chunk_overlap=128,
    include_metadata=True,
    paragraph_separator="\n\n"
)
Settings.transformations = [node_parser] # Настраиваем парсер в Settings

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
    loader = SimpleDirectoryReader(
        input_dir=DATA_PATH,
        file_extractor={".pdf": UnstructuredReader()},
        recursive=True,
    )
    print("Загрузка документов...")
    documents = loader.load_data()
    print(f"Загружено {len(documents)} документов.")

    # Добавляем метаданные
    for doc in documents:
         if 'filename' in doc.metadata:
              doc.metadata['source'] = os.path.basename(doc.metadata['filename'])
              doc.metadata['doc_type'] = "ГОСТ"
         else:
              # Если 'filename' нет, используем уникальный ID документа как источник
              doc.metadata['source'] = f"doc_{doc.id_}"
              doc.metadata['doc_type'] = "ГОСТ"

except Exception as e:
     print(f"Ошибка при загрузке документов из папки '{DATA_PATH}': {e}")
     print("Проверьте установку unstructured, pdfminer.six и других зависимостей для PDF.")
     documents = []


if not documents:
    print(f"Предупреждение: В папке '{DATA_PATH}' не найдено документов или не удалось их обработать. Невозможно создать RAPTOR.")
    exit() # Выходим, если нет документов для обработки

# 6. Инициализация и запуск RaptorPack
raptor_pack = None
raptor_retriever = None
query_engine = None

print("Инициализация и построение структуры RAPTOR с помощью RaptorPack...")
try:
    # Инициализируем RaptorPack, передавая загруженные документы и настроенные модели
    # RaptorPack сам выполнит парсинг документов в ноды (используя Settings.transformations)
    # построит иерархию (кластеризация + суммирование LLM)
    raptor_pack = RaptorPack(
        documents=documents,
        embed_model=Settings.embed_model, # Модель эмбеддингов для кластеризации и поиска
        llm=Settings.llm, # LLM для суммирования кластеров
        # recursive_chunk_size_percentile=0.5, # Процент нод для включения в каждый рекурсивный чанк
        # num_layers=None, # Количество слоев (None = автоматическое)
        # mode="tree_retrieval", # Режим поиска ('tree_retrieval' или 'base')
        # vector_store=..., # Можно передать кастомный vector store
        # cluster_model=..., # Можно передать кастомную модель кластеризации
        # store_tree=True, # Сохранять структуру дерева (полезно для отладки)
        verbose=True # Выводить информацию о процессе построения дерева
    )

    print("RaptorPack успешно инициализирован. Структура RAPTOR построена.")

    # Получаем ретривер из пака
    raptor_retriever = raptor_pack.retriever
    print("RaptorRetriever получен из RaptorPack.")

    # 7. Настройка движка запросов (Query Engine) с использованием RaptorRetriever
    print("Настройка Query Engine с использованием RaptorRetriever...")
    # Используем RetrieverQueryEngine для использования нашего кастомного ретривера
    query_engine = RetrieverQueryEngine(
        retriever=raptor_retriever,
        # ResponseSynthesizer по умолчанию использует Settings.llm
        # response_synthesizer=..., # Можно настроить синтезатор ответа
    )
    print("Query Engine с RaptorRetriever успешно настроен.")

except Exception as e:
    print(f"Ошибка при инициализации RaptorPack или создании Query Engine: {e}")
    print("RAPTOR не будет создан и Query Engine не будет доступен.")
    raptor_pack = None
    raptor_retriever = None
    query_engine = None


# 8. Выполнение запроса
query = (
    """
    Что означает термин 'угроза безопасности информации'
    """
)

if query_engine:
    try:
        print(f"\nВыполнение запроса (через Query Engine с RaptorRetriever):\n'{query}'")
        # Выполнение запроса теперь использует логику RaptorRetriever для поиска по иерархии
        response = query_engine.query(query)

        # Извлекаем чистый текст ответа
        response_text_full = str(response)
        # Standard LlamaIndex delimiter for sources
        source_delimiter = "---------------------\n"
        if source_delimiter in response_text_full:
            response_text = response_text_full.split(source_delimiter, 1)[0].strip()
        else:
            response_text = response_text_full.strip()
        print("\nОтвет:")
        print(response_text)

        # Вывод источников
        # RaptorRetriever может возвращать ноды как базового уровня, так и уровня резюме.
        if hasattr(response, 'source_nodes') and response.source_nodes:
            print("\nИсточники (с оценкой сходства):")
            # Сортируем ноды по оценке сходства в убывающем порядке
            sorted_source_nodes = sorted(response.source_nodes, key=lambda node: node.score if node.score is not None else -1, reverse=True)
            # Выводим информацию для каждого использованного чанка/резюме
            for i, node in enumerate(sorted_source_nodes):
                source_name = node.node.metadata.get('source', 'Неизвестный источник')
                # RaptorPack/Retriever добавляет метаданные для типа ноды
                node_type = node.node.metadata.get('type', 'базовый чанк')
                similarity_score = node.score
                # Убедимся, что content существует перед взятием превью
                node_content = node.get_content()
                # Обрезаем текст и заменяем переносы строк для компактного вывода
                text_preview = (node_content[:200].replace('\n', ' ') + '...') if node_content else "Пустой чанк"
                score_info = f" (Оценка: {similarity_score:.4f})" if similarity_score is not None else ""
                print(f"  {i+1}. Источник: {source_name}, Тип: {node_type}{score_info}\n     Чанк: \"{text_preview}\"")
        else:
            print("\nИсточники не найдены для этого ответа.")

    except Exception as e:
        print(f"\nПроизошла ошибка во время выполнения запроса: {str(e)}")
        if torch.cuda.is_available():
            print("Попытка очистить кэш CUDA...")
            torch.cuda.empty_cache()
            print("Кэш CUDA очищен.")

else:
    print("\nQuery engine не был создан из-за предыдущих ошибок (отсутствие документов, ошибка загрузки моделей или построения RAPTOR).")
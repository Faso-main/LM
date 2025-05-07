from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import (
    SimilarityPostprocessor,
    MetadataReplacementPostProcessor
)
from transformers import BitsAndBytesConfig
import torch
import os

# 1. Настройка квантования (заменяем устаревшие параметры)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

# 2. Инициализация модели с исправленными параметрами
llm = HuggingFaceLLM(
    model_name="mistralai/Mistral-7B-Instruct-v0.3",
    tokenizer_name="mistralai/Mistral-7B-Instruct-v0.3",
    device_map="auto",
    model_kwargs={
        "quantization_config": quantization_config,
        "temperature": 0.3,
        "do_sample": True,  # Добавлено для работы с temperature
        "max_length": 8196,
        "torch_dtype": torch.float16
    },
    generate_kwargs={
        "do_sample": True,
        "temperature": 0.3,
        "top_p": 0.9
    }
)

# 3. Загрузка эмбеддингов с обработкой ошибок
try:
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-mpnet-base-v2",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
except Exception as e:
    print(f"Ошибка загрузки модели эмбеддингов: {str(e)}")
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device="cpu"
    )

# 4. Настройка чанкинга
node_parser = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=128,
    include_metadata=True,
    paragraph_separator="\n\n"
)

# 5. Загрузка документов
documents = SimpleDirectoryReader(
    input_dir=os.path.join(os.path.dirname(__file__), "data_GOST"),
    filename_as_id=True,
    required_exts=[".pdf"],
    file_metadata=lambda filename: {
        "source": os.path.basename(filename),
        "doc_type": "ГОСТ"
    }
).load_data()

# 6. Создание индекса
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=embed_model,
    transformations=[node_parser]
)

# 7. Настройка пост-процессоров
query_engine = index.as_query_engine(
    llm=llm,
    response_mode="compact"
)

# 8. Оптимизированный запрос
query = (
    "Отвечай строго на русском языке. Сформулируй полный перечень требований к техническому заданию. "
    "Ответ должен быть структурированным, без повторений. "
    "Используй только предоставленные документы. "
    "Формат:\n1. Требование 1\n2. Требование 2\n..."
)

# 9. Выполнение запроса с обработкой ошибок
try:
    response = query_engine.query(query)
    if not response or str(response).strip() == "":
        print("Получен пустой ответ. Попробуйте:")
        print("- Увеличить размер чанков (chunk_size)")
        print("- Проверить содержимое PDF-файлов")
    else:
        # Извлекаем чистый текст без метаданных
        response_text = str(response).split('---------------------')[0].strip()
        
        # Удаляем повторяющиеся фрагменты
        lines = response_text.split('\n')
        unique_lines = []
        seen_lines = set()
        
        for line in lines:
            clean_line = line.strip()
            if clean_line and clean_line not in seen_lines:
                seen_lines.add(clean_line)
                unique_lines.append(clean_line)
        
        # Форматируем вывод
        print("\nОтвет:")
        print("\n".join(unique_lines))
        
        # Вывод источников
        if hasattr(response, 'source_nodes'):
            print("\nИсточники:")
            sources = sorted({node.metadata['source'] for node in response.source_nodes})
            for src in sources:
                print(f"- {src}")

except Exception as e:
    print(f"Ошибка: {str(e)}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
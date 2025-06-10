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

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

# 2. Инициализация модели с исправленными параметрами
llm = HuggingFaceLLM(
    model_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
    tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
    context_window=8192, #Лучше не уменьшать
    max_new_tokens=2048, #Лучше не уменьшать
    model_kwargs={"quantization_config": quantization_config},
    generate_kwargs={"temperature": 0.6, #При temperature<0.4 выдаёт странные ответы. При 0.4-0.5 более-менее приличные ответы, но без подробностей. При >0.6 добавляет подробности, но при 0.9 начинает галлюцинировать
                     }, 
    device_map="cuda",
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
    chunk_size=2048, #Размер 1 чанка
    chunk_overlap=256, #На сколько 1 чанк заходит на другой
    include_metadata=True, #Включение метаданных в чанк
    paragraph_separator="\n\n"
)

# 5. Загрузка документов
documents = SimpleDirectoryReader(
    input_dir=os.path.join(os.path.dirname(__file__), "data_GOST"), #Путь к папке
    filename_as_id=True,
    required_exts=[".pdf"], #Какие файлы будут браться
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
    response_mode="compact", # Можно поставить "compact" - быстрый ответ с меньшими запросами к LLM, можно поставить "refine" - делает множество запросов для уточнения. Требует больше ВСЕГО
    similarity_top_k=6 #Сколько чанков рассмотрит ИИ
)

# 8. Оптимизированный запрос
query = (
    """Отвечай строго на русском языке. 
    Сформулируй короткий перечень требований к оформлению технического задания. " 
    Ответ должен быть структурированным, без повторений. 
    Используй только предоставленные документы. 
    Формат:\n1. Требование 1\n2. Требование 2\n..."""
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
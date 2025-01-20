from datetime import date
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains.retrieval_qa.base import RetrievalQA
from transformers import AutoModelForCausalLM
from langchain_community.embeddings import HuggingFaceInstructEmbeddings


# Загрузка языковой модели с huggingface
model = AutoModelForCausalLM.from_pretrained(
  "mistralai/Mistral-7B-v0.1", device_map="auto", load_in_4bit=True
)


"""Определяем размер чанков - длина кусков, на которые мы разбиваем исходный текст и chunk_overlap - пересечение между чанками """
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(date)


# Инициализация эмбеддингов для дальнейшего использования
embeddings  = HuggingFaceInstructEmbeddings(model_name="mistralai/Mistral-7B-v0.1",trust_remote_code=True,
                                     model_kwargs={'device': 'mps'}, encode_kwargs={'device': 'mps'})


# Создание векторной базы данных для хранения текстов и соответствующих им векторных представлений
db = Chroma.from_documents(texts, embeddings)


""" Настройка ретривера (системы поиска по схожести векторных представлений документов)
   Здесь параметр k в search_kwargs отвечает за количество наиболее релевантных документов в выдаче"""
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})


# Создаем llm_chain со встроенным Retrieval из langchain для удобства использования


qa = RetrievalQA.from_chain_type(
 llm=model, chain_type="rag", retriever=retriever, return_source_documents=True)


# Формулировка запроса и получение ответа на вопрос
query = "Где мне искать информацию по инвентаризации?"
result = qa({"query": query})
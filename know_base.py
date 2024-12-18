# Импорты
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import warnings

# Отключаем предупреждения
warnings.filterwarnings("ignore")


# Функция для создания индекса FAISS из текста и сохранения его на диск
def create_faiss_index(text, faiss_index_path):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(model_name="cointegrated/rubert-tiny2")
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    # Создаем директорию, если она не существует
    os.makedirs(os.path.dirname(faiss_index_path), exist_ok=True)
    knowledge_base.save_local(faiss_index_path)
    print(f"FAISS index created and saved to {faiss_index_path}")


# Функция для загрузки индекса FAISS с диска
def load_faiss_index(faiss_index_path, text):
    # Проверка существования индекса FAISS; если не существует — создание
    if not os.path.exists(f"{faiss_index_path}"):  # Индекс FAISS сохраняется с расширением ".faiss"
        # text = extract(pdf_path)
        create_faiss_index(text, faiss_index_path)
    else:
        print(f"FAISS index already exists at {faiss_index_path}")

    embeddings = HuggingFaceEmbeddings(model_name="cointegrated/rubert-tiny2")
    knowledge_base = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)

    return knowledge_base


# Функция для усечения контекста до указанного лимита символов
def truncate_context(context, max_length):
    if len(context) > max_length:
        return context[:max_length] + "..."  # Добавляем многоточие для обозначения усечения
    return context

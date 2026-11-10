from pathlib import Path
from uuid import uuid4
import chromadb
import ollama
from modules.text_chunker import split_text

DB_PATH = str(Path(__file__).resolve().parents[1] / 'chroma_db')
COLLECTION_NAME = 'smartdoc_documents'
EMBEDDING_MODEL = 'embeddinggemma'


def _collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_or_create_collection(name=COLLECTION_NAME, metadata={'hnsw:space': 'cosine'})


def store_document(text, filename):
    chunks = split_text(text)
    if not chunks:
        raise ValueError('Document has no indexable text.')
    embeddings = ollama.embed(model=EMBEDDING_MODEL, input=chunks).embeddings
    collection = _collection()
    collection.add(
        ids=[str(uuid4()) for _ in chunks],
        documents=chunks,
        embeddings=embeddings,
        metadatas=[{'source': filename, 'chunk_number': i + 1} for i in range(len(chunks))],
    )
    return len(chunks)


def document_count():
    return _collection().count()


def clear_database():
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection(COLLECTION_NAME)
    except (ValueError, chromadb.errors.NotFoundError):
        pass
    client.get_or_create_collection(name=COLLECTION_NAME, metadata={'hnsw:space': 'cosine'})


def retrieve_documents(question, top_k=4):
    collection = _collection()
    if collection.count() == 0:
        return []
    query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=question).embeddings[0]
    result = collection.query(query_embeddings=[query_embedding], n_results=min(top_k, collection.count()), include=['documents', 'metadatas', 'distances'])
    return [
        {'text': doc, 'source': meta.get('source', 'Unknown'), 'chunk_number': meta.get('chunk_number', 0), 'distance': distance}
        for doc, meta, distance in zip(result['documents'][0], result['metadatas'][0], result['distances'][0])
    ]

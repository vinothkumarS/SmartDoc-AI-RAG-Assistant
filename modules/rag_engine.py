
from pathlib import Path

import chromadb
import ollama


BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "chroma_db"

COLLECTION_NAME = "smartdoc_documents"
LLM_MODEL = "gemma3:1b"
EMBEDDING_MODEL = "embeddinggemma"

FALLBACK = (
    "I could not find that information in the uploaded documents."
)


SYSTEM_PROMPT = """
You are SmartDoc AI, a precise document question-answering assistant.

RULES:
1. Answer ONLY from the provided document context.
2. Identify the exact subject of the user's question.
3. Never confuse annual leave with sick leave.
4. Never confuse work-from-home days with leave days.
5. Do not answer with unrelated document facts.
6. If the requested information is missing, reply exactly:
I could not find that information in the uploaded documents.
7. Do not invent facts or use external knowledge.
8. Keep answers short, clear, and factual.
9. Include units for numerical answers.

EXAMPLES:

Question: How many annual leave days?
Context: Employees receive 12 days of annual leave.
Answer: Employees receive 12 days of annual leave every year.

Question: How many sick leave days?
Context: Employees receive 10 days of sick leave.
Answer: Employees receive 10 days of sick leave every year.

Question: What is the salary policy?
Context: Employees receive annual leave and sick leave.
Answer: I could not find that information in the uploaded documents.
"""
import re

def find_exact_policy_answer(question, documents):
    q = question.lower()

    sentences = []
    for document in documents:
        sentences.extend(
            re.split(r'(?<=[.!?])\s+', document)
        )

    rules = [
        (
            ["annual leave", "vacation leave"],
            ["annual leave"]
        ),
        (
            ["sick leave"],
            ["sick leave"]
        ),
        (
            ["work from home", "wfh", "remote work"],
            ["work from home"]
        ),
        (
            ["approve leave", "approves leave", "leave approval"],
            ["approve leave", "approve leave requests"]
        ),
        (
            ["health insurance"],
            ["health insurance"]
        ),
    ]

    for question_terms, answer_terms in rules:
        if any(term in q for term in question_terms):
            for sentence in sentences:
                if any(term in sentence.lower()
                       for term in answer_terms):
                    return sentence.strip()

    return None

def generate_answer(question, top_k=4, temperature=0.2):
    question = question.strip()

    if not question:
        return "Please enter a question.", []

    # 1. Connect to persistent ChromaDB
    client = chromadb.PersistentClient(path=str(DB_PATH))

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    total_chunks = collection.count()

    if total_chunks == 0:
        return (
            "Please upload and process a document first.",
            []
        )

    # 2. Generate question embedding
    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=question
    )

    question_vector = response.embeddings[0]

    # 3. Retrieve relevant chunks
    results = collection.query(
        query_embeddings=[question_vector],
        n_results=min(max(1, int(top_k)), total_chunks),
        include=["documents", "metadatas", "distances"]
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    # 4. Remove duplicate content
    unique_documents = []
    sources = []
    seen = set()

    for document, metadata in zip(documents, metadatas):
        if not document or not document.strip():
            continue

        cleaned = document.strip()

        if cleaned in seen:
            continue

        seen.add(cleaned)
        unique_documents.append(cleaned)

        metadata = metadata or {}

        sources.append({
            "source": metadata.get("source", "Unknown"),
            "text": cleaned
        })

    if not unique_documents:
        return FALLBACK, []

    exact_answer = find_exact_policy_answer(
        question, unique_documents
    )

    if exact_answer:
        return exact_answer, sources

    # 5. Build retrieved context
    context = "\n\n".join(
        f"[Document {i}]\n{text}"
        for i, text in enumerate(unique_documents, start=1)
    )

    # 6. Grounded question-answering prompt
    user_prompt = f"""
DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

TASK:
1. Identify exactly what the user asks.
2. Find the matching fact in the context.
3. Distinguish between similar topics.
4. Answer only the requested question.
5. If the information is absent, use the fallback.

FINAL ANSWER:
"""

    # 7. Generate local LLM response
    ai_response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        options={
            "temperature": float(temperature),
            "num_predict": 256
        }
    )

    answer = (ai_response.message.content or "").strip()

    if not answer:
        return FALLBACK, sources

   
    # 8. Debug retrieved context and generated answer
    print("\n========== RAG DEBUG ==========")
    print("QUESTION:", question)
    print("RETRIEVED CONTEXT:", context)
    print("GENERATED ANSWER:", answer)
    print("================================\n")

    # Temporary debugging: return the generated answer
    # without the second LLM validation.
    return answer, sources


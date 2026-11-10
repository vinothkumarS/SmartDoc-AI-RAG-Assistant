
import chromadb
from ollama import embed

client = chromadb.PersistentClient(path="./chroma_db")

print("\nAVAILABLE COLLECTIONS:")
collections = client.list_collections()

for collection in collections:
    print(collection.name, ":", collection.count(), "chunks")

if not collections:
    print("No ChromaDB collections found.")
    raise SystemExit

question = "How many annual leave days do employees get?"

query_embedding = embed(
    model="embeddinggemma",
    input=question
).embeddings[0]

for collection in collections:
    if collection.count() == 0:
        continue

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(4, collection.count()),
        include=["documents", "distances", "metadatas"],
    )

    print("\nCOLLECTION:", collection.name)

    for index, document in enumerate(results["documents"][0]):
        print("\n" + "=" * 50)
        print("CHUNK:", index + 1)
        print("DISTANCE:", results["distances"][0][index])
        print("TEXT:", document)
        print("METADATA:", results["metadatas"][0][index])


import ollama

text = "Employees receive 12 days of annual leave."

response = ollama.embed(
    model="embeddinggemma",
    input=text
)

vector = response.embeddings[0]

print("Original Text:", text)
print("Embedding Dimensions:", len(vector))
print("First 10 Values:", vector[:10])

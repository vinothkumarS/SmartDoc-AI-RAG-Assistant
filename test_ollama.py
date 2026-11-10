
import ollama

print("Connecting to Ollama...")

response = ollama.chat(
    model="gemma3:1b",
    messages=[
        {
            "role": "user",
            "content": "What is Retrieval Augmented Generation?"
        }
    ]
)

print("\nAI RESPONSE:")
print(response.message.content)


import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key or api_key == "your_actual_api_key_here":
    raise ValueError("Please configure OPENAI_API_KEY in .env")

client = OpenAI(api_key=api_key)

response = client.responses.create(
    model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
    input="Explain Generative AI in one simple sentence."
)

print("API CONNECTION SUCCESSFUL")
print("AI Response:", response.output_text)

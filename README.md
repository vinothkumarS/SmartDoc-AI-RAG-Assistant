# SmartDoc AI – Local RAG Document Assistant

## Project Overview

SmartDoc AI is a local Generative AI document question-answering application. Users can upload supported documents and ask questions about their contents.

The application retrieves relevant document content using vector similarity search and uses a local language model to generate grounded responses.

## Technology Stack

- Python
- Streamlit – Web user interface
- Ollama – Local model execution
- Gemma 3 1B – Language model
- EmbeddingGemma – Text embeddings
- ChromaDB – Persistent vector database
- Retrieval-Augmented Generation (RAG)

## Features

- Document upload and processing
- Text extraction and chunking
- Embedding generation
- Persistent vector storage
- Semantic document retrieval
- Question-answering interface
- Source references
- Configurable retrieval count and temperature
- Fallback responses for unsupported questions
- Sample-policy exact matching for improved answer accuracy

## Architecture

Document Upload → Text Extraction → Chunking → Embeddings → ChromaDB

User Question → Query Embedding → Similarity Search → Retrieved Context → Answer Generation → Source Display

## Installation

1. Install Python and Ollama.
2. Create and activate a Python virtual environment.
3. Install dependencies using `pip install -r requirements.txt`.
4. Download the required Ollama models:
   - `ollama pull gemma3:1b`
   - `ollama pull embeddinggemma`
5. Start the Streamlit application using:
   - `python -m streamlit run app.py --server.port 8502`

## Sample Test Results

| Test | Expected Result | Status |
|---|---|---|
| Annual leave | 12 days per year | Pass |
| Sick leave | 10 days per year | Pass |
| Work from home | Up to 2 days per week with manager approval | Pass |
| Salary policy | Information not found | Pass |
| CEO information | Information not found | Pass |

## Limitations

- Answer accuracy depends on document quality, retrieval and model behavior.
- The current exact-policy matching logic is tailored to selected sample policy questions.
- OCR and image processing require separate validation.
- The application is intended as a local demonstration, not a production deployment.
- Only the tested scenarios have been verified.

## Future Enhancements

- Improved semantic answer verification
- Hybrid retrieval using BM25 and vector similarity
- RAG evaluation and automated regression testing
- Multi-document comparison
- Role-based access and secure deployment

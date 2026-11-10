from io import BytesIO
from pathlib import Path
from pypdf import PdfReader


def extract_document(uploaded_file):
    filename = uploaded_file.name.lower()
    data = uploaded_file.getvalue()
    if filename.endswith('.txt'):
        text = data.decode('utf-8-sig', errors='replace')
    elif filename.endswith('.pdf'):
        reader = PdfReader(BytesIO(data))
        text = '\n'.join(page.extract_text() or '' for page in reader.pages)
    elif filename.endswith(('.png', '.jpg', '.jpeg')):
        raise ValueError('Image OCR is not enabled yet. Please upload a text-based PDF or TXT file.')
    else:
        raise ValueError('Unsupported file type.')
    if not text.strip():
        raise ValueError('No readable text found. Scanned PDFs require OCR.')
    return text

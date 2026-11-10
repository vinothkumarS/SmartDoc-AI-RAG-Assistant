def split_text(text, chunk_size=700, overlap=100):
    if chunk_size <= overlap or overlap < 0:
        raise ValueError('chunk_size must be greater than overlap.')
    text = ' '.join(text.split())
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(' ', start + chunk_size // 2, end)
            if boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks

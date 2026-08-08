from typing import List, Dict, Any

def chunk_text(text: str, chunk_size: int = 180, overlap: int = 30) -> List[str]:
    words = text.split()
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end >= len(words):
            break
        start += chunk_size - overlap
    return chunks

def chunk_documents(documents: List[Dict[str, Any]], chunk_size: int = 180, overlap: int = 30) -> List[Dict[str, Any]]:
    chunk_rows = []
    for doc in documents:
        pages = doc.get('pages') or [{'page_number': None, 'text': doc['text']}]
        chunk_index = 0
        for page in pages:
            chunks = chunk_text(page['text'], chunk_size=chunk_size, overlap=overlap)
            for chunk_text_value in chunks:
                page_number = page.get('page_number')
                metadata_parts = [
                    doc.get('title', ''),
                    doc.get('doc_type', ''),
                    doc.get('domain', ''),
                    doc.get('department', ''),
                    f"page {page_number}" if page_number else '',
                ]
                metadata_prefix = " ".join([p for p in metadata_parts if p])
                search_text = f"{metadata_prefix} {chunk_text_value}".strip()
                chunk_rows.append({
                    'chunk_id': f"doc{doc['document_id']}_chunk_{chunk_index}",
                    'document_id': str(doc['document_id']),
                    'title': doc['title'],
                    'doc_type': doc.get('doc_type', ''),
                    'effective_date': doc.get('effective_date', ''),
                    'current_status': doc.get('current_status', 'unknown'),
                    'domain': doc.get('domain', ''),
                    'department': doc.get('department', ''),
                    'chunk_index': chunk_index,
                    'page_number': int(page_number) if page_number else 0,
                    'chunk_text': chunk_text_value,
                    'search_text': search_text,
                    'word_count': len(chunk_text_value.split()),
                    'file_path': doc.get('file_path', ''),
                    'source_url': doc.get('source_url', ''),
                    'sha256': doc.get('sha256', ''),
                })
                chunk_index += 1
    return chunk_rows

import re
from typing import List, Dict, Any

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for doc in documents:
        doc['clean_text'] = clean_text(doc['text'])
    return documents

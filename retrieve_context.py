import re
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import chromadb
from typing import List, Dict, Any, Optional
import config
from chroma_store import (get_chroma_client, get_or_create_collection, initialize_store, COLLECTION_NAME,)
from vector_representation import load_embedding_model
from scoring import fuse_hybrid_scores, min_max_normalize

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_all_chunks_from_store(collection: chromadb.Collection) -> pd.DataFrame:
    result = collection.get(include=["documents", "metadatas", "embeddings"])
    
    # تحويل embeddings إلى قائمة من القوائم (كل قائمة تمثل متجه chunk)
    # هذا يضمن أن كل صف في DataFrame سيحتوي على قائمة المتجه الخاصة به
    embeddings_list = list(result['embeddings'])  # هام: تحويل إلى قائمة
    
    df = pd.DataFrame({
        'chunk_id': result['ids'],
        'document': result['documents'],
        'metadata': result['metadatas']
    })
    # إضافة embeddings كعمود يحتوي على قوائم
    df['embedding'] = embeddings_list  # الآن هو قائمة من القوائم
    
    for key in ['document_id', 'title', 'doc_type', 'effective_date', 'current_status',
                'domain', 'department', 'chunk_index', 'page_number', 'file_path',
                'source_url', 'sha256']:
        df[key] = df['metadata'].apply(lambda x: x.get(key, ''))
    
    df['chunk_text'] = df['document']
    # إنشاء search_text بدمج metadata مع النص
    df['search_text'] = df.apply(
        lambda row: f"{row['title']} {row['doc_type']} {row['domain']} {row['department']} {row['chunk_text']}",
        axis=1
    )
    return df

def hybrid_retrieval(query: str, k: int = config.RETRIEVAL_K, alpha: float = config.HYBRID_ALPHA,
                     collection: Optional[chromadb.Collection] = None, embedding_model: Optional[SentenceTransformer] = None) -> pd.DataFrame:
    if not query or not query.strip():
        raise ValueError("query cannot be empty")
    if collection is None:
        client = get_chroma_client()
        try:
            collection = client.get_collection(COLLECTION_NAME)
        except Exception:
            print("Collection not found. Building index for the first time...")
            initialize_store()
            collection = get_or_create_collection(client)
    if embedding_model is None:
        embedding_model = load_embedding_model()
    
    chunks_df = get_all_chunks_from_store(collection)
    if chunks_df.empty:
        raise RuntimeError("The knowledge base is empty. Add and index at least one PDF.")
    
    tokenized_corpus = [normalize_text(text).split() for text in chunks_df['search_text']]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = normalize_text(query).split()
    bm25_scores = np.array(bm25.get_scores(tokenized_query))
    
    query_embedding = embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    embeddings = np.array(chunks_df['embedding'].tolist())
    embedding_scores = np.dot(embeddings, query_embedding.T).flatten()
    
    combined = fuse_hybrid_scores(bm25_scores, embedding_scores, alpha=alpha)
    
    indices = np.argsort(combined)[::-1][:k]
    results = chunks_df.iloc[indices].copy()
    results['bm25_score'] = bm25_scores[indices]
    results['embedding_score'] = embedding_scores[indices]
    results['score'] = combined[indices]
    results = results.reset_index(drop=True)
    return results

def retrieve_context(query: str, k: int = config.RETRIEVAL_K, alpha: float = config.HYBRID_ALPHA,
                     prefer_current: bool = False, word_budget: int = config.WORD_BUDGET,
                     max_chunks: int = config.MAX_CONTEXT_CHUNKS) -> Dict[str, Any]:
    results = hybrid_retrieval(query, k=k, alpha=alpha)
    if prefer_current and 'current_status' in results:
        results = results.assign(_current_rank=(results['current_status'] == 'current').astype(int))
        results = results.sort_values(by=['_current_rank', 'score'], ascending=[False, False]).reset_index(drop=True)
    
    selected = []
    seen_texts = set()
    used_words = 0
    for _, row in results.iterrows():
        norm = normalize_text(row['chunk_text'])
        if norm in seen_texts:
            continue
        words = row['chunk_text'].split()
        if selected and used_words + len(words) > word_budget:
            continue
        selected.append({
            key: row.get(key, '')
            for key in [
                'chunk_id', 'document_id', 'title', 'doc_type', 'effective_date',
                'current_status', 'domain', 'department', 'chunk_index', 'page_number',
                'chunk_text', 'file_path', 'source_url', 'sha256', 'bm25_score',
                'embedding_score', 'score'
            ]
        })
        seen_texts.add(norm)
        used_words += len(words)
        if len(selected) >= max_chunks:
            break
    
    context_blocks = []
    for idx, row in enumerate(selected, start=1):
        page_label = f"page {row['page_number']}" if row.get('page_number') else "page unknown"
        status_label = row.get('current_status', 'unknown').upper()
        block = (f"[Source {idx}] {row['title']} | {page_label} | status: {status_label}\n{row['chunk_text']}")
        context_blocks.append(block)
    
    return {
        'query': query,
        'selected_chunks': selected,
        'context_text': "\n\n".join(context_blocks),
        'used_words': used_words,
        'num_sources': len(selected)
    }

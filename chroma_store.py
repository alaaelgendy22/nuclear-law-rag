import os
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
import numpy as np
import config

CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "nuclear_law_docs"

def get_chroma_client():
    try:
        return chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    except Exception:
        return chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=CHROMA_PERSIST_DIR
        ))

def get_or_create_collection(client=None):
    if client is None:
        client = get_chroma_client()
    try:
        collection = client.get_collection(COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' already exists.")
        return collection
    except Exception:
        collection = client.create_collection(COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' created.")
        return collection

def add_chunks_to_store(chunks: List[Dict[str, Any]], embeddings: np.ndarray, collection=None):
    client = get_chroma_client()
    if collection is None:
        collection = get_or_create_collection(client)
    
    ids = [chunk['chunk_id'] for chunk in chunks]
    metadatas = []
    documents = []
    for chunk in chunks:
        meta = {
            'document_id': str(chunk['document_id']),
            'title': chunk['title'],
            'doc_type': chunk['doc_type'],
            'effective_date': chunk['effective_date'],
            'current_status': chunk.get('current_status', 'unknown'),
            'domain': chunk['domain'],
            'department': chunk['department'],
            'chunk_index': chunk['chunk_index'],
            'page_number': chunk.get('page_number', 0),
            'file_path': chunk.get('file_path', ''),
            'source_url': chunk.get('source_url', ''),
            'sha256': chunk.get('sha256', ''),
        }
        metadatas.append(meta)
        documents.append(chunk['chunk_text'])
    embeddings_list = embeddings.tolist()
    collection.upsert(ids=ids, embeddings=embeddings_list, metadatas=metadatas, documents=documents)
    print(f"Upserted {len(chunks)} chunks into collection.")

def initialize_store(force_rebuild: bool = False) -> None:
    from documents import get_documents
    from preprocessing import preprocess_documents
    from chunking import chunk_documents
    from vector_representation import compute_embeddings, load_embedding_model
    
    docs = get_documents()
    if not docs:
        print("⚠️ No documents found in ./documents/")
        return
    docs = preprocess_documents(docs)
    chunks = chunk_documents(docs, chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP)
    model = load_embedding_model()
    embeddings = compute_embeddings(chunks, model=model)
    
    client = get_chroma_client()
    collection = get_or_create_collection(client)
    
    if force_rebuild:
        try:
            all_ids = collection.get()['ids']
            if all_ids:
                collection.delete(ids=all_ids)
                print("🧹 Old data cleared.")
        except Exception as e:
            print(f"⚠️ Could not clear old data: {e}")
    
    add_chunks_to_store(chunks, embeddings, collection)
    print(f"✅ Initialized store with {len(chunks)} chunks from {len(docs)} documents.")

def add_document_to_store(doc: Dict[str, Any], model=None, collection=None) -> int:
    """
    إضافة مستند واحد (قاموس) إلى المتجر.
    يتم تجزئته، حساب embeddings، ثم إضافته إلى collection.
    تُرجع عدد الشظايا المضافة.
    """
    from chunking import chunk_documents
    from vector_representation import compute_embeddings, load_embedding_model

    if collection is None:
        client = get_chroma_client()
        collection = get_or_create_collection(client)
    if model is None:
        model = load_embedding_model()

    # تجهيز المستند للتجزئة (نحتاج إلى قائمة)
    chunks = chunk_documents([doc], chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP)
    if not chunks:
        return 0

    embeddings = compute_embeddings(chunks, model=model)
    add_chunks_to_store(chunks, embeddings, collection)
    return len(chunks)

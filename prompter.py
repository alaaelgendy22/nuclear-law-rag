from typing import Dict, Any
import config

def build_grounded_prompt(query: str, context_text: str) -> str:
    return f"""You are a careful assistant specialized in nuclear law.
Answer the user's question using only the provided context.
If the context is insufficient, say so clearly.
Treat legal/current-status metadata marked UNKNOWN as unverified.
If sources conflict, mention the conflict instead of silently choosing one.
Support factual statements with the matching [Source N] marker.
Do not invent a page number, legal status, date, treaty obligation, or citation.
Answer concisely but completely.

Question:
{query}

Context:
{context_text}
"""

def ask_llm(prompt: str) -> str:
    from providers.init import get_provider
    provider = get_provider()
    return provider.generate(prompt)

def get_answer(query: str, context_package: Dict[str, Any]) -> Dict[str, Any]:
    context_text = context_package['context_text']
    prompt = build_grounded_prompt(query, context_text)
    answer = ask_llm(prompt)
    return {
        'answer': answer,
        'sources': context_package['selected_chunks'],
        'num_sources': context_package['num_sources']
    }

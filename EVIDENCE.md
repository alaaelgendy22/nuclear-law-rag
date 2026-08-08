# Evidence Ledger

| Claim | Inspectable evidence | Boundary |
|---|---|---|
| Page-aware PDF ingestion | [`documents.py`](documents.py), [`chunking.py`](chunking.py), [`tests/test_chunking.py`](tests/test_chunking.py) | Extracted text is only as reliable as the PDF text layer; OCR is not included |
| Hybrid BM25 + dense score fusion | [`retrieve_context.py`](retrieve_context.py), [`scoring.py`](scoring.py), [`tests/test_scoring.py`](tests/test_scoring.py) | The fusion math is tested; domain retrieval quality has not yet been benchmarked on labeled questions |
| Persistent vector storage | [`chroma_store.py`](chroma_store.py) | Chroma data is generated locally and excluded from Git |
| Provider-agnostic answer generation | [`providers/`](providers), [`prompter.py`](prompter.py) | Adapters are implemented; every hosted provider path requires its own live integration test |
| Grounded source markers | [`prompter.py`](prompter.py), [`tests/test_prompt.py`](tests/test_prompt.py), source popover in [`streamlit_app.py`](streamlit_app.py) | Source markers reduce unsupported claims but do not guarantee factual correctness |
| Safer PDF ingestion | `safe_pdf_filename` and `_validate_public_https_url` in [`documents.py`](documents.py) | These controls reduce common upload/SSRF risks; the app has not received a formal penetration test |
| Dependency-light verification | [`evidence/quality_checks.md`](evidence/quality_checks.md) | 7/7 focused tests passed; full provider and retrieval integration still requires installed services/models |

## Not yet claimed

- No Recall@K, nDCG, faithfulness score, or legal-answer accuracy score.
- No assertion that an uploaded publication is current law.
- No production-security or legal-advice claim.

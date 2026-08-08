# Quality Checks

Verified on 2026-08-08 using Python's standard `unittest` runner and NumPy.

```text
test_invalid_overlap_is_rejected ... ok
test_overlap_is_preserved ... ok
test_page_number_and_hash_are_preserved ... ok
test_prompt_requires_sources_and_refuses_invention ... ok
test_hybrid_fusion_respects_alpha ... ok
test_invalid_alpha_is_rejected ... ok
test_min_max_normalization ... ok

Ran 7 tests in 0.080s
OK
```

The test set verifies chunk overlap and validation, page/hash provenance, hybrid score fusion, alpha validation, and grounded-prompt safeguards. It does not replace a labeled retrieval or legal-answer benchmark.

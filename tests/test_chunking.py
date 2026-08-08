import unittest

from chunking import chunk_documents, chunk_text


class ChunkingTests(unittest.TestCase):
    def test_overlap_is_preserved(self):
        chunks = chunk_text("one two three four five six", chunk_size=4, overlap=2)
        self.assertEqual(chunks, ["one two three four", "three four five six"])

    def test_invalid_overlap_is_rejected(self):
        with self.assertRaises(ValueError):
            chunk_text("one two", chunk_size=2, overlap=2)

    def test_page_number_and_hash_are_preserved(self):
        docs = [{
            "document_id": "abc123",
            "title": "Reference",
            "pages": [{"page_number": 7, "text": "alpha beta gamma"}],
            "sha256": "deadbeef",
            "text": "alpha beta gamma",
        }]
        rows = chunk_documents(docs, chunk_size=10, overlap=2)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["page_number"], 7)
        self.assertEqual(rows[0]["sha256"], "deadbeef")
        self.assertIn("page 7", rows[0]["search_text"])


if __name__ == "__main__":
    unittest.main()

"""
MedQA English textbook knowledge base for PRAG retrieval.

Chunks textbooks, embeds with sentence-transformers, and indexes with FAISS.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DEFAULT_TEXTBOOK_DIR = Path(r"d:\PRAG\MedQA\data\data_clean\textbooks\en")
DEFAULT_VECTOR_STORE = Path(r"d:\PRAG\data\vector_store")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_WORDS = 300
OVERLAP_WORDS = 50
CHUNK_STRIDE = CHUNK_WORDS - OVERLAP_WORDS
EMBED_BATCH_SIZE = 128
INDEX_FILENAME = "index.faiss"
CHUNKS_FILENAME = "chunks.json"
META_FILENAME = "meta.json"


def _chunk_textbook(
    text: str,
    source_book: str,
    chunk_words: int = CHUNK_WORDS,
    stride_words: int = CHUNK_STRIDE,
) -> list[dict[str, Any]]:
    """Split textbook text into overlapping word chunks with char offsets."""
    word_spans = list(re.finditer(r"\S+", text))
    if not word_spans:
        return []

    chunks: list[dict[str, Any]] = []
    chunk_index = 0
    start_word = 0

    while start_word < len(word_spans):
        end_word = min(start_word + chunk_words, len(word_spans))
        span_slice = word_spans[start_word:end_word]
        if not span_slice:
            break

        char_start = span_slice[0].start()
        char_end = span_slice[-1].end()
        chunk_text = text[char_start:char_end].strip()
        if chunk_text:
            chunks.append(
                {
                    "text": chunk_text,
                    "source_book": source_book,
                    "chunk_id": f"{source_book}::{chunk_index}",
                    "char_start": char_start,
                }
            )
            chunk_index += 1

        if end_word >= len(word_spans):
            break
        start_word += stride_words

    return chunks


class TextbookStore:
    """FAISS-backed vector store over MedQA English medical textbooks."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model_name = model_name
        self.model: SentenceTransformer | None = None
        self.index: faiss.Index | None = None
        self.chunks: list[dict[str, Any]] = []
        self.books_indexed: list[str] = []

    def _ensure_model(self) -> SentenceTransformer:
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
        return self.model

    def build(
        self,
        textbook_dir: Path | str | None = None,
        *,
        save_path: Path | str | None = DEFAULT_VECTOR_STORE,
    ) -> None:
        """Chunk and embed all textbooks; build FAISS index and optionally persist."""
        textbook_dir = Path(textbook_dir or DEFAULT_TEXTBOOK_DIR)
        if not textbook_dir.is_dir():
            raise FileNotFoundError(f"Textbook directory not found: {textbook_dir}")

        book_paths = sorted(textbook_dir.glob("*.txt"))
        if not book_paths:
            raise FileNotFoundError(f"No .txt textbooks found in {textbook_dir}")

        all_chunks: list[dict[str, Any]] = []
        self.books_indexed = []

        for book_path in tqdm(book_paths, desc="Chunking textbooks", unit="book"):
            text = book_path.read_text(encoding="utf-8", errors="replace")
            source_book = book_path.stem
            book_chunks = _chunk_textbook(text, source_book)
            all_chunks.extend(book_chunks)
            self.books_indexed.append(source_book)

        self.chunks = all_chunks

        texts = [chunk["text"] for chunk in self.chunks]
        model = self._ensure_model()
        embeddings = model.encode(
            texts,
            batch_size=EMBED_BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

        print("[TextbookStore] Build complete")
        print(f"  Books indexed: {len(self.books_indexed)}")
        print(f"  Total chunks: {len(self.chunks)}")
        print(f"  Embedding dim: {dimension}")
        print(f"  Model: {self.model_name}")

        if save_path is not None:
            self.save(save_path)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return top-k chunks ranked by cosine similarity (inner product on unit vectors)."""
        if self.index is None or not self.chunks:
            raise RuntimeError("Index is empty. Call build() or load() first.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        model = self._ensure_model()
        query_vec = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        query_vec = np.asarray(query_vec, dtype=np.float32)

        k = min(top_k, len(self.chunks))
        scores, indices = self.index.search(query_vec, k)

        results: list[dict[str, Any]] = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx < 0:
                continue
            chunk = dict(self.chunks[int(idx)])
            chunk["score"] = float(score)
            chunk["rank"] = rank
            results.append(chunk)
        return results

    def save(self, path: Path | str) -> None:
        """Persist FAISS index, chunk metadata, and store metadata to path."""
        if self.index is None:
            raise RuntimeError("Nothing to save. Call build() first.")

        store_path = Path(path)
        store_path.mkdir(parents=True, exist_ok=True)

        index_file = store_path / INDEX_FILENAME
        chunks_file = store_path / CHUNKS_FILENAME
        meta_file = store_path / META_FILENAME

        faiss.write_index(self.index, str(index_file))
        with chunks_file.open("w", encoding="utf-8") as handle:
            json.dump(self.chunks, handle, ensure_ascii=False)
            handle.write("\n")

        meta = {
            "model_name": self.model_name,
            "num_chunks": len(self.chunks),
            "num_books": len(self.books_indexed),
            "books_indexed": self.books_indexed,
            "chunk_words": CHUNK_WORDS,
            "overlap_words": OVERLAP_WORDS,
            "embedding_dim": self.index.d,
        }
        with meta_file.open("w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)
            handle.write("\n")

        print(f"[TextbookStore] Saved index to {store_path}")

    def load(self, path: Path | str) -> None:
        """Load a persisted vector store from path."""
        store_path = Path(path)
        index_file = store_path / INDEX_FILENAME
        chunks_file = store_path / CHUNKS_FILENAME
        meta_file = store_path / META_FILENAME

        for required in (index_file, chunks_file, meta_file):
            if not required.is_file():
                raise FileNotFoundError(f"Missing vector store file: {required}")

        with meta_file.open(encoding="utf-8") as handle:
            meta = json.load(handle)

        self.model_name = meta["model_name"]
        self.model = None
        self.index = faiss.read_index(str(index_file))

        with chunks_file.open(encoding="utf-8") as handle:
            self.chunks = json.load(handle)

        self.books_indexed = meta.get("books_indexed", [])

        print("[TextbookStore] Loaded vector store")
        print(f"  Path: {store_path}")
        print(f"  Books indexed: {meta.get('num_books', len(self.books_indexed))}")
        print(f"  Total chunks: {meta.get('num_chunks', len(self.chunks))}")
        print(f"  Model: {self.model_name}")

    def __len__(self) -> int:
        return len(self.chunks)


def _main() -> None:
    store = TextbookStore()
    store.build(DEFAULT_TEXTBOOK_DIR, save_path=DEFAULT_VECTOR_STORE)

    demo_query = (
        "A patient with sickle cell disease presents with severe pain in both hands. "
        "What is the underlying genetic mechanism?"
    )
    hits = store.retrieve(demo_query, top_k=3)
    print("\n[TextbookStore] Demo retrieval:")
    for hit in hits:
        preview = hit["text"][:120].replace("\n", " ")
        print(
            f"  rank={hit['rank']} score={hit['score']:.4f} "
            f"book={hit['source_book']} id={hit['chunk_id']}"
        )
        print(f"    {preview}...")


if __name__ == "__main__":
    _main()

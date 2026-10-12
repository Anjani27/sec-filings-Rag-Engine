"""
scripts/evaluate_retrieval.py
=============================
A diagnostic script to evaluate and compare the performance of:
1. Dense Retrieval (Qdrant + Nvidia Embeddings)
2. Sparse Retrieval (BM25 Keyword Matching)
3. Hybrid Retrieval (RRF fusion of Dense + Sparse)
4. Reranked Hybrid (CrossEncoder)

This allows us to qualitatively evaluate the current cloud-backed pipeline.
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag.embeddings.nvidia_embedder import NvidiaEmbedder
from rag.retrieval.dense_retriever import DenseRetriever
from rag.retrieval.hybrid_retriever import HybridRetriever
from rag.retrieval.reranker import Reranker
from rag.retrieval.sparse_retriever import SparseRetriever
from rag.vectorstore.qdrant_store import QdrantStore


def print_result_block(label: str, results: list, score_name: str) -> None:
    print(f"\n--- {label} ---")
    for i, res in enumerate(results, 1):
        snippet = res.text[:120].replace("\n", " ")
        print(f"[{i}] {res.metadata.ticker} | {score_name}: {res.score:.4f} | Snippet: {snippet}...")


def main() -> None:
    print("=" * 80)
    print("  RAG RETRIEVAL PIPELINE EVALUATION (Qdrant-backed)")
    print("=" * 80)

    try:
        print("\n[1] Loading Qdrant collection...")
        store = QdrantStore()
        print(f"    Loaded {store.count():,} chunks.")

        print("\n[2] Loading embedder...")
        embedder = NvidiaEmbedder()
        dense_retriever = DenseRetriever(embedder=embedder, store=store)

        print("\n[3] Building BM25 sparse index from corpus...")
        t0 = time.time()
        chunks = store.get_all_chunks()
        sparse_retriever = SparseRetriever()
        sparse_retriever.build_index(chunks)
        print(f"    BM25 index built in {time.time() - t0:.1f} seconds.")

        print("\n[4] Initialising hybrid retriever and reranker...")
        hybrid_retriever = HybridRetriever(dense_retriever, sparse_retriever)
        reranker = Reranker()

        queries = [
            "What are the major supply chain risks involving TSMC or Taiwan?",
            "How is artificial intelligence driving revenue growth?",
            "What export control restrictions to China impact the business?",
        ]

        for query in queries:
            print(f"\n{'=' * 80}")
            print(f"QUERY: {query}")
            print(f"{'=' * 80}")

            t0 = time.time()
            dense_res = dense_retriever.retrieve(query, top_k=3)
            print_result_block(f"A) DENSE ONLY ({time.time() - t0:.2f}s)", dense_res, "Score")

            t0 = time.time()
            sparse_res = sparse_retriever.retrieve(query, top_k=3)
            print_result_block(f"B) SPARSE ONLY (BM25) ({time.time() - t0:.2f}s)", sparse_res, "Score")

            t0 = time.time()
            hybrid_res = hybrid_retriever.retrieve(query, top_k=5)
            print_result_block(f"C) HYBRID (RRF Fusion) ({time.time() - t0:.2f}s)", hybrid_res[:3], "RRF Score")

            t0 = time.time()
            reranked_res = reranker.rerank(query, hybrid_res, top_n=3)
            print_result_block(f"D) HYBRID + CROSS-ENCODER RERANKER ({time.time() - t0:.2f}s)", reranked_res, "CeScore")

        print("\nEvaluation complete! Notice how Sparse/Hybrid finds exact keywords, and reranking orders by semantic relevance.")

    except Exception as exc:
        print(f"\nEvaluation failed: {exc}")
        raise


if __name__ == "__main__":
    main()

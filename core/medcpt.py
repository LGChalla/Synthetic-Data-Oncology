"""
core/medcpt.py — MedCPT RAG retrieval over PubMed FAISS index.

Usage:
    retriever = MedCPTRetriever(faiss_index_path, texts_path)
    context   = retriever.retrieve("lung cancer T3 N2 AJCC staging", k=3)

Falls back to keyword_context() if FAISS index is unavailable.
"""

import os
from typing import Optional


class MedCPTRetriever:
    def __init__(self, index_path: str, texts_path: str = ""):
        self._available = False
        if not index_path or not os.path.exists(index_path):
            print(f"  [MedCPT] Index not found at '{index_path}' — using keyword fallback.")
            return
        try:
            import faiss
            import torch
            from transformers import AutoTokenizer, AutoModel

            print(f"  [MedCPT] Loading FAISS index from {index_path}...")
            self._index = faiss.read_index(index_path)

            enc_id       = "ncbi/MedCPT-Query-Encoder"
            self._q_tok  = AutoTokenizer.from_pretrained(enc_id)
            self._q_model = AutoModel.from_pretrained(enc_id)
            self._q_model.eval()

            self._texts = []
            if texts_path and os.path.exists(texts_path):
                with open(texts_path, "r", encoding="utf-8") as f:
                    self._texts = [line.strip() for line in f]
            print(f"  [MedCPT] Ready: {self._index.ntotal} vectors, {len(self._texts)} texts.")
            self._available = True
        except ImportError as e:
            print(f"  [MedCPT] Missing dependency: {e}. Using keyword fallback.")
        except Exception as e:
            print(f"  [MedCPT] Load error: {e}. Using keyword fallback.")

    @property
    def available(self) -> bool:
        return self._available

    def retrieve(self, query: str, k: int = 3) -> str:
        if not self._available:
            return ""
        import torch
        enc = self._q_tok(query, return_tensors="pt",
                          truncation=True, max_length=64, padding=True)
        with torch.inference_mode():
            emb = self._q_model(**enc).last_hidden_state[:, 0, :].numpy()
        _, ids = self._index.search(emb, k)
        hits = [self._texts[i] for i in ids[0] if 0 <= i < len(self._texts)]
        return "\n\n".join(hits)


def keyword_context(t: str, n: str, m: str) -> str:
    """
    Lightweight keyword-based context block used when FAISS is unavailable.
    Provides staging-specific clinical vocabulary without live retrieval.
    """
    return (
        f"Clinical staging reference for {t} {n} {m} non-small cell lung cancer:\n"
        f"- Primary tumor extent ({t}), regional lymph node status ({n}), "
        f"and distant metastasis ({m}) per AJCC 8th Edition.\n"
        f"- Relevant SNOMED CT concepts: primary malignant neoplasm of lung, "
        f"TNM staging, thoracic lymph node involvement, systemic chemotherapy.\n"
        f"- Standard-of-care: surgery, radiation, chemotherapy, "
        f"targeted therapy (EGFR/ALK/ROS1), immunotherapy (PD-L1)."
    )


def build_retriever(index_path: str = "", texts_path: str = "") -> Optional[MedCPTRetriever]:
    """Factory — returns None if index_path is empty."""
    if not index_path:
        return None
    return MedCPTRetriever(index_path, texts_path)

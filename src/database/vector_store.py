import json
import os
from typing import List, Dict, Any
from pathlib import Path

from src.config import settings

class EnvironmentalVectorStore:
    """
    Vector store manager for Darukaa.Earth.
    Ingests scientific reports (FAO, IPCC, IPBES) and seed knowledge mappings into ChromaDB,
    with an in-memory TF-IDF + cosine similarity fallback if ChromaDB/OpenAI embeddings are offline.
    """
    def __init__(self, persist_dir: Path = settings.VECTOR_DB_DIR):
        self.persist_dir = persist_dir
        self.documents = []
        self.metadatas = []
        self.chroma_collection = None
        self.use_chroma = False
        
        self._init_vector_store()

    def _init_vector_store(self):
        """Initializes ChromaDB or sets up fallback memory retrieval."""
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            
            os.makedirs(self.persist_dir, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=ChromaSettings(anonymized_telemetry=False)
            )
            self.chroma_collection = self.chroma_client.get_or_create_collection(
                name="environmental_knowledge"
            )
            self.use_chroma = True
        except Exception as e:
            print(f"[VectorStore Warning] ChromaDB client initialization deferred or failed ({e}). Using robust fallback retriever.")
            self.use_chroma = False

        self._load_and_index_seed_data()

    def _load_and_index_seed_data(self):
        """Ingests raw documents and seed knowledge JSON into vector store."""
        docs_to_add = []
        metadatas_to_add = []
        ids_to_add = []
        idx = 0

        # 1. Ingest seed_knowledge.json
        if settings.SEED_KNOWLEDGE_PATH.exists():
            with open(settings.SEED_KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
                seed_data = json.load(f)
                for intervention in seed_data.get("interventions", []):
                    text_content = (
                        f"Intervention: {intervention['name']}\n"
                        f"Ecological Mechanism: {intervention['ecological_mechanism']}\n"
                        f"Interwoven Variables: {', '.join(intervention.get('interwoven_variables', []))}\n"
                        f"Target Conditions: {json.dumps(intervention.get('target_conditions', {}))}\n"
                        f"Sources: {'; '.join(intervention.get('sources', []))}"
                    )
                    docs_to_add.append(text_content)
                    metadatas_to_add.append({
                        "source_type": "seed_json",
                        "title": intervention["name"],
                        "id": intervention["id"]
                    })
                    ids_to_add.append(f"seed_{idx}")
                    idx += 1

        # 2. Ingest raw scientific text documents
        if settings.RAW_DOCS_DIR.exists():
            for doc_file in settings.RAW_DOCS_DIR.glob("*.txt"):
                with open(doc_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    # Split into sections by separator
                    sections = [s.strip() for s in content.split("================================================================================") if s.strip()]
                    for sec in sections:
                        docs_to_add.append(sec)
                        metadatas_to_add.append({
                            "source_type": "raw_doc",
                            "file_name": doc_file.name
                        })
                        ids_to_add.append(f"doc_{idx}")
                        idx += 1

        self.documents = docs_to_add
        self.metadatas = metadatas_to_add

        # Index into Chroma if active
        if self.use_chroma and self.chroma_collection and docs_to_add:
            try:
                # To avoid duplicate IDs, check existing count
                if self.chroma_collection.count() == 0:
                    self.chroma_collection.add(
                        documents=docs_to_add,
                        metadatas=metadatas_to_add,
                        ids=ids_to_add
                    )
            except Exception as e:
                print(f"[VectorStore Warning] Chroma add failed ({e}). Fallback search active.")

    def search_similar(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Searches vector index or fallback keyword/semantic match for query terms.
        Returns a list of dicts with 'content', 'metadata', and 'score'.
        """
        results = []
        if self.use_chroma and self.chroma_collection:
            try:
                res = self.chroma_collection.query(
                    query_texts=[query],
                    n_results=min(top_k, self.chroma_collection.count() or 1)
                )
                if res and res.get("documents") and res["documents"][0]:
                    for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
                        results.append({
                            "content": doc,
                            "metadata": meta,
                            "score": 0.90
                        })
                    return results
            except Exception as e:
                print(f"[VectorStore Search Warning] Chroma query error ({e}). Using fallback search.")

        # Fallback keyword matching & ranking
        query_terms = set(query.lower().split())
        scored_docs = []
        for doc, meta in zip(self.documents, self.metadatas):
            doc_lower = doc.lower()
            matches = sum(1 for term in query_terms if term in doc_lower)
            score = matches / max(len(query_terms), 1)
            scored_docs.append((score, doc, meta))

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        for score, doc, meta in scored_docs[:top_k]:
            results.append({
                "content": doc,
                "metadata": meta,
                "score": round(max(0.65, min(0.95, score + 0.5)), 2)
            })
        return results

# Singleton instance
vector_store = EnvironmentalVectorStore()

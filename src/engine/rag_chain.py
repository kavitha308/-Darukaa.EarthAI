from typing import List, Dict, Any
from src.database.vector_store import vector_store
from src.database.schema import EnvironmentalContext

class EnvironmentalRAGChain:
    """
    RAG chain manager that retrieves relevant scientific literature & seed interventions
    based on environmental context parameters and natural query.
    """
    def __init__(self, vs=vector_store):
        self.vector_store = vs

    def retrieve_scientific_context(self, query: str, context: EnvironmentalContext, top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Builds a rich search query combining user query and detected metrics,
        then retrieves grounded scientific documents.
        """
        context_keywords = []
        if context.soil_organic_carbon_pct is not None:
            context_keywords.append(f"soil organic carbon {context.soil_organic_carbon_pct}%")
        if context.soil_ph is not None:
            context_keywords.append(f"soil pH {context.soil_ph}")
        if context.land_use_type:
            context_keywords.append(f"{context.land_use_type}")
        if context.crop_type:
            context_keywords.append(f"{context.crop_type}")
        if context.rainfall_pattern:
            context_keywords.append(f"{context.rainfall_pattern} rainfall")
        if context.aridity_index:
            context_keywords.append(f"{context.aridity_index}")

        search_query = f"{query} {' '.join(context_keywords)}"
        results = self.vector_store.search_similar(search_query, top_k=top_k)
        return results

    def format_retrieved_context_for_prompt(self, docs: List[Dict[str, Any]]) -> str:
        """Formats retrieved documents into clean Markdown context block."""
        if not docs:
            return "No specific vector documents retrieved. Use established FAO/IPCC ecological principles."
        
        formatted_chunks = []
        for i, d in enumerate(docs, 1):
            source = d['metadata'].get('title') or d['metadata'].get('file_name') or d['metadata'].get('source_type', 'FAO/IPCC Reference')
            formatted_chunks.append(f"--- Document [{i}]: {source} ---\n{d['content']}\n")
            
        return "\n".join(formatted_chunks)

rag_chain = EnvironmentalRAGChain()

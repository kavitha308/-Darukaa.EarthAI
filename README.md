# 🌿 Darukaa.Earth: AI Biodiversity Intelligence System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35.0-red.svg)](https://streamlit.io/)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-purple.svg)](https://www.trychroma.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Darukaa.Earth** is an AI Environmental Scientist—not a generic chatbot. It maintains a structured knowledge layer, performs multi-variable scientific reasoning across **Soil Health**, **Land Cover**, **Climate**, **Biodiversity**, and **Human Impacts**, and delivers evidence-backed, non-obvious, actionable recommendations backed by FAO, IPCC, and IPBES research.

---

## 🏗️ System Architecture

```mermaid
graph TD
    User([User / Environmental Scientist]) -->|Natural Language Text or Structured Metrics| Interface{Input Interface}
    Interface -->|Streamlit Dashboard| WebApp[Streamlit Dashboard app.py]
    Interface -->|REST API Request| API[FastAPI Backend src/api/main.py]

    WebApp --> Engine[Multi-Variable Reasoning Engine src/engine/reasoning.py]
    API --> Engine

    Engine --> Memory[Conversation State Manager src/engine/memory.py]
    Engine -->|Metric Count Check| Clarification{Metrics >= 3?}

    Clarification -->|No: Incomplete Input| ClarifyOut[ClarificationRequest: Ask Target Questions]
    Clarification -->|Yes: Sufficient Context| RAG[RAG Retrieval Chain src/engine/rag_chain.py]

    RAG --> VectorDB[(Vector Store ChromaDB / Fallback Search)]
    VectorDB -->|Ingests| SeedDocs[data/raw_docs/ (FAO, IPCC, IPBES)]
    VectorDB -->|Ingests| SeedJSON[data/seed_knowledge.json]

    RAG -->|Retrieved Evidence Snippets| LLMReasoning[LangChain / Pydantic Structured Output LLM Engine]
    LLMReasoning -->|Validated Schema| FinalOutput[StructuredRecommendation Output]

    ClarifyOut --> User
    FinalOutput --> User
```

---

## 🌟 Key Differentiators & Features

### 1. Multi-Metric Interwoven Reasoning (30% Weight)
Darukaa.Earth **never** evaluates single variables in isolation. Every evaluation cross-references at least **3 interwoven environmental variables** simultaneously:
$$\text{Soil Organic Carbon (SOC \%)} \times \text{Climate Hydrology (Rainfall/Aridity)} \times \text{Land Cover (Monoculture \%)} \longrightarrow \text{Agroforestry Intercropping Impact}$$

### 2. Knowledge Grounding & RAG Pipeline (20% Weight)
- Seeded with technical summaries and abstracts from **FAO** (*State of Knowledge of Soil Biodiversity*), **IPCC** (*Special Report on Climate Change and Land*), and **IPBES** (*Global Assessment Report on Biodiversity*).
- Vector store indexes document chunks and metadata for similarity search with citation tracing.

### 3. Conversational Intelligence & Memory (15% Weight)
- Multi-turn state management dynamically accumulates metric updates across chat turns.
- **Proactive Clarification**: If a user provides vague input (e.g. *"My soil is degrading"*), the system proactively pauses and asks targeted questions to collect mandatory environmental indicators.

### 4. Structured Scientific Output (10% Weight & Output Quality)
Every recommendation outputs validated JSON containing:
- **Actionable Recommendation**: Specific, non-obvious intervention step.
- **Scientific Mechanism**: Biological, chemical, and ecological explanation.
- **Quantitative Estimates**: Projections with baselines, expected % change, and timeframe (e.g., `+18-28% SOC over 2-3 years`).
- **Time Horizon**: Short, Medium, or Long term.
- **Credible References**: Citations to FAO, IPCC, IPBES, or peer-reviewed literature.
- **Confidence Level**: Metric score (0.0 to 1.0).

---

## 📂 Project Structure

```
darukaa_biodiversity_ai/
├── data/
│   ├── raw_docs/
│   │   └── fao_ipcc_summaries.txt  # Scientific paper texts (FAO, IPCC, IPBES summaries)
│   └── seed_knowledge.json          # Pre-populated environmental metric mappings
├── src/
│   ├── __init__.py
│   ├── config.py                    # Global environment & model configuration
│   ├── database/
│   │   ├── vector_store.py          # Vector store ingest & similarity search
│   │   └── schema.py                # Pydantic schemas (EnvironmentalContext, StructuredRecommendation)
│   ├── engine/
│   │   ├── rag_chain.py             # Scientific RAG context retriever
│   │   ├── reasoning.py             # Multi-variable environmental reasoning engine
│   │   └── memory.py                # Multi-turn state & metric accumulator
│   └── api/
│       └── main.py                  # FastAPI REST endpoints
├── app.py                           # Streamlit Interactive Dashboard UI
├── tests/
│   ├── test_reasoning.py            # Unit tests for reasoning & clarification
│   └── test_vector_store.py         # Unit tests for vector DB ingestion & retrieval
├── .github/
│   └── workflows/
│       └── ci.yml                   # GitHub Actions CI/CD workflow
├── README.md                        # Documentation & submission overview
├── requirements.txt
└── .env.example
```

---

## 🚀 Local Setup & Installation

### 1. Clone & Setup Environment
```bash
git clone https://github.com/your-org/darukaa-biodiversity-ai.git
cd darukaa-biodiversity-ai

# Create virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
*(Optional)* Add your `OPENAI_API_KEY`. If no API key is provided, Darukaa.Earth automatically runs using its **Built-In Deterministic Scientific Reasoning Engine**, allowing full offline hackathon demonstration!

---

## 💻 Running the Application

### Option A: Interactive Streamlit Web Dashboard
Launch the dashboard UI:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Option B: FastAPI Backend REST API
Launch the REST server:
```bash
uvicorn src.api.main:app --reload --port 8000
```
- API Documentation (Swagger UI): `http://localhost:8000/docs`

---

## 🧪 Running Unit Tests

Run the complete pytest test suite:
```bash
pytest -v
```

---

## 📊 Evaluation Criteria Summary

| Evaluation Criteria | Weight | Implementation Details in Darukaa.Earth |
| :--- | :---: | :--- |
| **Depth of Reasoning** | **30%** | Cross-evaluates $\ge 3$ interwoven variables (Soil, Climate, Land Use, Biodiversity). Avoids shallow LLM responses. |
| **Scientific Grounding** | **25%** | RAG retrieval over FAO, IPCC, IPBES papers; detailed ecological mechanism explanations. |
| **Knowledge System Design**| **20%** | Vector index (ChromaDB) with metadata citation tracing and structured fallback search. |
| **Conversational Intelligence**| **15%** | Stateful multi-turn memory; proactive clarification prompt when metrics $< 3$. |
| **Output Clarity** | **10%** | Strict Pydantic JSON/Markdown output formatting with quantitative estimates and timeframes. |

---

## 👥 Repository Access & Submission Info

For evaluation access, please grant repository permissions to:
- `ankita.dasgupta@darukaa.com`
- `harsh.kumar@darukaa.com`
- `utkarsh.gauniyal@darukaa.com`
- `guneet.mutreja@darukaa.com`

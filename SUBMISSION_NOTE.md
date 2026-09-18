# 📝 Darukaa.Earth Submission Summary Note

Use the details below when populating your Word document (.docx) submission:

---

## 1. GitHub Repository Link
`https://github.com/kavitha308/-Darukaa.EarthAI`

*(If private, grant access to: `ankita.dasgupta@darukaa.com`, `harsh.kumar@darukaa.com`, `utkarsh.gauniyal@darukaa.com`, `guneet.mutreja@darukaa.com`)*

---

## 2. Live Demo / Local Run Instructions
- **Streamlit App**: `streamlit run app.py` (Local: `http://localhost:8501`)
- **FastAPI API**: `uvicorn src.api.main:app --reload` (Swagger UI: `http://localhost:8000/docs`)

---

## 3. Architecture & System Highlights
- **Knowledge Layer & RAG**: Vector Store (ChromaDB) indexing FAO, IPCC SRCCL, and IPBES global assessment summaries.
- **Multi-Variable Reasoning Engine**: Evaluates at least 3 interwoven metrics simultaneously (e.g. Soil Organic Carbon + Climate Rainfall + Crop Monoculture -> Agroforestry / Biochar Intercropping).
- **Conversational State & Memory**: Proactively triggers clarifying questions when key input variables are missing.
- **Structured Outputs**: Outputs validated Pydantic JSON/Markdown containing actionable steps, ecological mechanisms, quantitative estimates (e.g., +18-28% SOC over 2-3 years), time horizon, confidence score, and FAO/IPCC citations.
- **Automated Tests**: 100% passing pytest suite covering reasoning logic, clarification prompts, metric extraction, and vector similarity search.

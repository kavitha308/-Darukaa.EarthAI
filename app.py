import os
import sys
import json
import textwrap
import streamlit as st
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import settings
from src.database.schema import EnvironmentalContext, GeoLocation, StructuredRecommendation, ClarificationRequest
from src.engine.memory import ConversationMemory
from src.engine.reasoning import reasoning_engine
from src.database.vector_store import vector_store

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Darukaa.Earth | AI Biodiversity Intelligence System",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Futuristic Dark Bio-Tech Aesthetics & High-Contrast Typography
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800;900&family=Inter:wght@400;500;600;700&display=swap');

    /* Global Theme Overrides */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #0D261E 0%, #071224 45%, #030712 100%) !important;
        font-family: 'Outfit', 'Inter', sans-serif !important;
        color: #FFFFFF !important;
    }

    /* Prevent Code/Pre blocks from rendering as white boxes */
    code, pre, div[data-testid="stCodeBlock"], .stCode, pre code {
        background-color: #0F172A !important;
        background: #0F172A !important;
        color: #00FF9D !important;
        border: 1px solid #1E293B !important;
        border-radius: 10px !important;
        font-family: monospace !important;
    }
    
    code *, pre *, div[data-testid="stCodeBlock"] * {
        background-color: transparent !important;
        color: #00FF9D !important;
    }
    
    /* Ensure ALL Paragraphs, Headings, Subheaders, Labels, List Items, and Spans are Bright White */
    .stApp p, .stApp span, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6, .stApp b, .stApp strong {
        color: #FFFFFF !important;
        font-family: 'Outfit', sans-serif !important;
    }

    /* Form Input Labels */
    label, label[data-testid="stWidgetLabel"], label p, label span, .stSlider label, .stSelectbox label, .stTextInput label, .stNumberInput label {
        color: #38BDF8 !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.2px;
    }

    .stMarkdown p, .stMarkdown span, .stMarkdown div {
        color: #F8FAFC !important;
        font-size: 1.05rem;
    }

    .stCaption, p[data-testid="stCaptionContainer"] {
        color: #94A3B8 !important;
        font-size: 0.95rem;
    }
    
    /* Header Card - Glassmorphism & Neon Accent */
    .main-header {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.22) 0%, rgba(6, 182, 212, 0.18) 50%, rgba(99, 102, 241, 0.15) 100%);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(16, 185, 129, 0.45);
        border-radius: 20px;
        padding: 28px 32px;
        margin-bottom: 24px;
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }
    
    .main-title {
        color: #00FF9D !important;
        font-size: 2.6rem;
        font-weight: 900;
        margin: 0;
        letter-spacing: -0.5px;
        text-shadow: 0 2px 16px rgba(0, 255, 157, 0.4);
    }
    
    .subtitle {
        color: #E2E8F0 !important;
        font-size: 1.1rem;
        margin-top: 6px;
        font-weight: 500;
    }

    /* Stat Banner Cards */
    .stat-card {
        background: rgba(15, 23, 42, 0.75);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-top: 3px solid #10B981;
        border-radius: 14px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4);
        transition: transform 0.2s ease;
    }
    .stat-card:hover {
        transform: translateY(-3px);
        border-top-color: #00FF9D;
    }
    .stat-number {
        font-size: 1.5rem;
        font-weight: 800;
        color: #00FF9D !important;
    }
    .stat-label {
        font-size: 0.85rem;
        color: #94A3B8 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 4px;
    }
    
    /* Buttons Styling - Vibrant Neon Emerald & Glowing Hover */
    .stButton > button, 
    div.stButton > button,
    button[data-testid="baseButton-secondary"],
    button[data-baseweb="button"] {
        background: linear-gradient(135deg, #059669 0%, #10B981 100%) !important;
        color: #FFFFFF !important;
        font-weight: 800 !important;
        font-size: 0.98rem !important;
        border: 1px solid #34D399 !important;
        border-radius: 12px !important;
        padding: 12px 24px !important;
        box-shadow: 0 4px 18px rgba(16, 185, 129, 0.45) !important;
        transition: all 0.25s ease-in-out !important;
        cursor: pointer !important;
    }
    
    .stButton > button p, 
    .stButton > button span,
    div.stButton > button p, 
    div.stButton > button span,
    button[data-testid="baseButton-secondary"] p,
    button[data-testid="baseButton-secondary"] span,
    button[data-baseweb="button"] p,
    button[data-baseweb="button"] span {
        color: #FFFFFF !important;
        font-weight: 800 !important;
        font-size: 0.98rem !important;
    }
    
    .stButton > button:hover, 
    div.stButton > button:hover,
    button[data-testid="baseButton-secondary"]:hover {
        background: linear-gradient(135deg, #047857 0%, #059669 100%) !important;
        color: #FFFFFF !important;
        border-color: #00FF9D !important;
        box-shadow: 0 6px 24px rgba(0, 255, 157, 0.6) !important;
        transform: translateY(-2px);
    }

    /* Form Primary Submit Button */
    button[data-testid="stFormSubmitButton"],
    button[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #0284C7 0%, #10B981 100%) !important;
        background-color: #10B981 !important;
        color: #FFFFFF !important;
        font-weight: 800 !important;
        border: 1px solid #38BDF8 !important;
        border-radius: 12px !important;
        padding: 12px 24px !important;
        box-shadow: 0 4px 18px rgba(14, 165, 233, 0.45) !important;
    }

    button[data-testid="stFormSubmitButton"] p,
    button[data-testid="stFormSubmitButton"] span,
    button[data-testid="baseButton-primary"] p,
    button[data-testid="baseButton-primary"] span {
        color: #FFFFFF !important;
        font-weight: 800 !important;
    }
    
    button[data-testid="stFormSubmitButton"]:hover,
    button[data-testid="baseButton-primary"]:hover {
        background: linear-gradient(135deg, #0369A1 0%, #059669 100%) !important;
        box-shadow: 0 6px 24px rgba(56, 189, 248, 0.6) !important;
    }

    /* Input Controls Text & Background */
    .stTextInput input, .stNumberInput input, div[data-baseweb="select"] {
        background-color: #0F172A !important;
        color: #FFFFFF !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
        font-size: 1rem !important;
    }

    .stTextInput input:focus, .stNumberInput input:focus {
        border-color: #00FF9D !important;
        box-shadow: 0 0 0 2px rgba(0, 255, 157, 0.3) !important;
    }

    /* Badges */
    .badge {
        display: inline-block;
        background: rgba(16, 185, 129, 0.25);
        color: #00FF9D !important;
        border: 1px solid rgba(0, 255, 157, 0.5);
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 700;
        margin-right: 8px;
        box-shadow: 0 2px 10px rgba(0, 255, 157, 0.2);
    }
    
    .badge-cyan {
        background: rgba(6, 182, 212, 0.25);
        color: #38BDF8 !important;
        border: 1px solid rgba(56, 189, 248, 0.5);
        box-shadow: 0 2px 10px rgba(56, 189, 248, 0.2);
    }

    /* Recommendation Card Styling */
    .rec-card {
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.95), rgba(6, 44, 34, 0.9)) !important;
        backdrop-filter: blur(16px);
        border: 1px solid rgba(16, 185, 129, 0.4) !important;
        border-left: 6px solid #00FF9D !important;
        border-radius: 16px !important;
        padding: 28px !important;
        margin-top: 20px !important;
        margin-bottom: 24px !important;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
    }

    .rec-card p, .rec-card span, .rec-card b, .rec-card li, .rec-card div {
        color: #FFFFFF !important;
    }

    .rec-card h2 {
        color: #00FF9D !important;
        font-size: 1.7rem !important;
        font-weight: 900 !important;
        margin-bottom: 14px !important;
        text-shadow: 0 2px 12px rgba(0, 255, 157, 0.3);
    }

    .rec-card h4 {
        color: #38BDF8 !important;
        font-size: 1.25rem !important;
        font-weight: 800 !important;
        margin-top: 20px !important;
        margin-bottom: 10px !important;
    }
    
    .clarification-card {
        background: linear-gradient(145deg, rgba(30, 27, 75, 0.95), rgba(15, 23, 42, 0.9)) !important;
        backdrop-filter: blur(16px);
        border: 1px solid rgba(129, 140, 248, 0.4) !important;
        border-left: 6px solid #818CF8 !important;
        border-radius: 16px !important;
        padding: 28px !important;
        margin-top: 20px !important;
        margin-bottom: 24px !important;
        box-shadow: 0 12px 32px rgba(99, 102, 241, 0.3) !important;
    }

    .clarification-card p, .clarification-card span, .clarification-card b, .clarification-card li, .clarification-card div {
        color: #FFFFFF !important;
    }

    .clarification-card h3 {
        color: #A5B4FC !important;
        font-size: 1.5rem !important;
        font-weight: 900 !important;
        margin-bottom: 14px !important;
    }
    
    .metric-chip {
        background: #0F172A !important;
        color: #FFFFFF !important;
        padding: 8px 16px;
        border-radius: 10px;
        font-size: 0.92rem;
        margin-right: 8px;
        margin-bottom: 8px;
        display: inline-block;
        border: 1px solid #334155;
    }

    /* Chat Messages Styling - Deep Slate Glass Container */
    div[data-testid="stChatMessage"] {
        background: rgba(15, 23, 42, 0.85) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 14px !important;
        margin-bottom: 14px !important;
        padding: 16px 20px !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4) !important;
    }

    div[data-testid="stChatMessage"] p,
    div[data-testid="stChatMessage"] span,
    div[data-testid="stChatMessage"] div,
    div[data-testid="stChatMessage"] h1,
    div[data-testid="stChatMessage"] h2,
    div[data-testid="stChatMessage"] h3,
    div[data-testid="stChatMessage"] h4,
    div[data-testid="stChatMessage"] li {
        color: #FFFFFF !important;
        font-size: 1.05rem !important;
        line-height: 1.65 !important;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 14px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(15, 23, 42, 0.8);
        border-radius: 12px;
        color: #CBD5E1 !important;
        padding: 12px 24px;
        border: 1px solid #1E293B;
        font-weight: 700;
        transition: all 0.25s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #059669 0%, #10B981 100%) !important;
        color: #FFFFFF !important;
        font-weight: 800;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.45);
        border: 1px solid #00FF9D !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Helper Function: Render Evaluation Output Cards
def render_evaluation_response(res):
    if not res:
        return
        
    if isinstance(res, ClarificationRequest):
        card_html = textwrap.dedent(f"""
        <div class="clarification-card">
        <h3>❓ Proactive Scientific Clarification Required</h3>
        <p>{res.clarifying_question}</p>
        <div style="margin-top:16px; background:rgba(0,0,0,0.35); padding:16px; border-radius:10px; border:1px solid rgba(255,255,255,0.1);">
        <b style="color:#A5B4FC !important; font-size:1.05rem;">Detected Metrics So Far:</b> <span style="color:#FFFFFF !important; font-size:1.05rem;">{json.dumps(res.current_detected_metrics)}</span><br><br>
        <b style="color:#FCA5A5 !important; font-size:1.05rem;">Missing Key Variables:</b> <span style="color:#FFFFFF !important; font-size:1.05rem;">{', '.join(res.missing_metrics)}</span>
        </div>
        </div>
        """).strip()
        st.markdown(card_html, unsafe_allow_html=True)
        
    elif isinstance(res, StructuredRecommendation):
        vars_html = ' • '.join([f"<code style='color:#00FF9D !important; background:rgba(0, 255, 157, 0.15) !important; border:1px solid rgba(0, 255, 157, 0.4); padding:6px 12px; border-radius:8px; font-weight:700;'>{v}</code>" for v in res.interwoven_variables_evaluated])
        
        card_html = textwrap.dedent(f"""
        <div class="rec-card">
        <h2>🎯 Recommendation: {res.actionable_recommendation}</h2>
        <p style="margin-bottom:18px;"><b style="color:#F9FAFB !important; font-size:1.05rem;">Time Horizon:</b> <span class="badge">{res.time_horizon}</span> | <b style="color:#F9FAFB !important; font-size:1.05rem;">Confidence:</b> <span class="badge badge-cyan">{res.confidence_level*100:.0f}%</span></p>
        <h4>🔬 Scientific Mechanism & Reasoning:</h4>
        <p style="color:#FFFFFF !important; font-size:1.08rem; line-height:1.7;">{res.scientific_reasoning}</p>
        <h4>🕸️ Interwoven Variables Cross-Evaluated:</h4>
        <p style="color:#FFFFFF !important; margin-top:10px;">{vars_html}</p>
        </div>
        """).strip()
        st.markdown(card_html, unsafe_allow_html=True)
        
        st.markdown("<h3 style='color:#00FF9D !important; margin-top:24px; font-size:1.4rem; font-weight:800;'>📈 Quantitative Impact Estimates</h3>", unsafe_allow_html=True)
        cols = st.columns(len(res.quantitative_estimates))
        for i, est in enumerate(res.quantitative_estimates):
            with cols[i]:
                st.metric(
                    label=est.metric_name,
                    value=est.expected_change,
                    delta=f"Baseline: {est.baseline_value} ({est.timeframe})"
                )

        st.markdown("<h3 style='color:#00FF9D !important; margin-top:28px; font-size:1.4rem; font-weight:800;'>📖 Credible Scientific References & Citations</h3>", unsafe_allow_html=True)
        for cit in res.citations:
            with st.expander(f"📌 {cit.citation_source} - {cit.study_title_or_report}"):
                st.write(f"**Ecological Mechanism**: {cit.ecological_mechanism}")

# Initialize Session State Memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory()
if "last_response" not in st.session_state:
    st.session_state.last_response = None

# Header Display
st.markdown("""
<div class="main-header">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1 class="main-title">🌿 Darukaa.Earth</h1>
            <div class="subtitle">AI Biodiversity Intelligence & Ecological Multi-Variable Reasoning Engine</div>
        </div>
        <div>
            <span class="badge">RAG Vector DB Active</span>
            <span class="badge badge-cyan">Multi-Metric Engine (3+ Var)</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Hero Stat Cards Banner
stat1, stat2, stat3, stat4 = st.columns(4)
with stat1:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{len(vector_store.documents)}</div>
        <div class="stat-label">Scientific Knowledge Chunks</div>
    </div>
    """, unsafe_allow_html=True)
with stat2:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-number">3+ Var</div>
        <div class="stat-label">Interwoven Variable Analysis</div>
    </div>
    """, unsafe_allow_html=True)
with stat3:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-number">FAO / IPCC</div>
        <div class="stat-label">Evidence Grounding</div>
    </div>
    """, unsafe_allow_html=True)
with stat4:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-number">100%</div>
        <div class="stat-label">Actionable & Quantitative</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

# Sidebar System Specs
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/leaf.png", width=64)
    st.title("System Status")
    st.info(f"**Indexed Knowledge**: {len(vector_store.documents)} scientific chunks")
    st.success(f"**Knowledge Sources**: FAO, IPCC, IPBES, Peer-Reviewed Journals")
    
    st.markdown("---")
    st.subheader("Current Detected Context")
    ctx = st.session_state.memory.get_accumulated_context()
    
    metrics_present = {k: v for k, v in ctx.model_dump().items() if v is not None}
    if metrics_present:
        for k, v in metrics_present.items():
            st.markdown(f"<span class='metric-chip'><b>{k}</b>: {v}</span>", unsafe_allow_html=True)
        st.markdown(f"**Total Metrics**: `{ctx.count_provided_metrics()}` / 3 required")
    else:
        st.caption("No environmental metrics detected yet.")
        
    if st.button("🔄 Reset Conversation & State", use_container_width=True):
        st.session_state.memory.clear()
        st.session_state.last_response = None
        st.rerun()

# Main Application Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💬 AI Environmental Scientist",
    "🎛️ Metric Matrix Input",
    "🕸️ Multi-Variable Reasoning",
    "📚 RAG Knowledge Explorer",
    "📊 Impact & Export"
])

# ==========================================
# TAB 1: Conversational AI Scientist
# ==========================================
with tab1:
    st.subheader("Multi-Turn Ecological Intelligence")
    st.caption("Ask natural language queries or provide land details. The AI will evaluate >3 environmental variables or ask clarifying questions if details are missing.")
    
    # Pre-set demo buttons for fast evaluation
    col_demo1, col_demo2, col_demo3 = st.columns(3)
    with col_demo1:
        if st.button("💡 Demo Case 1: Incomplete Input"):
            st.session_state.preset_query = "Biodiversity is declining rapidly on my agricultural farm."
    with col_demo2:
        if st.button("⚡ Demo Case 2: Full Multi-Metric Input"):
            st.session_state.preset_query = "My soil organic carbon is 0.3%, rainfall is low in a semi-arid region, and I grow monoculture wheat. How can I boost biodiversity?"
    with col_demo3:
        if st.button("🧪 Demo Case 3: Pesticide & Fragmentation"):
            st.session_state.preset_query = "High pesticide load with severe habitat fragmentation and 4.8 soil pH."

    # Query Input
    user_input = st.text_input(
        "Enter your environmental query or site description:",
        value=st.session_state.get("preset_query", ""),
        placeholder="e.g., My soil organic carbon is 0.4%, rainfall is low, and crop is monoculture wheat in semi-arid region."
    )
    
    if st.button("Evaluate Ecosystem", type="primary"):
        if user_input.strip():
            with st.spinner("Analyzing multi-variable environmental interactions and searching scientific knowledge base..."):
                st.session_state.memory.add_user_message(user_input)
                accumulated_ctx = st.session_state.memory.get_accumulated_context()
                
                response = reasoning_engine.evaluate(
                    user_query=user_input,
                    context=accumulated_ctx,
                    chat_history=st.session_state.memory.get_history()
                )
                
                st.session_state.last_response = response
                if isinstance(response, StructuredRecommendation):
                    st.session_state.memory.add_assistant_message(f"Recommendation: {response.actionable_recommendation}")
                else:
                    st.session_state.memory.add_assistant_message(response.clarifying_question)

    # Render Current Response
    res = st.session_state.last_response
    if res:
        render_evaluation_response(res)

    # Chat History
    st.markdown("---")
    st.subheader("📜 Conversation History")
    for msg in st.session_state.memory.get_history():
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            st.chat_message("assistant").write(msg["content"])

# ==========================================
# TAB 2: Structured Metric Input
# ==========================================
with tab2:
    st.subheader("Structured Environmental Metric Matrix")
    st.caption("Directly input quantitative parameters across 5 ecological categories.")
    
    with st.form("structured_metric_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 1. Soil Health")
            soc = st.number_input("Soil Organic Carbon (SOC %)", min_value=0.0, max_value=10.0, value=0.3, step=0.1)
            ph = st.number_input("Soil pH", min_value=1.0, max_value=14.0, value=5.5, step=0.1)
            moisture = st.selectbox("Soil Moisture", ["low", "moderate", "high"])
            microbial = st.selectbox("Microbial Activity", ["depleted", "moderate", "high"])
            
        with col2:
            st.markdown("### 2. Land Cover & Use")
            land_use = st.selectbox("Land Use Type", ["monoculture", "agroforestry", "pastoral", "degraded"])
            crop = st.text_input("Primary Crop", value="wheat")
            monoculture_pct = st.slider("Monoculture Share (%)", 0, 100, 85)
            fragmentation = st.selectbox("Habitat Fragmentation", ["low", "moderate", "high", "severe"])

        with col3:
            st.markdown("### 3. Climate & Human Impact")
            rainfall = st.selectbox("Rainfall Pattern", ["low", "erratic", "moderate", "high"])
            aridity = st.selectbox("Aridity Index", ["semi-arid", "hyper-arid", "sub-humid", "humid"])
            pesticide = st.selectbox("Pesticide Load", ["none", "low", "moderate", "high"])
            
            st.markdown("### 4. Geo-Coordinates (Optional)")
            lat = st.number_input("Latitude", value=28.6139, format="%.4f")
            lon = st.number_input("Longitude", value=77.2090, format="%.4f")

        submit_structured = st.form_submit_button("Run Multi-Variable Evaluation", type="primary")

    if submit_structured:
        new_ctx = EnvironmentalContext(
            soil_organic_carbon_pct=soc,
            soil_ph=ph,
            soil_moisture=moisture,
            microbial_activity=microbial,
            land_use_type=land_use,
            crop_type=crop,
            monoculture_pct=monoculture_pct,
            habitat_fragmentation=fragmentation,
            rainfall_pattern=rainfall,
            aridity_index=aridity,
            pesticide_load=pesticide,
            location=GeoLocation(latitude=lat, longitude=lon, region_name="Custom Site")
        )
        st.session_state.memory.update_context(new_ctx)
        
        eval_res = reasoning_engine.evaluate(
            user_query="Evaluate input metrics",
            context=st.session_state.memory.get_accumulated_context()
        )
        st.session_state.last_response = eval_res
        st.success("Structured metric matrix evaluated! View evidence-backed recommendations below:")
        render_evaluation_response(eval_res)

# ==========================================
# TAB 3: Multi-Variable Reasoning Matrix
# ==========================================
with tab3:
    st.subheader("Interwoven 3+ Variable Ecological Reasoning Graph")
    st.markdown("""
    Darukaa.Earth's core differentiator is **Multi-Metric Interwoven Reasoning**. 
    Single-variable answers (e.g. only looking at pH or only at crop type) yield shallow recommendations.
    Below is the interaction matrix evaluated for your current environmental context:
    """)
    
    st.image("https://mermaid.ink/svg/pTk2Q-30", use_column_width=True, caption="Multi-Metric Cross-Evaluation Flow")
    
    ctx = st.session_state.memory.get_accumulated_context()
    st.markdown("### Current Active Variables evaluated in parallel:")
    st.json(ctx.model_dump(exclude_none=True))

# ==========================================
# TAB 4: RAG Knowledge Base Explorer
# ==========================================
with tab4:
    st.subheader("Retrieval-Augmented Generation (RAG) Document Explorer")
    st.caption("Directly query the indexed FAO, IPCC, IPBES scientific document store and view vector matches.")
    
    rag_query = st.text_input("Search scientific vector index:", value="agroforestry soil organic carbon rainfall semi-arid")
    if st.button("Search Vector DB"):
        results = vector_store.search_similar(rag_query, top_k=4)
        for idx, doc in enumerate(results, 1):
            st.markdown(f"#### Match #{idx} | Similarity Score: `{doc['score']}`")
            st.info(f"**Metadata**: {doc['metadata']}")
            st.text(doc["content"])
            st.markdown("---")

# ==========================================
# TAB 5: Quantitative Impact & JSON Export
# ==========================================
with tab5:
    st.subheader("Quantitative Impact Visualization & Export")
    
    res = st.session_state.last_response
    if isinstance(res, StructuredRecommendation):
        # Create chart from quantitative estimates
        data = []
        for est in res.quantitative_estimates:
            data.append({
                "Metric": est.metric_name,
                "Expected Impact": est.expected_change,
                "Timeframe": est.timeframe
            })
        df = pd.DataFrame(data)
        st.table(df)
        
        st.subheader("Download Structured JSON Report")
        st.download_button(
            label="📥 Download Structured Report (JSON)",
            data=json.dumps(res.model_dump(), indent=2),
            file_name="darukaa_biodiversity_recommendation.json",
            mime="application/json"
        )
    else:
        st.info("Run an evaluation in Tab 1 or Tab 2 to generate quantitative charts and export JSON.")

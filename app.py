"""
🎫 Smart Ticket Understanding Engine — Streamlit App
Main entry point for the ticket classifier application.
Includes Gemini LLM classification, RAG root cause analysis, and SLA reports.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import json
import time

# Project root
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from models.predictor import TicketPredictor
from llm.gemini_classifier import GeminiTicketClassifier
from rag.root_cause_analyzer import RootCauseAnalyzer
from rag.vector_store import TicketVectorStore
from utils.actions import CATEGORY_TO_DEPARTMENT, get_recommended_action, get_sla_hours
from utils.metrics import calculate_metrics, get_dataframe

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Ticket Engine",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }
    .main-header { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem; border-radius: 1rem; color: white; margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 { color: white !important; font-size: 2.2rem; margin: 0; }
    .main-header p { color: rgba(255,255,255,0.85); font-size: 1.1rem; margin: 0.5rem 0 0 0; }
    .result-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.2rem; border-radius: 0.8rem; margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    .action-box {
        background: linear-gradient(135deg, #f3e8ff, #e9d5ff);
        padding: 1rem; border-radius: 0.8rem; border-left: 4px solid #7c3aed;
        margin-top: 1rem; color: #4c1d95;
    }
    .reasoning-box {
        background: #f8fafc;
        padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #3b82f6;
        margin-top: 1rem; font-style: italic; color: #334155;
    }
    .outlier-box {
        background: #fef2f2;
        padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #ef4444;
        margin-top: 1rem; color: #991b1b;
        font-weight: bold;
    }
    .metric-card {
        background: white; border-radius: 8px; padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05); text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_ml_predictor():
    """Load the fallback ML prediction engine."""
    return TicketPredictor()

@st.cache_resource
def load_llm_classifier():
    """Load Gemini classifier."""
    return GeminiTicketClassifier()

@st.cache_resource
def load_rag_analyzer():
    """Load RAG analyzer."""
    return RootCauseAnalyzer()


def priority_emoji(p):
    return {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}.get(p, "⚪")

def sentiment_emoji(s):
    return {"Urgent": "🚨", "Frustrated": "😤", "Neutral": "😐", "Positive": "😊"}.get(s, "❓")


def render_header():
    st.markdown("""
    <div class="main-header">
        <h1>🎫 Smart Ticket Understanding Engine</h1>
        <p>AI-powered IT service desk ticket classification, root cause analysis & routing</p>
    </div>
    """, unsafe_allow_html=True)


def render_smart_classify(llm, ml_fallback):
    """Smart Classification with Reasoning (Page 1)"""
    st.subheader("🤖 Smart Classify (with LLM Reasoning)")
    st.markdown("Uses Gemini to classify tickets and provide an explanation. Falls back to ML if API is unavailable.")
    
    samples = [
        "Unable to connect VPN since morning, urgent client call in 20 mins.",
        "The office air conditioning is broken in meeting room A.",
        "SAP is crashing every time I try to generate a report.",
        "Locked out of my account after too many wrong password attempts.",
        "Received a suspicious phishing email claiming to be from HR.",
    ]
    
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🎲 Try Sample", use_container_width=True):
            import random
            st.session_state['sample_ticket'] = random.choice(samples)
            
    default_text = st.session_state.get('sample_ticket', samples[0])
    ticket_text = st.text_area(
        "Enter ticket / email / chat message:",
        value=default_text, height=120
    )
    
    if st.button("🚀 Classify Ticket", type="primary", use_container_width=True):
        if not ticket_text.strip():
            st.warning("Please enter a ticket to analyze.")
            return
            
        with st.spinner("Analyzing ticket..."):
            is_llm = False
            if llm.is_ready():
                result = llm.classify_ticket(ticket_text)
                if result:
                    is_llm = True
            
            if not is_llm:
                st.warning("Gemini API not configured or failed. Falling back to ML model.")
                if not ml_fallback.is_ready():
                    st.error("ML fallback not trained. Run `python models/train.py`.")
                    return
                # Use ML fallback
                ml_res = ml_fallback.predict(ticket_text)
                result = {
                    "category": ml_res["category"],
                    "priority": ml_res["priority"],
                    "department": ml_res["department"],
                    "sentiment": ml_res["sentiment"],
                    "reasoning": "Reasoning unavailable (using ML fallback).",
                    "is_outlier": False
                }
            
            # Action logic
            action = get_recommended_action(result['category'], result['priority'], result.get('sentiment', 'Neutral'))
            sla = get_sla_hours(result['priority'])
            
            # Display results
            st.success(f"✅ Analysis Complete! (Powered by {'Gemini LLM' if is_llm else 'Scikit-Learn'})")
            
            if result.get('is_outlier', False) or result['category'] == "Outlier":
                st.markdown(f"""
                <div class="outlier-box">
                    ⚠️ OUTLIER DETECTED: This ticket does not match standard IT categories.
                </div>
                """, unsafe_allow_html=True)
                
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("📁 Category", result.get('category', 'Unknown'))
            with c2:
                pri = result.get('priority', 'Medium')
                st.metric(f"{priority_emoji(pri)} Priority", pri)
            with c3:
                st.metric("🏢 Route To", result.get('department', 'Service Desk'))
            with c4:
                sent = result.get('sentiment', 'Neutral')
                st.metric(f"{sentiment_emoji(sent)} Sentiment", sent)
                
            st.markdown(f"""
            <div class="reasoning-box">
                <strong>🧠 AI Reasoning:</strong><br/>
                {result.get('reasoning', 'No reasoning provided.')}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="action-box">
                <strong>🎯 Recommended Action (SLA: {sla}h)</strong><br/>
                {action}
            </div>
            """, unsafe_allow_html=True)


def render_root_cause_analyzer(rag):
    """RAG-based Root Cause Analyzer (Page 2)"""
    st.subheader("🔎 Root Cause Analyzer (RAG)")
    st.markdown("Searches historical resolved tickets to suggest root causes and step-by-step resolutions.")
    
    if not rag.vector_store.is_ready:
        st.error("ChromaDB vector store is not initialized. Please install 'chromadb'.")
        return
        
    ticket_text = st.text_area("Describe the new issue:", height=100, placeholder="E.g., Laptop won't connect to Wi-Fi on the 3rd floor.")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🔍 Find Root Cause", type="primary"):
            if not ticket_text.strip():
                st.warning("Please enter a ticket description.")
                return
                
            with st.spinner("Searching vector database & synthesizing resolution..."):
                res = rag.analyze(ticket_text)
                
            st.markdown("### 💡 AI Diagnosis")
            st.info(f"**Likely Root Cause:**\n\n{res.get('likely_root_cause', 'Unknown')}")
            st.success(f"**Suggested Resolution:**\n\n{res.get('suggested_resolution', 'Unknown')}")
            st.caption(f"Confidence: **{res.get('confidence', 'Unknown')}**")
            
            if res.get('similar_tickets'):
                with st.expander("📚 View Similar Historical Tickets Used as Context"):
                    for i, sim in enumerate(res['similar_tickets']):
                        st.markdown(f"**Ticket #{i+1}** (Distance: {sim['distance']:.4f})")
                        st.markdown(f"> {sim['ticket_text']}")
                        st.markdown(f"*Root Cause:* {sim['metadata'].get('root_cause')}")
                        st.markdown(f"*Fix:* {sim['metadata'].get('resolution_steps')}")
                        st.divider()
    with col2:
        st.markdown("### Database Operations")
        if st.button("🔄 Rebuild Vector Database from CSV"):
            csv_path = os.path.join(ROOT, "data", "tickets.csv")
            if os.path.exists(csv_path):
                with st.spinner("Embedding historical tickets... this may take a minute."):
                    success = rag.vector_store.build_from_csv(csv_path)
                if success:
                    st.success("Vector database rebuilt successfully!")
                else:
                    st.error("Failed to build database. Check console logs.")
            else:
                st.error("tickets.csv not found.")


def render_dashboard():
    """Analytics Dashboard (Page 3)"""
    st.subheader("📊 Advanced Analytics Dashboard")
    
    csv_path = os.path.join(ROOT, "data", "tickets.csv")
    if not os.path.exists(csv_path):
        st.warning("No data found. Please generate the dataset first.")
        return
        
    df = get_dataframe(csv_path)
    if df is None:
        st.error("Error loading dataframe.")
        return
        
    # Top metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card" style="border-top: 4px solid #667eea;">
            <p style="color:#64748b; font-size:0.9rem; margin:0; text-transform:uppercase; font-weight:bold;">Total Volume</p>
            <h1 style="color:#1e293b; margin:0; font-size:2.5rem;">{len(df)}</h1>
            </div>""", unsafe_allow_html=True)
    with c2:
        open_tix = len(df[df['ticket_status'] == 'Open']) if 'ticket_status' in df.columns else 0
        st.markdown(f"""<div class="metric-card" style="border-top: 4px solid #ed8936;">
            <p style="color:#64748b; font-size:0.9rem; margin:0; text-transform:uppercase; font-weight:bold;">Active Tickets</p>
            <h1 style="color:#1e293b; margin:0; font-size:2.5rem;">{open_tix}</h1>
            </div>""", unsafe_allow_html=True)
    with c3:
        high_pri = len(df[df['priority'].isin(['Critical', 'High'])])
        st.markdown(f"""<div class="metric-card" style="border-top: 4px solid #e53e3e;">
            <p style="color:#64748b; font-size:0.9rem; margin:0; text-transform:uppercase; font-weight:bold;">Critical/High Alerts</p>
            <h1 style="color:#1e293b; margin:0; font-size:2.5rem;">{high_pri}</h1>
            </div>""", unsafe_allow_html=True)
    with c4:
        closed = len(df[df['ticket_status'] == 'Closed']) if 'ticket_status' in df.columns else 0
        st.markdown(f"""<div class="metric-card" style="border-top: 4px solid #48bb78;">
            <p style="color:#64748b; font-size:0.9rem; margin:0; text-transform:uppercase; font-weight:bold;">Resolved</p>
            <h1 style="color:#1e293b; margin:0; font-size:2.5rem;">{closed}</h1>
            </div>""", unsafe_allow_html=True)
            
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Time Series Chart
    if 'created_at' in df.columns:
        st.markdown("### 📈 Ticket Influx Over Time")
        daily = df.groupby(df['created_at'].dt.date).size().reset_index(name='Tickets')
        fig_time = px.area(daily, x='created_at', y='Tickets', 
                          color_discrete_sequence=['#667eea'],
                          line_shape='spline')
        fig_time.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_time, use_container_width=True)
    
    # Charts
    st.markdown("### 🎯 Distribution & Status")
    col1, col2, col3 = st.columns(3)
    with col1:
        fig1 = px.pie(df, names='category', hole=0.5,
                      color_discrete_sequence=px.colors.qualitative.Pastel)
        fig1.update_layout(title_text='Category Breakdown', title_x=0.5)
        st.plotly_chart(fig1, use_container_width=True)
        
    with col2:
        if 'ticket_status' in df.columns:
            fig2 = px.pie(df, names='ticket_status', 
                          color_discrete_map={'Closed':'#48bb78', 'Open':'#ed8936', 'Reopened':'#e53e3e'}, hole=0.5)
            fig2.update_layout(title_text='Status Distribution', title_x=0.5)
            st.plotly_chart(fig2, use_container_width=True)
            
    with col3:
        if 'department' in df.columns:
            dept_counts = df['department'].value_counts().reset_index()
            fig3 = px.bar(dept_counts, x='department', y='count', 
                          color='department', color_discrete_sequence=px.colors.qualitative.Set2)
            fig3.update_layout(title_text='Department Workload', title_x=0.5, showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)
            
    st.markdown("### 🚨 Urgent Priority Backlog")
    if 'ticket_status' in df.columns:
        high_open = df[(df['priority'].isin(['Critical', 'High'])) & (df['ticket_status'] == 'Open')]
        if not high_open.empty:
            st.dataframe(high_open[['ticket_id', 'category', 'priority', 'department', 'created_at']], use_container_width=True)
        else:
            st.success("No critical open tickets! Great job team! 🎉")


def render_mttr_reports():
    """MTTR/MTTM Reports (Page 4)"""
    st.subheader("📈 MTTR & Resolution Reports")
    
    csv_path = os.path.join(ROOT, "data", "tickets.csv")
    if not os.path.exists(csv_path):
        st.warning("No data found.")
        return
        
    metrics = calculate_metrics(csv_path)
    if not metrics:
        st.error("Could not calculate metrics. Check dataset format.")
        return
        
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Overall MTTR (Mean Time To Resolve)", f"{metrics.get('overall_mttr_hours', 0)} hours")
    with c2:
        st.metric("Overall MTTM (Mean Time To Mitigate)", f"{metrics.get('overall_mttm_hours', 0)} hours")
    with c3:
        st.metric("Resolution Success Rate", f"{metrics.get('resolution_success_rate', 0)}%")
        
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### MTTR by Category (Hours)")
        cat_mttr = metrics.get('mttr_by_category', {})
        if cat_mttr:
            fig = go.Figure(data=[go.Bar(x=list(cat_mttr.keys()), y=list(cat_mttr.values()), marker_color='#667eea')])
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
    with col2:
        st.markdown("### MTTR by Priority (Hours)")
        pri_mttr = metrics.get('mttr_by_priority', {})
        if pri_mttr:
            fig = go.Figure(data=[go.Bar(x=list(pri_mttr.keys()), y=list(pri_mttr.values()), 
                                         marker_color=['#e53e3e', '#ed8936', '#ecc94b', '#48bb78'])])
            st.plotly_chart(fig, use_container_width=True)


def render_csv_upload(llm, ml_fallback):
    """CSV bulk upload and prediction page."""
    st.subheader("📂 Bulk CSV Upload")
    
    uploaded = st.file_uploader("Choose a CSV file", type=['csv'])
    if uploaded:
        df = pd.read_csv(uploaded)
        text_col = next((col for col in ['ticket_text', 'text', 'description'] if col in df.columns), None)
        
        if not text_col:
            st.error(f"Could not find a text column. Available: {list(df.columns)}")
            return
            
        st.dataframe(df.head(), use_container_width=True)
        
        st.info("⚡ Bulk upload uses our lightning-fast Scikit-Learn ML models to process hundreds of tickets instantly without hitting any API rate limits.")
        
        if st.button("🚀 Classify All Tickets", type="primary"):
            progress = st.progress(0, text="Classifying with ML models...")
            results = []
            
            process_df = df
            for i, text in enumerate(process_df[text_col]):
                ml_res = ml_fallback.predict(str(text))
                results.append({
                    "ticket_text": str(text),
                    "category": ml_res["category"],
                    "priority": ml_res["priority"],
                    "department": ml_res["department"],
                    "reasoning": "Determined via ML Model"
                })
                progress.progress((i + 1) / len(process_df), text=f"Classified {i+1}/{len(process_df)}")
                
            res_df = pd.DataFrame(results)
            st.success(f"✅ Successfully classified {len(res_df)} tickets using ML!")
            st.dataframe(res_df, use_container_width=True)
            
            csv_data = res_df.to_csv(index=False)
            st.download_button("📥 Download Results CSV", csv_data, "ticket_predictions.csv", "text/csv")


def render_sidebar():
    """Sidebar with navigation and model info."""
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/ticket.png", width=60)
        st.title("Navigation")
        page = st.radio("Go to:", [
            "🤖 Smart Classify (LLM)",
            "🔎 Root Cause Analyzer",
            "📊 Analytics Dashboard",
            "📈 MTTR & Reports",
            "📂 CSV Bulk Upload",
        ], label_visibility="collapsed")
        
        st.markdown("---")
        
        # Setup Status
        st.markdown("**System Status**")
        
        has_gemini = os.getenv("GOOGLE_API_KEY") is not None
        st.markdown(f"{'✅' if has_gemini else '❌'} Gemini API")
        
        has_chroma = False
        try:
            import chromadb
            has_chroma = True
        except:
            pass
        st.markdown(f"{'✅' if has_chroma else '❌'} ChromaDB")
        
        st.markdown("---")
        return page


def main():
    render_header()
    page = render_sidebar()
    
    # Init engines
    ml_predictor = load_ml_predictor()
    llm_classifier = load_llm_classifier()
    rag_analyzer = load_rag_analyzer()
    
    if page == "🤖 Smart Classify (LLM)":
        render_smart_classify(llm_classifier, ml_predictor)
    elif page == "🔎 Root Cause Analyzer":
        render_root_cause_analyzer(rag_analyzer)
    elif page == "📊 Analytics Dashboard":
        render_dashboard()
    elif page == "📈 MTTR & Reports":
        render_mttr_reports()
    elif page == "📂 CSV Bulk Upload":
        render_csv_upload(llm_classifier, ml_predictor)

if __name__ == "__main__":
    main()

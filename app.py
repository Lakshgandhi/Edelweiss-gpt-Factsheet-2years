import streamlit as st
import os
import numpy as np
import fitz  # PyMuPDF
import faiss
import requests
import io
from sentence_transformers import SentenceTransformer
from groq import Groq

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Edelweiss Factsheet GPT",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
    .stApp { background: #050d1a; }
    [data-testid="stSidebar"] { background: #070f1f !important; border-right: 1px solid #0d2137; }
    .main-header {
        background: linear-gradient(135deg, #071628 0%, #0a2040 50%, #071628 100%);
        border: 1px solid #1a4a6b; border-radius: 16px;
        padding: 28px 36px; margin-bottom: 24px;
    }
    .main-header h1 { color: #e8f4fd; font-size: 2rem; font-weight: 700; margin: 0 0 6px 0; }
    .main-header p { color: #5b8fa8; margin: 0; font-size: 0.95rem; }
    .badge {
        display: inline-block; background: rgba(0,180,255,0.12); color: #00b4ff;
        border: 1px solid rgba(0,180,255,0.25); padding: 3px 12px;
        border-radius: 20px; font-size: 0.75rem; font-weight: 600; margin-top: 10px;
        font-family: 'JetBrains Mono', monospace;
    }
    .user-msg {
        background: linear-gradient(135deg, #0d2744, #0a1f36);
        border: 1px solid #1a3d5c; border-radius: 14px 14px 4px 14px;
        padding: 14px 18px; margin: 10px 0; color: #c8e0f0; line-height: 1.6;
    }
    .user-label { color: #00b4ff; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; font-family: 'JetBrains Mono', monospace; }
    .bot-msg {
        background: linear-gradient(135deg, #0d1f12, #0a1a0f);
        border: 1px solid #1a3d22; border-radius: 14px 14px 14px 4px;
        padding: 14px 18px; margin: 10px 0; color: #c0ddc8; line-height: 1.7;
    }
    .bot-label { color: #00e676; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; font-family: 'JetBrains Mono', monospace; }
    .source-tag {
        display: inline-block; background: rgba(255,215,0,0.08); color: #ffd700;
        border: 1px solid rgba(255,215,0,0.2); padding: 2px 8px;
        border-radius: 4px; font-size: 0.72rem; margin: 2px; font-family: 'JetBrains Mono', monospace;
    }
    .stat-card { background: linear-gradient(135deg, #071628, #0a1f30); border: 1px solid #1a3d5c; border-radius: 12px; padding: 16px 20px; text-align: center; }
    .stat-number { color: #00b4ff; font-size: 1.8rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
    .stat-label { color: #4a7a96; font-size: 0.8rem; margin-top: 4px; }
    .welcome-box { background: linear-gradient(135deg, #071628, #0a1a2e); border: 1px dashed #1a3d5c; border-radius: 16px; padding: 40px; text-align: center; color: #4a7a96; }
    .welcome-box h3 { color: #5ba8c4; margin-bottom: 12px; }
    .stButton button { background: linear-gradient(135deg, #005f8a, #007ab5) !important; color: white !important; border: none !important; border-radius: 10px !important; font-weight: 600 !important; }
    .stTextInput input { background: #071628 !important; border: 1px solid #1a3d5c !important; color: #c8e0f0 !important; border-radius: 10px !important; }
    hr { border-color: #0d2137 !important; }
    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# GOOGLE DRIVE FOLDER ID
# ─────────────────────────────────────────────
FOLDER_ID = "1uAn28AcjEa0rCIKqNpNc1tUPp-UBqrrD"

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

# ─────────────────────────────────────────────
# FETCH PDF LIST FROM GOOGLE DRIVE
# ─────────────────────────────────────────────
def get_pdf_files_from_drive(folder_id):
    """Get list of PDF files from public Google Drive folder"""
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    api_url = f"https://www.googleapis.com/drive/v3/files?q='{folder_id}'+in+parents+and+mimeType='application/pdf'&fields=files(id,name)&key=AIzaSyD-placeholder"
    
    # Use the export URL approach for public folders
    folder_url = f"https://drive.google.com/drive/folders/{folder_id}"
    
    # Scrape folder listing
    import re
    response = requests.get(folder_url)
    # Extract file IDs from the page
    file_ids = re.findall(r'"(1[a-zA-Z0-9_-]{28,})"', response.text)
    file_ids = list(set(file_ids))
    
    return file_ids

def download_pdf_from_drive(file_id):
    """Download a PDF from Google Drive by file ID"""
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(url, allow_redirects=True)
    if response.status_code == 200 and len(response.content) > 1000:
        return response.content
    # Try alternate URL
    url2 = f"https://drive.google.com/uc?id={file_id}&export=download"
    response2 = requests.get(url2, allow_redirects=True)
    if response2.status_code == 200:
        return response2.content
    return None

def extract_text_from_bytes(pdf_bytes, filename="file"):
    """Extract text from PDF bytes"""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = "".join([page.get_text() for page in doc])
        doc.close()
        return text
    except:
        return ""

# ─────────────────────────────────────────────
# BUILD KNOWLEDGE BASE FROM DRIVE
# ─────────────────────────────────────────────
@st.cache_resource
def build_knowledge_base_from_drive(folder_id):
    embedder = load_embedder()
    all_chunks = []
    doc_count = 0

    # Get individual file IDs shared by the user
    # Since folder scraping may be unreliable, use pre-listed file approach
    st.info("📥 Fetching PDFs from Google Drive... This takes 2-3 minutes on first load.")
    
    file_ids = get_pdf_files_from_drive(folder_id)
    
    if not file_ids:
        return None, None, 0, 0

    progress = st.progress(0)
    status = st.empty()
    
    for i, file_id in enumerate(file_ids[:30]):  # max 30 files
        try:
            status.text(f"📄 Loading file {i+1}/{len(file_ids[:30])}...")
            pdf_bytes = download_pdf_from_drive(file_id)
            if pdf_bytes:
                text = extract_text_from_bytes(pdf_bytes, file_id)
                if len(text) > 500:  # valid PDF with content
                    words = text.split()
                    for j in range(0, len(words), 800):
                        chunk = ' '.join(words[j:j+1000])
                        if chunk.strip():
                            all_chunks.append({'text': chunk, 'source': f"Factsheet_{i+1}"})
                    doc_count += 1
            progress.progress((i+1) / len(file_ids[:30]))
        except:
            continue

    progress.empty()
    status.empty()

    if not all_chunks:
        return None, None, 0, 0

    embeddings = embedder.encode([c['text'] for c in all_chunks], show_progress_bar=False)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings, dtype='float32'))

    return index, all_chunks, doc_count, len(all_chunks)

# ─────────────────────────────────────────────
# RETRIEVE & ASK
# ─────────────────────────────────────────────
def retrieve_chunks(query, index, chunks, embedder, top_k=20):
    q_emb = embedder.encode([query])
    _, indices = index.search(np.array(q_emb, dtype='float32'), top_k)
    results = [chunks[i] for i in indices[0] if i < len(chunks)]
    seen_sources = {}
    diverse_results = []
    for chunk in results:
        src = chunk['source']
        seen_sources[src] = seen_sources.get(src, 0)
        if seen_sources[src] < 1:
            words = chunk['text'].split()[:300]
            diverse_results.append({'text': ' '.join(words), 'source': src})
            seen_sources[src] += 1
    return diverse_results[:12]

def ask_groq(question, context_chunks, api_key):
    client = Groq(api_key=api_key)
    context = "\n\n".join([f"[{c['source']}]\n{c['text']}" for c in context_chunks])
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are an expert financial analyst for Edelweiss Mutual Fund with access to 26 monthly factsheets from Feb 2024 to Feb 2026. Answer questions using ALL the provided context. Always cite which report each data point comes from. Be precise with numbers and percentages. Synthesize information across multiple months."},
            {"role": "user", "content": f"Factsheet context:\n{context}\n\nQuestion: {question}"}
        ],
        max_tokens=1000
    )
    return response.choices[0].message.content

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
# Auto-load API key from Streamlit secrets
groq_api_key = st.secrets["GROQ_API_KEY"]

with st.sidebar:
    st.markdown("### 📊 Edelweiss Factsheet GPT")
    st.markdown("---")
    st.markdown("<div style='color:#00e676;font-size:0.85rem;'>✅ API Connected</div>", unsafe_allow_html=True)
    load_btn = False  # No button needed - auto loads
    st.markdown("---")
    st.markdown("### 💡 Sample Questions")
    for q in ["Best performing fund in 2024?", "Top holdings across all months?", "Sector allocation trend?", "AUM growth over 2 years?", "Which fund beat its benchmark?", "Equity vs debt allocation?"]:
        st.markdown(f"<div style='background:rgba(0,180,255,0.07);border:1px solid rgba(0,180,255,0.2);color:#5ba8c4;padding:6px 14px;border-radius:20px;font-size:0.8rem;margin:4px 0;'>💬 {q}</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<div style='color:#2a4a5c;font-size:0.75rem;text-align:center;'>Powered by Groq × Llama 3 × FAISS<br>Built for Edelweiss Research</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
st.markdown("""
<div class='main-header'>
    <h1>📊 Edelweiss Factsheet GPT</h1>
    <p>AI-powered research assistant — 26 months of factsheets, instant answers</p>
    <span class='badge'>⚡ Powered by Groq · Llama 3 · FAISS · Google Drive</span>
</div>
""", unsafe_allow_html=True)

if "messages" not in st.session_state: st.session_state.messages = []
if "kb_ready" not in st.session_state: st.session_state.kb_ready = False
if "index" not in st.session_state: st.session_state.index = None
if "chunks" not in st.session_state: st.session_state.chunks = None
if "doc_count" not in st.session_state: st.session_state.doc_count = 0
if "chunk_count" not in st.session_state: st.session_state.chunk_count = 0

# Auto-load on startup if not already loaded
if not st.session_state.kb_ready:
    with st.spinner("📚 Loading 26 months of Edelweiss factsheets... Please wait 2-3 minutes..."):
        index, chunks, doc_count, chunk_count = build_knowledge_base_from_drive(FOLDER_ID)
    if index is None:
        st.error("❌ Could not load PDFs from Google Drive. Please refresh the page.")
    else:
        st.session_state.index = index
        st.session_state.chunks = chunks
        st.session_state.doc_count = doc_count
        st.session_state.chunk_count = chunk_count
        st.session_state.kb_ready = True
        st.rerun()

if st.session_state.kb_ready:
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f"<div class='stat-card'><div class='stat-number'>{st.session_state.doc_count}</div><div class='stat-label'>Factsheets Loaded</div></div>", unsafe_allow_html=True)
    with c2: st.markdown(f"<div class='stat-card'><div class='stat-number'>{st.session_state.chunk_count:,}</div><div class='stat-label'>Chunks Indexed</div></div>", unsafe_allow_html=True)
    with c3: st.markdown(f"<div class='stat-card'><div class='stat-number'>{len(st.session_state.messages)//2}</div><div class='stat-label'>Questions Asked</div></div>", unsafe_allow_html=True)
    with c4: st.markdown("<div class='stat-card'><div class='stat-number' style='color:#00e676'>●</div><div class='stat-label'>System Ready</div></div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

chat_container = st.container()
with chat_container:
    if not st.session_state.kb_ready:
        st.markdown("""
        <div class='welcome-box'>
            <h3>👋 Welcome to Edelweiss Factsheet GPT</h3>
            <p>All 26 months of factsheets are stored on Google Drive.</p>
            <p>1️⃣ &nbsp; Enter your <b>Groq API key</b> in the sidebar</p>
            <p>2️⃣ &nbsp; Click <b>Load Factsheets from Drive</b></p>
            <p>3️⃣ &nbsp; Start asking questions!</p>
            <p style='margin-top:20px;font-size:0.85rem;color:#3a6a7c;'>No file uploads needed — PDFs load automatically from Google Drive!</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        if not st.session_state.messages:
            st.markdown("<div class='welcome-box'><h3>✅ Knowledge Base Ready!</h3><p>Ask me anything about your Edelweiss factsheets below 👇</p></div>", unsafe_allow_html=True)
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"<div class='user-msg'><div class='user-label'>🧑 You</div>{msg['content']}</div>", unsafe_allow_html=True)
            else:
                sources_html = "".join([f"<span class='source-tag'>📄 {s}</span>" for s in msg.get("sources", [])])
                st.markdown(f"<div class='bot-msg'><div class='bot-label'>🤖 Edelweiss GPT</div>{msg['content'].replace(chr(10), '<br>')}<div style='margin-top:10px;'>{sources_html}</div></div>", unsafe_allow_html=True)

if st.session_state.kb_ready:
    st.markdown("---")
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input("Ask a question", placeholder="e.g. Which fund had highest returns in 2024?", label_visibility="collapsed", key="user_input")
    with col2:
        send = st.button("Ask ➤", use_container_width=True)
    if send and user_input.strip():
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("🔍 Searching factsheets..."):
            embedder = load_embedder()
            chunks = retrieve_chunks(user_input, st.session_state.index, st.session_state.chunks, embedder)
            sources = list(set([c['source'] for c in chunks]))
            answer = ask_groq(user_input, chunks, groq_api_key)
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
        st.rerun()
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

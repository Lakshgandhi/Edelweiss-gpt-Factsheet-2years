import streamlit as st
import os
import numpy as np
import fitz  # PyMuPDF
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
import time

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

    html, body, [class*="css"] {
        font-family: 'Sora', sans-serif;
    }

    .stApp {
        background: #050d1a;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #070f1f !important;
        border-right: 1px solid #0d2137;
    }

    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #071628 0%, #0a2040 50%, #071628 100%);
        border: 1px solid #1a4a6b;
        border-radius: 16px;
        padding: 28px 36px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at 60% 50%, rgba(0, 180, 255, 0.06), transparent 60%);
        pointer-events: none;
    }
    .main-header h1 {
        color: #e8f4fd;
        font-size: 2rem;
        font-weight: 700;
        margin: 0 0 6px 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #5b8fa8;
        margin: 0;
        font-size: 0.95rem;
        font-weight: 300;
    }
    .badge {
        display: inline-block;
        background: rgba(0, 180, 255, 0.12);
        color: #00b4ff;
        border: 1px solid rgba(0, 180, 255, 0.25);
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 10px;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Chat messages */
    .user-msg {
        background: linear-gradient(135deg, #0d2744, #0a1f36);
        border: 1px solid #1a3d5c;
        border-radius: 14px 14px 4px 14px;
        padding: 14px 18px;
        margin: 10px 0;
        color: #c8e0f0;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .user-label {
        color: #00b4ff;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 6px;
        font-family: 'JetBrains Mono', monospace;
    }
    .bot-msg {
        background: linear-gradient(135deg, #0d1f12, #0a1a0f);
        border: 1px solid #1a3d22;
        border-radius: 14px 14px 14px 4px;
        padding: 14px 18px;
        margin: 10px 0;
        color: #c0ddc8;
        font-size: 0.95rem;
        line-height: 1.7;
    }
    .bot-label {
        color: #00e676;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 6px;
        font-family: 'JetBrains Mono', monospace;
    }
    .source-tag {
        display: inline-block;
        background: rgba(255, 215, 0, 0.08);
        color: #ffd700;
        border: 1px solid rgba(255, 215, 0, 0.2);
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.72rem;
        margin: 2px;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Stat cards */
    .stat-card {
        background: linear-gradient(135deg, #071628, #0a1f30);
        border: 1px solid #1a3d5c;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
    }
    .stat-number {
        color: #00b4ff;
        font-size: 1.8rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }
    .stat-label {
        color: #4a7a96;
        font-size: 0.8rem;
        margin-top: 4px;
        font-weight: 300;
    }

    /* Suggestion chips */
    .chip-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 12px;
    }
    .chip {
        background: rgba(0, 180, 255, 0.07);
        border: 1px solid rgba(0, 180, 255, 0.2);
        color: #5ba8c4;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        cursor: pointer;
    }

    /* Input styling */
    .stTextInput input {
        background: #071628 !important;
        border: 1px solid #1a3d5c !important;
        color: #c8e0f0 !important;
        border-radius: 10px !important;
        font-family: 'Sora', sans-serif !important;
        padding: 12px 16px !important;
    }
    .stTextInput input:focus {
        border-color: #00b4ff !important;
        box-shadow: 0 0 0 2px rgba(0, 180, 255, 0.15) !important;
    }

    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #005f8a, #007ab5) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-family: 'Sora', sans-serif !important;
        font-weight: 600 !important;
        padding: 10px 24px !important;
        transition: all 0.2s !important;
    }
    .stButton button:hover {
        background: linear-gradient(135deg, #007ab5, #009be0) !important;
        transform: translateY(-1px) !important;
    }

    /* Welcome box */
    .welcome-box {
        background: linear-gradient(135deg, #071628, #0a1a2e);
        border: 1px dashed #1a3d5c;
        border-radius: 16px;
        padding: 40px;
        text-align: center;
        color: #4a7a96;
    }
    .welcome-box h3 {
        color: #5ba8c4;
        margin-bottom: 12px;
    }

    /* Divider */
    hr { border-color: #0d2137 !important; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #050d1a; }
    ::-webkit-scrollbar-thumb { background: #1a3d5c; border-radius: 3px; }

    /* Hide Streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODEL (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')


# ─────────────────────────────────────────────
# PDF PROCESSING (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def build_knowledge_base(folder_path):
    embedder = load_embedder()
    all_chunks = []
    doc_count = 0

    pdf_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')])

    for filename in pdf_files:
        full_path = os.path.join(folder_path, filename)
        try:
            doc = fitz.open(full_path)
            text = "".join([page.get_text() for page in doc])
            doc.close()

            words = text.split()
            for i in range(0, len(words), 800):
                chunk = ' '.join(words[i:i + 1000])
                if chunk.strip():
                    all_chunks.append({'text': chunk, 'source': filename})
            doc_count += 1
        except Exception:
            continue

    if not all_chunks:
        return None, None, 0, 0

    texts = [c['text'] for c in all_chunks]
    embeddings = embedder.encode(texts, show_progress_bar=False)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings, dtype='float32'))

    return index, all_chunks, doc_count, len(all_chunks)


def retrieve_chunks(query, index, chunks, embedder, top_k=20):
    q_emb = embedder.encode([query])
    _, indices = index.search(np.array(q_emb, dtype='float32'), top_k)
    results = [chunks[i] for i in indices[0] if i < len(chunks)]
    # Ensure diverse coverage across different months
    seen_sources = {}
    diverse_results = []
    for chunk in results:
        src = chunk['source']
        seen_sources[src] = seen_sources.get(src, 0)
        if seen_sources[src] < 1:
            # Truncate each chunk to 300 words to save tokens
            words = chunk['text'].split()[:300]
            truncated = {'text': ' '.join(words), 'source': chunk['source']}
            diverse_results.append(truncated)
            seen_sources[src] += 1
    return diverse_results[:12]


def ask_groq(question, context_chunks, api_key):
    client = Groq(api_key=api_key)
    context = "\n\n".join([f"[{c['source']}]\n{c['text']}" for c in context_chunks])

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert financial analyst for Edelweiss Mutual Fund with access to 26 monthly factsheets from Feb 2024 to Feb 2026. "
                    "Answer questions using ALL the provided context chunks which come from different months. "
                    "When asked about performance over time, compare data across ALL months present in context. "
                    "Always cite which specific month/report each data point comes from. "
                    "Be precise with numbers and percentages. "
                    "Synthesize information across multiple months to give comprehensive answers."
                )
            },
            {
                "role": "user",
                "content": f"Factsheet context:\n{context}\n\nQuestion: {question}"
            }
        ],
        max_tokens=1000
    )
    return response.choices[0].message.content


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")

    groq_api_key = st.text_input(
        "🔑 Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Get your free key at console.groq.com"
    )

    folder_path = st.text_input(
        "📂 Factsheets Folder Path",
        placeholder=r"C:\Users\Name\Desktop\Factsheets",
        help="Full path to your folder containing all PDF factsheets"
    )

    load_btn = st.button("🚀 Load & Index PDFs", use_container_width=True)

    st.markdown("---")
    st.markdown("### 💡 Sample Questions")
    sample_qs = [
        "Best performing fund in 2023?",
        "Top holdings across all months?",
        "Sector allocation trend in 2024?",
        "AUM growth over 24 months?",
        "Which fund beat its benchmark?",
        "Equity vs debt allocation trend?",
    ]
    for q in sample_qs:
        st.markdown(f"<div class='chip'>💬 {q}</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        "<div style='color:#2a4a5c; font-size:0.75rem; text-align:center;'>"
        "Powered by Groq × Llama 3 × FAISS<br>Built for Edelweiss Research"
        "</div>",
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────
st.markdown("""
<div class='main-header'>
    <h1>📊 Edelweiss Factsheet GPT</h1>
    <p>AI-powered research assistant trained on your monthly factsheets</p>
    <span class='badge'>⚡ Powered by Groq · Llama 3 · FAISS</span>
</div>
""", unsafe_allow_html=True)

# ─── Session state ───
if "messages" not in st.session_state:
    st.session_state.messages = []
if "kb_ready" not in st.session_state:
    st.session_state.kb_ready = False
if "index" not in st.session_state:
    st.session_state.index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "doc_count" not in st.session_state:
    st.session_state.doc_count = 0
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

# ─── Load KB on button click ───
if load_btn:
    if not groq_api_key:
        st.error("⚠️ Please enter your Groq API key in the sidebar.")
    elif not folder_path or not os.path.exists(folder_path):
        st.error("⚠️ Folder path not found. Please check the path and try again.")
    else:
        with st.spinner("📚 Reading PDFs and building knowledge base... (1-2 mins)"):
            index, chunks, doc_count, chunk_count = build_knowledge_base(folder_path)

        if index is None:
            st.error("❌ No PDFs found in that folder. Please check the path.")
        else:
            st.session_state.index = index
            st.session_state.chunks = chunks
            st.session_state.doc_count = doc_count
            st.session_state.chunk_count = chunk_count
            st.session_state.kb_ready = True
            st.session_state.groq_key = groq_api_key
            st.success(f"✅ {doc_count} factsheets loaded! {chunk_count:,} chunks indexed.")

# ─── Stats bar (when ready) ───
if st.session_state.kb_ready:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"<div class='stat-card'><div class='stat-number'>{st.session_state.doc_count}</div><div class='stat-label'>Factsheets Loaded</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='stat-card'><div class='stat-number'>{st.session_state.chunk_count:,}</div><div class='stat-label'>Text Chunks Indexed</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='stat-card'><div class='stat-number'>{len(st.session_state.messages) // 2}</div><div class='stat-label'>Questions Asked</div></div>", unsafe_allow_html=True)
    with c4:
        st.markdown("<div class='stat-card'><div class='stat-number' style='color:#00e676'>●</div><div class='stat-label'>System Ready</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

# ─── Chat area ───
chat_container = st.container()

with chat_container:
    if not st.session_state.kb_ready:
        st.markdown("""
        <div class='welcome-box'>
            <h3>👋 Welcome to Edelweiss Factsheet GPT</h3>
            <p>To get started:</p>
            <p>1️⃣ &nbsp; Enter your <b>Groq API key</b> in the sidebar</p>
            <p>2️⃣ &nbsp; Enter the <b>folder path</b> to your factsheet PDFs</p>
            <p>3️⃣ &nbsp; Click <b>Load & Index PDFs</b></p>
            <p style='margin-top:20px; font-size:0.85rem;'>Your AI will read all 24 months of factsheets and be ready to answer anything!</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        if not st.session_state.messages:
            st.markdown("""
            <div class='welcome-box'>
                <h3>✅ Knowledge Base Ready!</h3>
                <p>Ask me anything about your Edelweiss factsheets below 👇</p>
            </div>
            """, unsafe_allow_html=True)

        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class='user-msg'>
                    <div class='user-label'>🧑 You</div>
                    {msg["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                sources_html = "".join([
                    f"<span class='source-tag'>📄 {s}</span>"
                    for s in msg.get("sources", [])
                ])
                st.markdown(f"""
                <div class='bot-msg'>
                    <div class='bot-label'>🤖 Edelweiss GPT</div>
                    {msg["content"].replace(chr(10), "<br>")}
                    <div style='margin-top:10px;'>{sources_html}</div>
                </div>
                """, unsafe_allow_html=True)

# ─── Input area ───
if st.session_state.kb_ready:
    st.markdown("---")
    col1, col2 = st.columns([5, 1])

    with col1:
        user_input = st.text_input(
            "Ask a question",
            placeholder="e.g. Which fund had the highest returns in Q3 2023?",
            label_visibility="collapsed",
            key="user_input"
        )
    with col2:
        send = st.button("Ask ➤", use_container_width=True)

    if send and user_input.strip():
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.spinner("🔍 Searching factsheets..."):
            embedder = load_embedder()
            chunks = retrieve_chunks(
                user_input,
                st.session_state.index,
                st.session_state.chunks,
                embedder
            )
            sources = list(set([c['source'] for c in chunks]))
            answer = ask_groq(user_input, chunks, st.session_state.groq_key)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })
        st.rerun()

    # Clear chat button
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat", use_container_width=False):
            st.session_state.messages = []
            st.rerun()

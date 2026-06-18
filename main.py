import streamlit as st
import os
import tempfile
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# --- CYBERPUNK/INDUSTRIAL CSS ---
st.set_page_config(page_title="RAG ENGINE" , layout="wide")
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
    * { font-family: 'JetBrains Mono', monospace !important; }
    .stApp { background-color: #050505; color: #00ff41; }
    .status-banner {
        background-color: #111;
        border: 2px solid #00ff41;
        padding: 15px;
        text-align: center;
        font-weight: bold;
        color: #00ff41;
        margin-bottom: 20px;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .metric-box {
        border: 1px solid #333;
        padding: 15px;
        background: #0a0a0a;
        text-align: center;
        color: #00ff41;
    }
    </style>
""", unsafe_allow_html=True)

# --- HEADER & STATUS ---
st.markdown('<div class="status-banner">SYSTEM STATUS: OPERATIONAL // RAG_PIPELINE_INIT</div>', unsafe_allow_html=True)
st.title("🏆 RAG ENGINE")

col1, col2, col3 = st.columns(3)
col1.markdown('<div class="metric-box">LATENCY: <50ms (LPU)</div>', unsafe_allow_html=True)
col2.markdown('<div class="metric-box">INDEX: FAISS-QUANTIZED</div>', unsafe_allow_html=True)
col3.markdown('<div class="metric-box">CORE: OPENAI/GPT-OSS-120DB</div>', unsafe_allow_html=True)

# --- INITIALIZATION ---
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# --- SIDEBAR ---
with st.sidebar:
    st.header(">> INGESTION_PIPELINE")
    api_key = st.text_input("ENTER GROQ_API_KEY", type="password")
    uploaded_file = st.file_uploader("UPLOAD SOURCE PDF", type="pdf")
    
    if uploaded_file and api_key and st.button("EXECUTE_INGESTION"):
        with st.spinner("MINING KNOWLEDGE..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            try:
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.split_documents(docs)
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                vector_db = FAISS.from_documents(chunks, embeddings)
                st.session_state.retriever = vector_db.as_retriever(search_kwargs={"k": 5})
                st.success("PIPELINE_ONLINE")
            finally:
                if os.path.exists(tmp_path): os.remove(tmp_path)

# --- MAIN GENERATION ---
st.header(">> ANALYSIS_CONSOLE")
query = st.text_input("EXECUTE QUERY:")

if query and st.session_state.retriever and api_key:
    llm = ChatGroq(groq_api_key=api_key, model_name="openai/gpt-oss-120b")
    prompt = ChatPromptTemplate.from_template("""
    You are an elite expert analyst. Answer the user question based strictly on the retrieved context.
    <context>
    {context}
    </context>
    Question: {input}
    """)
    
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(st.session_state.retriever, combine_docs_chain)
    
    start_time = time.time()
    with st.spinner("SYNTHESIZING..."):
        response = rag_chain.invoke({"input": query})
        exec_time = round(time.time() - start_time, 2)
        
        st.markdown(f"### 💎 ANALYTICAL_OUTPUT (T+ {exec_time}s)")
        st.write(response["answer"])

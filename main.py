import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- UI SETTINGS ---
st.set_page_config(page_title="GoldRush RAG", layout="wide")
st.title("🏆 Extreme GoldRush RAG Engine")

# --- SESSION STATE ---
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# --- SIDEBAR: DATA INGESTION ---
with st.sidebar:
    st.header("1. Ingestion Pipeline")
    api_key = st.text_input("Groq API Key", type="password")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    
    if uploaded_file and api_key and st.button("Initialize Engine"):
        with st.spinner("Mining knowledge..."):
            # Fixed Temp File Handling
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp.flush()
                tmp_path = tmp.name
            
            try:
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.split_documents(docs)
                
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                vector_db = FAISS.from_documents(chunks, embeddings)
                
                st.session_state.retriever = vector_db.as_retriever(search_kwargs={"k": 5})
                st.success("Pipeline Online & Indexed!")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

# --- MAIN: QUERY INTERFACE ---
st.header("2. Search & Analysis")
query = st.text_input("Execute your query:")

if query and st.session_state.retriever and api_key:
    llm = ChatGroq(groq_api_key=api_key, model_name="kimi-k2")
    
    prompt = ChatPromptTemplate.from_template("""
    You are an elite expert analyst. Answer the user question based solely on the context.
    
    <context>
    {context}
    </context>
    
    Question: {input}
    """)
    
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(st.session_state.retriever, combine_docs_chain)
    
    with st.spinner("Synthesizing results..."):
        response = rag_chain.invoke({"input": query})
        st.markdown("### 💎 Answer")
        st.write(response["answer"])

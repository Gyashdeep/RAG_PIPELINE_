import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
# Using the classic chain abstractions for stable, production-grade retrieval
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# --- UI CONFIGURATION ---
st.set_page_config(page_title="GoldRush RAG", layout="wide")
st.title("🏆 Extreme GoldRush RAG Engine")

# --- INITIALIZATION ---
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# --- SIDEBAR: PIPELINE CONFIG ---
with st.sidebar:
    api_key = st.text_input("Enter Groq API Key", type="password")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    if uploaded_file and api_key and st.button("Initialize Engine"):
        with st.spinner("Refining Knowledge Base..."):
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(uploaded_file.getvalue())
                loader = PyPDFLoader(tmp.name)
                docs = loader.load()
            
            # High-Performance Chunking
            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
            chunks = splitter.split_documents(docs)
            
            # Industry-Standard Embedding Model
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vector_db = FAISS.from_documents(chunks, embeddings)
            st.session_state.retriever = vector_db.as_retriever(search_kwargs={"k": 5})
            st.success("Pipeline Online")

# --- MAIN: GOLD RUSH GENERATION ---
query = st.text_input("Execute Query:")
if query and st.session_state.retriever and api_key:
    llm = ChatGroq(groq_api_key=api_key, model_name="llama3-70b-8192")
    
    # Classic Chain Pattern for Maximum Stability
    prompt = ChatPromptTemplate.from_template("""
    Answer the user's question based strictly on the provided context:
    <context>
    {context}
    </context>
    Question: {input}
    """)
    
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(st.session_state.retriever, combine_docs_chain)
    
    with st.spinner("Mining insights..."):
        response = rag_chain.invoke({"input": query})
        st.markdown("### Answer")
        st.write(response["answer"])

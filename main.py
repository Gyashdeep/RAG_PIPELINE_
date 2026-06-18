import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
st.set_page_config(page_title="RAG ENGINE", layout="wide")
st.title("🚀 Industrial Grade RAG Pipeline")

# Initialize Session State
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
with st.sidebar:
    st.header("1. Ingest Data")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    if uploaded_file and st.button("Process Document"):
        with st.spinner("Chunking and Embedding..."):
            # Load and Split
            loader = PyPDFLoader(uploaded_file)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(docs)
            
            # Embed and Store
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            st.session_state.vector_db = FAISS.from_documents(chunks, embeddings)
            st.success("Vector Database Indexed!")

# Main Area: Query Engine
st.header("2. Ask the GoldRush Engine")
query = st.text_input("Enter your query regarding the document:")

if query and st.session_state.vector_db:
    llm = ChatGroq(model_name="llama3-70b-8192", groq_api_key="YOUR_GROQ_API_KEY")
    qa_chain = RetrievalQA.from_chain_type(
        llm, 
        retriever=st.session_state.vector_db.as_retriever(search_kwargs={"k": 3})
    )
    
    with st.spinner("Synthesizing answer..."):
        response = qa_chain.invoke(query)
        st.markdown("### Answer")
        st.write(response["result"])

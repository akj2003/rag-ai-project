import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from utils.ui_helpers import timer

st.set_page_config(page_title="Chat with PDF", page_icon="📄")
st.title("📄 Chat with PDF (Local RAG)")

# --- CRITICAL FIX: DYNAMIC URL ---
# This checks if we are in Docker. If yes, use the host address. If no, use localhost.
ollama_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
# ---------------------------------

# 1. Upload
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # 2. Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(uploaded_file.read())
        temp_pdf_path = temp_pdf.name

    # 3. Process PDF
    with st.spinner("Processing PDF..."):
        loader = PyPDFLoader(temp_pdf_path)
        docs = loader.load()
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = splitter.split_documents(docs)
        
        # --- FIX APPLIED HERE ---
        embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url=ollama_url  # Explicitly tell it where Ollama is
        )
        # ------------------------
        
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()

    # 4. Setup LLM
    # --- FIX APPLIED HERE ---
    llm = ChatOllama(
        model="llama3.2", 
        temperature=0,
        base_url=ollama_url # Explicitly tell it where Ollama is
    )
    # ------------------------
    
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # 5. Chat Interface
    user_question = st.text_input("Ask a question about your PDF:")

    if user_question:
        with st.spinner("Thinking..."):
            with timer("Retrieval & Generation"):
                response = rag_chain.invoke({"input": user_question})
                st.write(response["answer"])
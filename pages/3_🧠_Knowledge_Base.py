import streamlit as st
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from utils.ui_helpers import timer

st.set_page_config(page_title="Knowledge Base", page_icon="🧠")
st.title("🧠 Knowledge Base Chat")

# --- CRITICAL FIX: DYNAMIC URL ---
ollama_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
# ---------------------------------

PERSIST_DIR = "/app/chroma_db_persistent"  # Docker path

# --- FIX APPLIED HERE ---
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url=ollama_url
)
# ------------------------

try:
    vector_db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    
    # Check if empty
    if not vector_db.get()['ids']:
        st.warning("📂 No Knowledge Base found. Go to 'Chat with PDF' to add documents!")
    else:
        st.success(f"✅ Connected to Knowledge Base ({len(vector_db.get()['ids'])} documents)")
        
        # --- FIX APPLIED HERE ---
        llm = ChatOllama(
            model="llama3.2", 
            temperature=0,
            base_url=ollama_url
        )
        # ------------------------
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_db.as_retriever(search_kwargs={"k": 3})
        )
        
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask your Knowledge Base..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with timer("Database Search + AI Answer"):
                    response = qa_chain.invoke(prompt)
                    answer = response['result']
                    st.markdown(answer)
            
            st.session_state.messages.append({"role": "assistant", "content": answer})

except Exception as e:
    st.error(f"Error connecting to Knowledge Base: {e}")
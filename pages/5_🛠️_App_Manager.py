import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

st.set_page_config(page_title="App Manager", page_icon="🛠️")

st.title("🛠️ App Maintenance & Troubleshooting")
st.markdown("""
**I am the expert on THIS project.** Ask me how to upgrade, fix, or extend this application.
""")

# --- 1. THE BRAIN (Context about YOUR project) ---
# We inject this context so the AI knows your specific architecture.
PROJECT_CONTEXT = """
You are the Technical Maintenance Assistant for a Local RAG AI Suite.
The user is the Lead Developer/Director.

**TECHNICAL STACK:**
- **Frontend:** Streamlit (Multipage App).
- **LLM:** Llama 3.2 via Ollama (running locally).
- **Embeddings:** nomic-embed-text.
- **Vector DB:** ChromaDB (Persistent at ./chroma_db_persistent).
- **Containerization:** Docker & Docker Compose.
- **Monitoring:** Custom psutil script in utils/system_monitor.py.

**FILE STRUCTURE:**
- Home.py (Entry point)
- pages/ (1_Chat_with_PDF.py, 2_Data_Analyst.py, 3_Knowledge_Base.py)
- utils/ (system_monitor.py)
- Dockerfile (Python 3.11 slim)
- docker-compose.yml (Orchestrates App + Ollama networking)
- requirements.txt (Dependencies)

**COMMON COMMANDS YOU SHOULD KNOW:**
- Start App (Local): `streamlit run Home.py`
- Start App (Docker): `docker-compose up --build`
- Stop Docker: `docker-compose down`
- Rebuild Container: `docker-compose up --build --force-recreate`
- Clean Database: `rm -rf chroma_db_persistent`
- Install New Lib: Add to requirements.txt -> Rebuild Docker.

**YOUR GOAL:**
Answer questions about maintaining, upgrading, or fixing this specific app. 
Be concise. Give copy-paste commands.
"""

# --- 2. CHAT INTERFACE ---
if "manager_messages" not in st.session_state:
    st.session_state.manager_messages = [
        SystemMessage(content=PROJECT_CONTEXT),
        AIMessage(content="Hello! I know the entire architecture of this app. How can I help you maintain it?")
    ]

# Display History
for msg in st.session_state.manager_messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)

# Chat Input
if prompt := st.chat_input("E.g., 'How do I add a new Python library?' or 'How do I reset the database?'"):
    
    # 1. User Message
    st.session_state.manager_messages.append(HumanMessage(content=prompt))
    st.chat_message("user").write(prompt)
    
    # 2. AI Response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing project structure..."):
            try:
                # We use the same local Llama 3.2 model
                llm = ChatOllama(model="llama3.2", temperature=0.3) 
                
                # Send the entire conversation history (Context + Chat)
                response = llm.invoke(st.session_state.manager_messages)
                
                st.write(response.content)
                st.session_state.manager_messages.append(AIMessage(content=response.content))
                
            except Exception as e:
                st.error(f"⚠️ Error connecting to AI: {e}")
                st.info("Make sure Ollama is running!")
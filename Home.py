import streamlit as st
# IMPORT THE NEW MODULE
from utils.system_monitor import show_system_monitor

st.set_page_config(
    page_title="Local AI Suite",
    page_icon="🤖",
)

# CALL THE MONITOR FUNCTION (It puts itself in the sidebar)
show_system_monitor()

st.write("# 🤖 Local AI Command Center")

st.markdown(
    """
    Welcome to your personal AI Suite, running 100% locally on your Mac.
    
    ### 👈 Select a Tool from the Sidebar
    
    * **📄 Chat with PDF:** Upload a document and ask questions (Temporary memory).
    * **📊 Data Analyst:** Upload Excel/CSV files and get Python-powered insights.
    * **🧠 Knowledge Base:** A persistent database that remembers your documents forever.
    """
)
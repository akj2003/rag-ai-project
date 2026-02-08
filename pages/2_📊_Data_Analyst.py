import streamlit as st
import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_ollama import ChatOllama

# --- IMPORT TIMER ---
from utils.ui_helpers import timer

st.set_page_config(page_title="Data Analyst", page_icon="📊")
st.title("📊 Data Analyst Agent")

uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if uploaded_file:
    # Load Data
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    st.write("### Data Preview")
    st.dataframe(df.head())

    # Initialize Agent
    llm = ChatOllama(model="llama3.2", temperature=0)
    
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        allow_dangerous_code=True
    )

    # Chat Interface
    user_question = st.text_input("Ask a question about your data:")

    if user_question:
        with st.spinner("Analyzing data..."):
            # --- TIMER ADDED HERE ---
            with timer("Python Calculation"):
                try:
                    response = agent.invoke(user_question)
                    st.write(response["output"])
                except Exception as e:
                    st.error(f"Error: {e}")
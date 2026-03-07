import streamlit as st
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_react_agent, AgentExecutor
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents import create_react_agent, AgentExecutor
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Deep Research Agent", page_icon="🌐")
st.title("🌐 Deep Research Agent (Tavily)")

# --- CONFIGURATION ---
ollama_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# 1. SETUP API KEY
# (We check for the key in 3 places: Env Var, Sidebar, or Stop)
tavily_api_key = os.getenv("TAVILY_API_KEY")
if not tavily_api_key:
    tavily_api_key = st.sidebar.text_input("Tavily API Key", type="password")

if not tavily_api_key:
    st.warning("⚠️ Please provide a Tavily API Key to search the web.")
    st.stop()

# 2. Define Tools
# We pass the key directly so it works even if not in env vars
search = TavilySearchResults(max_results=3, tavily_api_key=tavily_api_key)
tools = [search]

# 3. Define Brain (LLM)
llm = ChatOllama(model="llama3.2", temperature=0, base_url=ollama_url)

# 4. Define the ReAct Prompt (The "Instructions")
# This tells the AI how to think: "Thought" -> "Action" -> "Observation"
template = '''Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}'''

prompt = PromptTemplate.from_template(template)

# 5. Create the Agent (New Way)
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True, 
    handle_parsing_errors=True
)

# --- UI LOGIC ---
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "I am connected to the internet. Ask me anything!"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if user_input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        try:
            # The new way to run the agent is .invoke()
            response = agent_executor.invoke(
                {"input": user_input}, 
                {"callbacks": [st_callback]}
            )
            output_text = response["output"]
            st.write(output_text)
            st.session_state.messages.append({"role": "assistant", "content": output_text})
        except Exception as e:
            st.error(f"Agent Error: {e}")
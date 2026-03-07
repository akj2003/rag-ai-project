import streamlit as st
import os
import base64
import re
from PIL import Image
from io import BytesIO
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from streamlit_mermaid import st_mermaid

# --- PAGE CONFIG ---
st.set_page_config(page_title="Social Media Engine", page_icon="📱", layout="wide")
st.title("📱 AI Social Media & Branding Engine")

st.markdown("""
    <div style='background-color: #f0f2f6; padding: 15px; border-radius: 10px;'>
        <b>Personal Branding Automation:</b><br>
        Generate high-engagement thought leadership for LinkedIn. Optionally generate professional flow 
        diagrams to accompany your posts. Or, use the Instagram tab to analyze images for captions.
    </div>
    <br>
""", unsafe_allow_html=True)

# --- AI CONFIGURATION ---
ollama_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")

@st.cache_resource
def get_text_llm():
    return ChatOllama(model="llama3.2", temperature=0.7, base_url=ollama_url)

@st.cache_resource
def get_vision_llm():
    return ChatOllama(model="llava", temperature=0.7, base_url=ollama_url)

# --- STATE MANAGEMENT ---
if "linkedin_post" not in st.session_state:
    st.session_state.linkedin_post = ""
if "mermaid_code" not in st.session_state:
    st.session_state.mermaid_code = ""

# --- UI TABS ---
tab1, tab2 = st.tabs(["💼 LinkedIn (Thought Leadership)", "📸 Instagram (Visual Storytelling)"])

# ==========================================
# TAB 1: LINKEDIN GENERATOR
# ==========================================
with tab1:
    st.subheader("1. Draft the Content")
    tech_trend = st.text_input("Enter a Technology Trend or Topic (e.g., 'Agentic AI Workflows', 'RAG Pipelines'):")
    tone = st.selectbox("Select Tone:", ["Strategic & Executive", "Educational & Technical", "Inspirational / Career Growth"])
    
    if st.button("📝 Generate LinkedIn Post"):
        if not tech_trend:
            st.warning("Please enter a topic.")
        else:
            with st.spinner("Drafting your thought leadership post..."):
                try:
                    llm = get_text_llm()
                    prompt = PromptTemplate.from_template("""
                    You are a Strategic IT Leader and Associate Director with 15+ years of experience in enterprise modernization, 
                    agile transformations, and AI integration. 
                    
                    Write a highly engaging LinkedIn post about: {topic}
                    
                    Tone: {tone}
                    
                    Structure:
                    1. A strong, scroll-stopping hook (1-2 sentences).
                    2. The core insight or business value (use bullet points if appropriate).
                    3. A clear takeaway or question to drive comments and engagement.
                    4. Exactly 5-7 highly relevant hashtags at the bottom.
                    
                    Do NOT include generic greetings. Make it sound authentic, experienced, and authoritative.
                    """)
                    
                    chain = prompt | llm
                    response = chain.invoke({"topic": tech_trend, "tone": tone})
                    
                    st.session_state.linkedin_post = response.content
                    st.session_state.mermaid_code = "" # Reset any old diagrams
                    st.success("Draft Generated!")
                except Exception as e:
                    st.error(f"Generation failed: {e}")

    # Display the generated post if it exists in session state
    if st.session_state.linkedin_post:
        st.text_area("Your Post (Edit before copying):", value=st.session_state.linkedin_post, height=300)
        
        st.markdown("---")
        st.subheader("2. Visual Assets (Optional)")
        st.write("Visuals increase LinkedIn engagement by over 200%. Let the AI map this concept into a flow diagram.")
        
        # The Optional Diagram Button
        if st.button("📊 Generate Flow Diagram"):
            with st.spinner("Translating post concepts into a visual flowchart..."):
                try:
                    llm = get_text_llm()
                    diagram_prompt = PromptTemplate.from_template("""
                    Analyze the following LinkedIn post and create a Mermaid.js Flowchart (graph TD) that visually represents the core concepts, architecture, or workflow described.
                    
                    LinkedIn Post:
                    {post_content}
                    
                    Rules:
                    1. Output ONLY valid Mermaid.js code. 
                    2. Do not include markdown formatting like ```mermaid. Just the raw code starting with 'graph TD'.
                    3. Keep the diagram professional, using standard nodes and clear directional arrows.
                    4. Limit it to a maximum of 8-10 nodes so it renders cleanly.
                    """)
                    
                    diagram_chain = diagram_prompt | llm
                    raw_diagram = diagram_chain.invoke({"post_content": st.session_state.linkedin_post}).content
                    
                    # Clean the output just in case the LLM wraps it in markdown blocks
                    clean_code = re.sub(r"```mermaid\n|```", "", raw_diagram).strip()
                    st.session_state.mermaid_code = clean_code
                    
                except Exception as e:
                    st.error(f"Diagram generation failed: {e}")

    # Display the diagram if it exists
    if st.session_state.mermaid_code:
        st.success("Diagram generated! You can take a screenshot of this to attach to your post.")
        st.markdown("<div style='background-color: white; padding: 20px; border-radius: 10px;'>", unsafe_allow_html=True)
        try:
            st_mermaid(st.session_state.mermaid_code, height=400)
        except Exception as e:
            st.warning("Could not render diagram. The AI might have produced invalid syntax.")
            st.code(st.session_state.mermaid_code) # Fallback to show the code
        st.markdown("</div>", unsafe_allow_html=True)


# ==========================================
# TAB 2: INSTAGRAM CAPTION GENERATOR
# ==========================================
with tab2:
    st.subheader("Generate an Instagram Caption from an Image")
    uploaded_image = st.file_uploader("Upload a photo (JPG/PNG)", type=["jpg", "jpeg", "png"])
    
    insta_vibe = st.selectbox("Caption Vibe:", ["Professional / Behind the Scenes", "Casual & Fun", "Motivational / Fitness"])
    
    if st.button("Generate Caption"):
        if not uploaded_image:
            st.warning("Please upload an image first.")
        else:
            with st.spinner("Analyzing image and generating caption..."):
                try:
                    image_bytes = uploaded_image.read()
                    st.image(image_bytes, caption="Uploaded Image", use_column_width=True)
                    
                    base64_image = base64.b64encode(image_bytes).decode("utf-8")
                    llm = get_vision_llm()
                    
                    message = HumanMessage(
                        content=[
                            {"type": "text", "text": f"You are an expert social media manager. Look at this image and write an engaging Instagram caption for it. The vibe should be: {insta_vibe}. Include emojis and 5-8 relevant hashtags to maximize reach."},
                            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"}
                        ]
                    )
                    
                    response = llm.invoke([message])
                    
                    st.success("Caption Generated!")
                    st.text_area("Copy your caption:", value=response.content, height=200)
                except Exception as e:
                    st.error(f"Vision analysis failed: {e}. (Ensure you have pulled the 'llava' model in your backend).")